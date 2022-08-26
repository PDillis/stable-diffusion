import copy

import click
import os
import re
from locale import atof
from typing import Union

import torch
import numpy as np
import random
from pyperlin import FractalPerlin2D

from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

try:
    import ffmpeg
except ImportError:
    raise ImportError('ffmpeg-python not found! Install it via "pip install ffmpeg-python"')

from einops import rearrange

from torch import autocast
from contextlib import nullcontext

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from txt2img import load_model_from_config

# Helper functions -----------------------------------------------------------


def parse_resolution(s: Union[str, int], min_resolution: int = 32) -> int:
    """Parse the resolution for the synthesized image (must be a multiple of 32/min_resolution)."""
    if isinstance(s, int):
        pass
    else:
        s = int(atof(s))  # In case someone says "512.0" (I don't judge)
    s = s - s % min_resolution  # Make sure it's divisible by 32
    return max(s, min_resolution)


# TODO: Use the same noise per channel and then broadcast it to all channels.
def generate_perlin_noise(seed: int, device: torch.device, persistence: float,
                          lacunarity: int, shape: Union[tuple, list]) -> torch.tensor:
    """Generate Perlin noise with the given parameters."""
    torch_gen = torch.Generator(device=device).manual_seed(seed)
    resolutions = [(lacunarity ** i, lacunarity ** i) for i in range(1, 3 + 1)]  # TODO: different lacunarities per axes
    factors = [persistence ** i for i in range(3)]

    perlin = torch.stack([FractalPerlin2D(shape[-3:], resolutions, factors, generator=torch_gen)() for _ in range(shape[0])])

    return perlin


def parse_lacunarity(s: Union[str, int]) -> int:
    """Parse the lacunarity for the Perlin noise (must be a power of 2)."""
    lacunarity = parse_resolution(s, min_resolution=1)  # Reuse this function
    if lacunarity & (lacunarity - 1) != 0:
        print("`--lacunarity` must be a power of 2. Approximating to the nearest power of 2...")
        lacunarity = 2 ** int(np.rint(np.log2(lacunarity)))
    return lacunarity


def make_run_dir(outdir: Union[str, os.PathLike], desc: str, dry_run: bool = False) -> str:
    """Reject modernity, return to automatically create the run dir."""
    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):  # sanity check, but click.Path() should clear this one
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1  # start with 00000
    run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(run_dir)  # make sure it doesn't already exist

    # Don't create the dir if it's a dry-run
    if not dry_run:
        print('Creating output directory...')
        os.makedirs(run_dir)
    return run_dir


# Interpolate w/Perlin noise -------------------------------------------------


@click.command()
@click.pass_context
@click.option('--prompt', type=click.STRING, help='Prompt to render; best to encase in double quotes. Go wild!', required=True)
@click.option('--seed', type=click.INT, help='Global seed for random number generator.', default=0, show_default=True)
@click.option('--config', type=click.Path(exists=True), help='Path to config file which constructs the model.', default=os.path.join(os.getcwd(), 'configs', 'stable-diffusion', 'v1-inference.yaml'), show_default=True)
@click.option('--ckpt', type=click.Path(exists=True), help='Path to checkpoint of the model.', default=os.path.join('models', 'ldm', 'stable-diffusion-v1', 'model.ckpt'), show_default=True)
@click.option('--precision', type=click.Choice(['full', 'autocast']), help='Evaluate the model at this precision.', default='autocast', show_default=True)
@click.option('--scale', type=click.FLOAT, help='Unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty)).', default=7.5, show_default=True)
# Image synthesis options
@click.option('--plms', is_flag=True, help='Use PLMS sampling instead of DDIM')
@click.option('--ddim-steps', type=click.IntRange(min=1), help='Number of DDIM sampling steps', default=50, show_default=True)
@click.option('--ddim-eta', type=click.FLOAT, help='DDIM eta (eta = 0.0 corresponds to deterministic sampling', default=0.0, show_default=True)
@click.option('--num-iter', '-n-iter', type=click.INT, help='Sample this often', default=2, show_default=True)
@click.option('--deterministic', is_flag=True, help='Set the seed at the start of every frame')
# @click.option('--num-samples', '-n-samples', type=click.IntRange(min=1), help='Batch size; how many samples to produce per prompt', default=3, show_default=True)
@click.option('--height', '-H', 'height', type=parse_resolution, help='Height of image (must be divisible by 32)', default=512, show_default=True)
@click.option('--width', '-W', 'width', type=parse_resolution, help='Width of image (must be divisible by 32)', default=512, show_default=True)
@click.option('--channels', '-C', 'channels', type=click.IntRange(min=1), help='Number of latent channels', default=4, show_default=True)
@click.option('--downsample-factor', '-f', 'downsample_factor', type=click.IntRange(min=1), help='Downsample factor', default=8, show_default=True)
# Perlin noise options; helpful guide: http://libnoise.sourceforge.net/glossary/; more visual one: http://www.campi3d.com/External/MariExtensionPack/userGuide5R4v1/Understandingsomebasicnoiseterms.html
@click.option('--perlin-seed', type=click.INT, help='Seed for Perlin noise; will use `--seed` if none provided', default=None, show_default=True)
@click.option('--lacunarity', '-l', 'lacunarity', type=parse_lacunarity, help='Lacunarity of Perlin noise (how quickly the frequency increases for each successive octave)', default=2, show_default=True)  # TODO: horizontal and vertical lacunarity
@click.option('--persistence', '-p', 'persistence', type=click.FLOAT, help='Persistence of Perlin noise (how quickly amplitudes diminish for each successive octave)', default=0.5, show_default=True)
@click.option('--max-strength', 'max_perlin_strength', type=click.FLOAT, help='Maximum strength of Perlin noise', default=0.25, show_default=True)
@click.option('--min-strength', 'min_perlin_strength', type=click.FLOAT, help='Minimum strength of Perlin noise', default=0.0, show_default=True)
# Video synthesis
@click.option('--fps', type=click.IntRange(min=1), help='Frames per second', default=30, show_default=True)
@click.option('--duration-sec', type=click.FLOAT, help='Duration of video in seconds', default=15.0, show_default=True)
@click.option('--reverse-video', is_flag=True, help='Reverse the final generated video')
# Additional settings
@click.option('--outdir', type=click.STRING, help='Output directory', default=os.path.join(os.getcwd(), 'out', 'perlin_interpolation'), show_default=True)
@click.option('--description', '-desc', type=str, help='Additional description name for the directory path to save results', default=None, show_default=True)
@click.option('--verbose', is_flag=True, help='Print out additional information')
def main(ctx,
         prompt: str,
         seed: int,
         config: Union[str, os.PathLike],
         ckpt: Union[str, os.PathLike],
         precision: str,
         scale: float,
         plms: bool,
         ddim_steps: int,
         ddim_eta: float,
         num_iter: int,
         deterministic: bool,
         # num_samples: int,
         height: int,
         width: int,
         channels: int,
         downsample_factor: int,
         perlin_seed: int,
         lacunarity: int,
         persistence: float,
         max_perlin_strength: float,
         min_perlin_strength: float,
         fps: int,
         duration_sec: float,
         reverse_video: bool,
         outdir: Union[str, os.PathLike],
         description: str,
         verbose: bool):
    """Interpolate around a prompt via adding Perlin noise to the original noise."""
    # Remove seed_everything
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if verbose:
        print(f'Global seed set to {seed}')

    # Load config and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = OmegaConf.load(config)
    model = load_model_from_config(config, ckpt).to(device)

    # Sampler; use PLMS or DDIM
    sampler = PLMSSampler(model, verbose=False) if plms else DDIMSampler(model, verbose=False)

    # Precision scope
    precision_scope = autocast if precision == 'autocast' else nullcontext

    # Make the output directory
    desc = f'perlin_interpolation-seed{seed}'
    desc = f'{desc}-{description}' if description is not None else desc
    run_dir = make_run_dir(outdir, desc)

    # TODO: Optimize this! Use a larger batch size to speed this up
    # Noise shape
    batch_size = 1
    noise_shape = (batch_size, channels, height // downsample_factor, width // downsample_factor)

    # Starting noise and Perlin noise (both fixed)
    perlin_seed = seed if perlin_seed is None else perlin_seed
    perlin_noise = generate_perlin_noise(seed=perlin_seed, device=device, persistence=persistence,
                                         lacunarity=lacunarity, shape=noise_shape)
    start_code = torch.randn(noise_shape, device=device)
    start_code_copy = copy.deepcopy(start_code)

    # Video settings
    num_frames = int(np.rint(fps * duration_sec))
    n_digits = int(np.log10(num_frames)) + 1  # Number of digits for naming each frame

    r = torch.arange(start=0, end=2*np.pi, step=2*np.pi/num_frames, device=device)
    perlin_strength = torch.sin(r) * (max_perlin_strength - min_perlin_strength) + min_perlin_strength

    # Let's get this prompt
    with torch.no_grad():
        with precision_scope('cuda'):
            with model.ema_scope():
                all_samples = list()

                # Unconditional conditioning
                uc = None
                if scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [''])

                c = model.get_learned_conditioning(prompt)
                shape = noise_shape[1:]

                for idx, frame in enumerate(tqdm(range(num_frames), desc='Interpolating...', unit='frame')):
                    start_code = start_code_copy.clone() + perlin_strength[idx] * perlin_noise

                    if deterministic:
                        os.environ["PL_GLOBAL_SEED"] = str(seed)
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)

                    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                     conditioning=c,
                                                     batch_size=batch_size,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=uc,
                                                     eta=ddim_eta,
                                                     x_T=start_code,
                                                     disable_inner_tqdm=True)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = 255. * rearrange(x_samples_ddim[0].cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_samples_ddim.astype(np.uint8)).save(os.path.join(run_dir, f'frame_{idx:0{n_digits}d}.jpg'))

                    all_samples.append(x_samples_ddim)

    # Save the final video
    print('Saving video...')
    if os.name == 'nt':  # No glob pattern for Windows
        stream = ffmpeg.input(os.path.join(run_dir, f'frame_%0{n_digits}d.jpg'), framerate=fps)
    else:
        stream = ffmpeg.input(os.path.join(run_dir, 'frame_*.jpg'), pattern_type='glob', framerate=fps)
    stream = ffmpeg.output(stream, os.path.join(run_dir, 'perlin_interpolation.mp4'), crf=20, pix_fmt='yuv420p')
    ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)  # I dislike ffmpeg's console logs, so I turn them off

    # Save the reversed video apart from the original one, so the user can compare both
    if reverse_video:
        stream = ffmpeg.input(os.path.join(run_dir, 'perlin_interpolation.mp4'))
        stream = stream.video.filter('reverse')
        stream = ffmpeg.output(stream, os.path.join(run_dir, 'perlin_interpolation_reversed.mp4'), crf=20, pix_fmt='yuv420p')
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)  # ibidem


# Here we go! ----------------------------------------------------------------


if __name__ == '__main__':
    main()


# ----------------------------------------------------------------------------
