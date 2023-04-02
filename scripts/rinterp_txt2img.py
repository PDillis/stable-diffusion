import click
import os
import re
from locale import atof
from typing import Union, Optional

import torch
import numpy as np
import random
import scipy.ndimage

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
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

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


# Interpolate w/random noise -------------------------------------------------


@click.command()
@click.pass_context
@click.option('--prompt', type=click.STRING, help='Prompt to render; best to encase in double quotes. Go wild!', required=True)
@click.option('--seed', type=click.INT, help='Global seed for random number generator.', default=0, show_default=True)
@click.option('--config', type=click.Path(exists=True), help='Path to config file which constructs the model.', default=os.path.join(os.getcwd(), 'configs', 'stable-diffusion', 'v1-inference.yaml'), show_default=True)
@click.option('--ckpt', type=click.Path(exists=True), help='Path to checkpoint of the model.', default=os.path.join('models', 'ldm', 'stable-diffusion-v1', 'model.ckpt'), show_default=True)
@click.option('--half', is_flag=True, help='Use half precision/fp16.')
@click.option('--precision', type=click.Choice(['full', 'autocast']), help='Evaluate the model at this precision.', default='autocast', show_default=True)
@click.option('--scale', type=click.FLOAT, help='Unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty)).', default=7.5, show_default=True)
# Image synthesis options
@click.option('--dpm-solver', is_flag=True, help='Use DPM sampling instead of DDIM')
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
# Video synthesis
@click.option('--fps', type=click.IntRange(min=1), help='Frames per second', default=30, show_default=True)
@click.option('--duration-sec', type=click.FLOAT, help='Duration of video in seconds', default=15.0, show_default=True)
@click.option('--reverse-video', is_flag=True, help='Reverse the final generated video')
# Additional settings
@click.option('--outdir', type=click.STRING, help='Output directory', default=os.path.join(os.getcwd(), 'out', 'random_interpolation'), show_default=True)
@click.option('--description', '-desc', type=str, help='Additional description name for the directory path to save results', default=None, show_default=True)
@click.option('--verbose', is_flag=True, help='Print out additional information')
def main(ctx,
         prompt: str,
         seed: int,
         config: Union[str, os.PathLike],
         ckpt: Union[str, os.PathLike],
         half: bool,
         precision: str,
         scale: float,
         dpm_solver: bool,
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
         fps: int,
         duration_sec: float,
         reverse_video: bool,
         outdir: Union[str, os.PathLike],
         description: str,
         verbose: bool,
         smoothing_sec: Optional[float] = 3.0):  # for Gaussian blur; won't be a command-line parameter, change at own risk
    """Interpolate around a prompt via a random interpolation in the latent space."""
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

    # Precision scope
    precision_scope = autocast if precision == 'autocast' else nullcontext

    # Make the output directory
    desc = f'random_interpolation-seed{seed}'
    desc = f'{desc}-{description}' if description is not None else desc
    run_dir = make_run_dir(outdir, desc)

    # Video settings
    num_frames = int(np.rint(fps * duration_sec))
    n_digits = int(np.log10(num_frames)) + 1  # Number of digits for naming each frame

    # Noise shape
    batch_size = 1  # TODO: Make this faster by increasing the 'batch size'
    noise_shape = (num_frames, channels, height // downsample_factor, width // downsample_factor)

    start_code = np.random.randn(*noise_shape)
    start_code = scipy.ndimage.gaussian_filter(start_code, sigma=[smoothing_sec * fps, 0, 0, 0], mode='wrap')
    start_code /= np.sqrt(np.mean(np.square(start_code)))  # Normalize; necessary?
    start_code = torch.from_numpy(start_code).to(device)

    config = OmegaConf.load(config)
    model = load_model_from_config(config, ckpt).to(device)

    # Half precision
    if half:
        model = model.half()
        start_code = start_code.half()

    # Sampler; use DPM, PLMS, or DDIM
    if dpm_solver:
        print('Using DPM solver sampler...')
        sampler = DPMSolverSampler(model, verbose=False)
    elif plms:
        print('Using PLMS sampler...')
        sampler = PLMSSampler(model, verbose=False)
    else:
        print('Using DDIM sampler...')
        sampler = DDIMSampler(model, verbose=False)


    # Let's get this prompt
    with torch.no_grad():
        with precision_scope('cuda'):
            with model.ema_scope():
                # Unconditional conditioning
                uc = None
                if scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [''])

                c = model.get_learned_conditioning(prompt)
                shape = noise_shape[1:]

                for idx, frame in enumerate(tqdm(range(num_frames), desc='Interpolating...', unit='frame')):

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
                                                     x_T=start_code[idx].unsqueeze(0),
                                                     disable_inner_tqdm=True)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = 255. * rearrange(x_samples_ddim[0].cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_samples_ddim.astype(np.uint8)).save(os.path.join(run_dir, f'frame_{idx:0{n_digits}d}.jpg'))

    # Save the final video
    print('Saving video...')
    ffmpeg_command = r'/usr/bin/ffmpeg' if os.name != 'nt' else r'C:\\Ffmpeg\\bin\\ffmpeg.exe'
    stream = ffmpeg.input(os.path.join(run_dir, f'frame_%0{n_digits}d.jpg'), framerate=fps)
    stream = ffmpeg.output(stream, os.path.join(run_dir, 'random_interpolation.mp4'), crf=20, pix_fmt='yuv420p')
    ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, cmd=ffmpeg_command)

    # Save the reversed video apart from the original one, so the user can compare both
    if reverse_video:
        stream = ffmpeg.input(os.path.join(run_dir, 'random_interpolation.mp4'))
        stream = stream.video.filter('reverse')
        stream = ffmpeg.output(stream, os.path.join(run_dir, 'random_interpolation_reversed.mp4'), crf=20, pix_fmt='yuv420p')
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)  # ibidem


# Here we go! ----------------------------------------------------------------


if __name__ == '__main__':
    main()


# ----------------------------------------------------------------------------
