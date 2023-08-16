import argparse
import os

import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import yaml
import numpy as np
import torch as th

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    add_dict_to_argparser,
    args_to_dict,
)

from guided_diffusion.diffiner_util import create_model_and_diffusion

import torchaudio.transforms
from speech_dataloader.data import load_datasets, AlignedDataset
from informed_denoiser import get_informed_denoiser, get_improved_informed_denoiser


def prepare_detaset(args, fft_settings):
    dataset_noisy_kwargs = {
        "root": Path(args.root_noisy),
        "seq_duration": None,
        "fft_settings": fft_settings,
    }

    dataset_proc_kwargs = {
        "root": Path(args.root_proc),
        "seq_duration": None,
        "fft_settings": fft_settings,
    }

    dataset_noisy = AlignedDataset(
        random_chunks=False, samples_per_track=1, **dataset_noisy_kwargs
    )

    dataset_proc = AlignedDataset(
        random_chunks=False, samples_per_track=1, **dataset_proc_kwargs
    )

    sampler_noisy = th.utils.data.DataLoader(
        dataset_noisy, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    sampler_proc = th.utils.data.DataLoader(
        dataset_proc, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    return sampler_noisy, sampler_proc


def genwav_from_compspec(x, fft_settings):

    """
    args:
    x : (b, 1, nb, nf) amplitude spectrogram
    x_phase : (b, 2, nb, nf) complex spectrogram whose phase is used to recover the signal x
    fft_settings:
    Returns :
    weaveform : (b, timedomain_samples)
    """

    x_comp = (x[:, 0, :, :] + 1j * x[:, 1, :, :]).squeeze()

    waveform = th.istft(
        x_comp,
        n_fft=fft_settings["fftsize"],
        hop_length=fft_settings["shiftsize"],
        win_length=fft_settings["wsize"],
        window=th.hann_window(fft_settings["wsize"]),
    )

    return waveform


def main():

    parser = argparse.ArgumentParser()

    # Diffiner / Diffiner+ (default: Diffiner+)
    parser.add_argument("--simple-diffiner", action="store_true")

    # Data & Model
    parser.add_argument(
        "--root-noisy",
        type=str,
        help="Path to a directory storing the target noisy (=unprocessed) speeches.",
    )
    parser.add_argument(
        "--root-proc",
        type=str,
        help="Path to a directory storing the target pre-processed speeches (aiming to refine).",
    )
    parser.add_argument(
        "--max-dur",
        type=float,
        default=10.0,
        help="Expected maximum duration of input speech. A longer speech than this duration will be automatically cut.",
    )

    # Model
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the pretrained model used to run diffiner+.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-channels", type=int, default=128)
    parser.add_argument("--num-res-blocks", type=int, default=3)

    # Parameters
    parser.add_argument(
        "--etas", nargs="*", type=float, help="a list of variables (eta_a, b, c, and d)"
    )
    parser.add_argument("--diffusion-steps", type=int, default=4000)
    parser.add_argument("--clip-denoised", action="store_true")
    parser.add_argument(
        "--noise-scheduler",
        type=str,
        default="linear",
        help="noise scheduler which was used to train the model",
    )
    parser.add_argument(
        "--timestep-respacing",
        type=str,
        default="ddim200",
        help="Specify which time step to select and execute during sampling from the time steps used during training.",
    )

    # Inference (bigger is faster till being finished)
    parser.add_argument("--batch-size", type=int, default=8)

    # Misc
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--use-fp16", action="store_true")

    add_dict_to_argparser(parser, model_and_diffusion_defaults())
    args = parser.parse_args()

    if args.no_gpu:
        device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = "cuda:0"

    # Set Etas; if they were not input by you, then the values used in our paper will be set.
    if args.simple_diffiner and args.etas is None:
        args.eta_a = 0.9
        args.eta_b = 0.9
        args.eta_c = None
        args.eta_d = None
    elif not (args.simple_diffiner) and args.etas is None:
        args.eta_a = 0.4
        args.eta_b = 0.6
        args.eta_c = 0.0
        args.eta_d = 0.6
    elif args.simple_diffiner and args.etas is not None:
        args.eta_a = args.etas[0]
        args.eta_b = args.etas[1]
        args.eta_c = None
        args.eta_d = None
    elif not (args.simple_diffiner) and args.etas is not None:
        args.eta_a = args.etas[0]
        args.eta_b = args.etas[1]
        args.eta_c = args.etas[2]
        args.eta_d = args.etas[3]

    # FFT settings
    fft_settings = {}
    fft_settings["shiftsize"] = 256
    fft_settings["wsize"] = 512
    fft_settings["fftsize"] = 512
    sr = 16000.0
    crit_frames = args.image_size  # depending on network architecture and training
    if args.max_dur < (crit_frames * (fft_settings["shiftsize"] / sr)):
        fft_settings["nf"] = crit_frames  # width
    else:
        block_num = args.max_dur / (crit_frames * (fft_settings["shiftsize"] / sr))
        block_num = int(np.ceil(block_num))
        fft_settings["nf"] = block_num * crit_frames
    fft_settings["spec_type"] = "complex"

    logger.configure()

    # Model preparation
    logger.log("creating model and diffusion...")
    dict_for_create_model = args_to_dict(args, model_and_diffusion_defaults().keys())
    dict_for_create_model["image_channels"] = (
        2 if fft_settings["spec_type"] == "complex" else 1
    )
    model, diffusion = create_model_and_diffusion(**dict_for_create_model)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    model_kwargs = {}

    # How to denoise: Diffiner / Diffiner+
    if args.simple_diffiner:
        informed_denoiser = get_informed_denoiser(diffusion)
    else:
        informed_denoiser = get_improved_informed_denoiser(diffusion)

    # data preparation
    sampler_noisy, sampler_proc = prepare_detaset(args, fft_settings)

    # Make output dir under the "root_proc"
    if args.simple_diffiner:
        dir_name = "diffiner_etaA={}_etaB={}".format(args.eta_a, args.eta_b)
    else:
        dir_name = "diffiner+_etaA={}_etaB={}_etaC={}_etaD={}".format(
            args.eta_a, args.eta_b, args.eta_c, args.eta_d
        )
    dir_audio = os.path.join(args.root_proc, dir_name)
    dir_audio = os.path.abspath(dir_audio)
    os.makedirs(dir_audio, exist_ok=True)

    # Save settings
    with open(os.path.join(dir_audio, "config.yml"), "w") as yf:
        yaml.dump(
            {
                "Model": {
                    "path": os.path.abspath(args.model_path),
                    "image_size": args.image_size,
                    "num_channels": args.num_channels,
                    "num_res_blocks": args.num_res_blocks,
                },
                "Used data": {
                    "noisy": args.root_noisy,
                    "pre-processed": args.root_proc,
                },
                "fft_settings": fft_settings,
                "DDRM_settings": {
                    "type": "diffiner" if args.simple_diffiner else "diffiner_plus",
                    "eta_A": args.eta_a,
                    "eta_B": args.eta_b,
                    "eta_C": args.eta_c,
                    "eta_D": args.eta_d,
                    "timestep_respacing": args.timestep_respacing,
                    "diffusion_steps": args.diffusion_steps,
                    "noise_schedule": args.noise_schedule,
                },
            },
            yf,
            default_flow_style=False,
        )

    # Run diffiner / diffiner+
    n_wav_saved = 0
    for (x_noisy, info_noisy), (x_proc, info_proc) in zip(sampler_noisy, sampler_proc):

        x_noisy = x_noisy.to(device)
        x_proc = x_proc.to(device)
        assert x_noisy.shape == x_proc.shape

        n_batch, _, _, nf = x_proc.shape

        noise_stft = x_noisy - x_proc
        noise_map = (
            noise_stft.pow(2).sum(1, keepdim=True).pow(1.0 / 2.0).repeat((1, 2, 1, 1))
        )

        if not (args.simple_diffiner) and args.eta_a ** 2 + args.eta_c ** 2 > 1.0:
            print("args.eta_a={} and eta_c={}:".format(args.eta_a, args.eta_c))
            print("args.eta_a^2 + args.eta_c^2 should be less than 1.0, so skip")
            continue

        if args.simple_diffiner:
            x_deno = informed_denoiser(
                model,
                x_noisy,
                noise_map,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                etaA_ddrm=args.eta_a,
                etaB_ddrm=args.eta_b,
            )
        else:
            x_deno = informed_denoiser(
                model,
                x_noisy,
                noise_map,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                etaA=args.eta_a,
                etaB=args.eta_a,
                etaC=args.eta_c,
                inp_mask=None,
                etaD=args.eta_d,
            )
        cat_x_deno = th.cat((th.zeros(n_batch, 2, 1, nf), x_deno.to("cpu")), dim=2)
        waveform_deno = genwav_from_compspec(cat_x_deno, fft_settings)[
            :, None, :
        ]  # -> n_batch x 1 x n_sample
        for i_batch in range(n_batch):
            fname = info_proc["file_path"][i_batch].split("/")[-1]
            n_sample = min(info_proc["samples"][i_batch].item(), waveform_deno.shape[2])
            torchaudio.save(
                os.path.join(dir_audio, fname),
                waveform_deno[i_batch, :, :n_sample],
                sample_rate=16000,
                encoding="PCM_S",
                bits_per_sample=16,
            )
        n_wav_saved += n_batch


if __name__ == "__main__":
    main()
