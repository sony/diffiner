"""
Train Diffiner
"""

import argparse

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

from guided_diffusion.diffiner_util import create_model_and_diffusion

from datetime import datetime
from pytz import timezone

from speech_dataloader import data, utils
from speech_dataloader.data import load_datasets
from torchviz import make_dot
from torchinfo import summary

import torch


def main():

    dt_now = datetime.now(timezone("Asia/Tokyo"))
    dt_str = dt_now.strftime("%Y_%m%d_%H%M")
    log_dir = "./exp_logs_diffiner_train_" + dt_str
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    os.environ["OPENAI_LOGDIR"] = log_dir

    parser = create_argparser()
    parser.add_argument(
        "--complex-conv",
        action="store_true",
        default=False,
        help="Using ComplexConv2D or not. Defauls is False.",
    )
    args = parser.parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    dict_for_create_model = args_to_dict(args, model_and_diffusion_defaults().keys())
    if args.spec_type == "amplitude":
        assert "Currently, only complex spectrogram is supported."
        dict_for_create_model["image_channels"] = 1
    elif args.spec_type == "complex":
        dict_for_create_model["image_channels"] = 2
    model, diffusion = create_model_and_diffusion(**dict_for_create_model)

    # Show the model summary
    summary(
        model,
        input_size=[(1, 2, 256, 256), (1,)],
        device="cpu",
    )

    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    # FFT settings
    fft_settings = {}
    fft_settings["shiftsize"] = 256
    fft_settings["wsize"] = 512
    fft_settings["fftsize"] = 512  # height = ['fftsize']/2.0
    fft_settings["nf"] = 256  # width
    fft_settings["spec_type"] = args.spec_type

    train_dataset, args = load_datasets(parser, args, fft_settings)
    train_dataset.seq_duration = args.seq_dur
    train_dataset.random_chunks = True

    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    data = create_generator_from_dataloader(train_sampler)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_generator_from_dataloader(dataloader):

    while True:
        yield from dataloader


def create_argparser():
    defaults = dict(
        root="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        target="vocals",
        seq_dur=4.2,  # Duration of <=0.0 will result in the full audio
        samples_per_track=1,  # The number of trimming pathes per a track.
        spec_type="complex",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
