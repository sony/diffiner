import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from speech_dataloader.utils import load_audio, load_info
from pathlib import Path
import numpy as np
import torch.utils.data
import argparse
import random
import torch
import torch.nn as nn
import tqdm
import glob
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Compose(object):
    """Composes several augmentation transforms.
    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio


def _augment_gain(audio, low=0.25, high=1.25):
    """Applies a random gain between `low` and `high`"""
    g = low + torch.rand(1) * (high - low)
    return audio * g


def _augment_channelswap(audio):
    """Swap channels of stereo signals with a probability of p=0.5"""
    if audio.shape[0] == 2 and torch.FloatTensor(1).uniform_() < 0.5:
        return torch.flip(audio, [0])
    else:
        return audio


def load_datasets(parser, args, fft_settings):
    """Loads the specified dataset from commandline arguments

    Returns:
        train_dataset, validation_dataset
    """
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file", type=str)

    args = parser.parse_args()
    # set output target to basename of output file

    dataset_kwargs = {
        "root": Path(args.root),
        "seq_duration": args.seq_dur,
        "input_file": args.input_file,
        "output_file": args.output_file,
        "fft_settings": fft_settings,
    }

    train_dataset = AlignedDataset(
        random_chunks=True, samples_per_track=args.samples_per_track, **dataset_kwargs
    )

    # valid_dataset = AlignedDataset(
    #     **dataset_kwargs
    # )

    # return train_dataset, valid_dataset, args
    return train_dataset, args


class AlignedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        fft_settings,
        samples_per_track,
        input_file=None,
        output_file=None,
        seq_duration=None,
        random_chunks=False,
        get_spec=True,
        sample_rate=16000,  # 44100
    ):
        """A dataset of that assumes multiple track folders
        where each track includes and input and an output file
        which directly corresponds to the the input and the
        output of the model. This dataset is the most basic of
        all datasets provided here, due to the least amount of
        preprocessing, it is also the fastest option, however,
        it lacks any kind of source augmentations or custum mixing.

        Typical use cases:

        * Source Separation (Mixture -> Target)
        * Denoising (Noisy -> Clean)
        * Bandwidth Extension (Low Bandwidth -> High Bandwidth)

        Example
        =======
        data/train/01/mixture.wav --> input
        data/train/01/vocals.wav ---> output

        """
        self.root = Path(root).expanduser()
        self.samples_per_track = samples_per_track

        self._nf = fft_settings["nf"]
        # self._shift_nf = fft_settings['shift_nf']
        self._wsize = fft_settings["wsize"]
        self._fftsize = fft_settings["fftsize"]
        self._shiftsize = fft_settings["shiftsize"]

        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.random_chunks = random_chunks
        self.get_spec = get_spec
        self.spec_type = fft_settings["spec_type"]
        if self.get_spec:
            self.window = nn.Parameter(
                torch.hann_window(self._wsize), requires_grad=False
            )

        # set the input and output files (accept glob)
        self.input_file = input_file
        self.output_file = output_file
        self.tuple_paths = sorted(list(self._get_paths()))
        if not self.tuple_paths:
            raise RuntimeError("Dataset is empty, please check parameters")

    def __getitem__(self, index):
        input_path, _ = self.tuple_paths[index // self.samples_per_track]

        input_info = load_info(input_path)
        input_info["file_path"] = input_path
        if self.random_chunks:
            duration = input_info["duration"]
            # output_info = load_info(output_path)
            # duration = min(input_info['duration'], output_info['duration'])
            start = random.uniform(0, duration - self.seq_duration)
        else:
            start = 0
            duration = self.seq_duration

        X_audio = load_audio(input_path, start=start, dur=self.seq_duration)

        # Convert to mono.
        if X_audio.shape[0] > 1:
            X_audio = X_audio[0, :].unsqueeze(0)

        # Applying STFT
        if self.get_spec:
            X_stft = torch.stft(
                X_audio,
                n_fft=self._fftsize,
                hop_length=self._shiftsize,
                window=self.window,
                center=True,
                normalized=False,
                onesided=True,
                pad_mode="reflect",
                return_complex=False,
            )

            if self.spec_type == "amplitude":
                X_spec = X_stft.pow(2).sum(-1).pow(1.0 / 2.0)
            elif self.spec_type == "complex":
                _, _nb, _nf, _ = X_stft.shape
                X_spec = torch.permute(torch.squeeze(X_stft), (2, 0, 1))
            elif self.spec_type == "power":
                X_spec = X_stft.pow(2).sum(-1)
            else:
                raise NotImplementedError(self.spec_type)

            # Confirm by visualizing spectrogram
            # tmp = X_spec[0, ...].data
            # plt.imshow(10.*np.log( tmp + 1.0 ), aspect="auto", cmap="jet")
            # plt.savefig('conf.png')

            # Cut DC component
            X_spec = X_spec[:, 1::]
            # Change the length to self._nf
            if X_spec.shape[2] >= self._nf:
                X_spec = X_spec[:, :, : self._nf]
            else:
                _b, _nb, _nf = X_spec.shape
                X_spec_ = torch.zeros(
                    (_b, _nb, self._nf), dtype=X_spec.dtype, device=X_spec.device
                )
                X_spec_[:, :, :_nf] = X_spec
                X_spec = X_spec_

            # print(X_audio.shape, X_spec.shape)  # torch.Size([1, 80000]) torch.Size([1, 256, 256])
            # return X_spec, {"y": duration}  # Y_audio
            return X_spec, input_info  # Y_audio
        else:
            return X_audio, input_info  # Y_audio

    def __len__(self):
        return len(self.tuple_paths) * self.samples_per_track

    def _get_paths(self):
        """Loads input tracks"""
        p = Path(self.root)  # , self.split)

        self.s_list = glob.glob(str(p) + "/**/*.wav", recursive=True)
        # print(len(self.s_list))  # 4000
        for input_path in tqdm.tqdm(self.s_list):
            # output_path = list(track_path.glob(self.output_file))
            if self.seq_duration is not None:
                input_info = load_info(input_path)
                # output_info = load_info(output_path[0])
                # min_duration = min(
                #     input_info['duration'], output_info['duration']
                # )
                min_duration = input_info["duration"]

                # check if both targets are available in the subfolder
                if min_duration > self.seq_duration:
                    yield input_path, None  # output_path[0]
            else:
                # raise ValueError("TBD func.")
                yield input_path, None  # output_path[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataloader for DGM SE model")
    parser.add_argument("--root", type=str, help="root path of dataset")

    parser.add_argument("--target", type=str, default="vocals")

    # I/O Parameters
    parser.add_argument(
        "--seq-dur",
        type=float,
        default=4.2,
        help="Duration of <=0.0 will result in the full audio",
    )

    parser.add_argument(
        "--samples-per-track",
        type=int,
        default=1,
        help="The number of trimming patches per a track.",
    )

    parser.add_argument("--batch-size", type=int, default=16)

    args, _ = parser.parse_known_args()

    # FFT settings
    fft_settings = {}
    fft_settings["shiftsize"] = 256
    fft_settings["wsize"] = 512
    fft_settings["fftsize"] = 512  # height = ['fftsize']/2.0
    fft_settings["nf"] = 256  # width
    fft_settings["spec_type"] = "complex"
    # fft_settings['shift_nf'] = 16

    train_dataset, args = load_datasets(parser, args, fft_settings)

    # Iterate over training dataset
    # total_training_duration = 0
    # for k in tqdm.tqdm(range(len(train_dataset))):
    #     x, dur = train_dataset[k]
    #     total_training_duration += dur  # / train_dataset.sample_rate
    # print("Total training duration (h): ", total_training_duration / 3600)
    # print("Number of train samples: ", len(train_dataset))

    # iterate over dataloader
    train_dataset.seq_duration = args.seq_dur
    train_dataset.random_chunks = True  # if true, trimming patch position randomly.

    # Test sampler
    train_sampler = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    cnt = 0
    for x, y in tqdm.tqdm(train_sampler):
        cnt += 1
        print("Iter#{}, sample shape={}".format(cnt, x.shape))
        pass
