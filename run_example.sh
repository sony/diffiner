#!/bin/bash

# You can easily run our refiner upon the pre-processed VoiceBank+Demand results by DCUnet.

# This example defaultly runs "Diffiner+".
# Note that you need to add "--simple-diffiner" if you want to run just "Diffiner".
# Please refer to the details described in README.md and Inference.md

# Exit on error
set -e
set -o pipefail

# Main storage directory.
# If you start from downloading VoiceBank+Demand (VBD), you'll need disk space to dump the VBD and its wav.
vbd_dir=data

# Path to the python you'll use for the experiment. Defaults to the current python
python_path=python

# Start from downloading or not
stage=0  # Controls from which stage to start

# The index of GPU. If you set negative number, then inference will run on only CPU (w/o GPU).
id=0  # $CUDA_VISIBLE_DEVICES

if [[ $stage -le  0 ]]; then
  echo "Stage 0: Downloading VoiceBank+Demand and its pre-processed results by DCUnet into $vbd_dir"
  wget -c --tries=0 --read-timeout=20 https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip -P $vbd_dir
  mkdir -p $vbd_dir/logs
  unzip $vbd_dir/noisy_testset_wav.zip -d $vbd_dir >> $vbd_dir/logs/unzip_vbdtestset.log
  mkdir -p $vbd_dir/noisy_testset_wav/16kHz
  find $vbd_dir/noisy_testset_wav -name '*.wav' -printf '%f\n' | xargs -I % sox $vbd_dir/noisy_testset_wav/% -r 16000 $vbd_dir/noisy_testset_wav/16kHz/%
  wget -c --tries=0 --read-timeout=20 https://zenodo.org/record/7988790/files/proc_dcunet.tar.gz -P $vbd_dir
  tar xzvf $vbd_dir/proc_dcunet.tar.gz -C $vbd_dir >> $vbd_dir/logs/tar_procdcunet.log
  wget -c --tries=0 --read-timeout=20 https://zenodo.org/record/7988790/files/pretrained_diffiner_onVB.pt
fi

if [[ $stage -le 1 ]]; then
	echo "Stage 1: Evaluation"
  if [[ $id -lt 0 ]]; then
    $python_path scripts/run_refiner.py \
    --no-gpu \
    --root-noisy $vbd_dir/noisy_testset_wav/16kHz \
    --root-proc $vbd_dir/proc_dcunet \
    --model-path ./pretrained_diffiner_onVB.pt 
  else
    CUDA_VISIBLE_DEVICES=$id $python_path scripts/run_refiner.py \
    --root-noisy $vbd_dir/noisy_testset_wav/16kHz \
    --root-proc $vbd_dir/proc_dcunet \
    --model-path ./pretrained_diffiner_onVB.pt 
  fi
fi

