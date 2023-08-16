# Parameters for running scripts

## Common Parameters
The following parameters are used in both the `run_refiner.py` and `stft_speech_train.py` scripts.
| Command line Argument      | Description                                                                     | Default         |
|----------------------------|---------------------------------------------------------------------------------|-----------------|
| `--image-size <int>` | Size of the complex spectrogram. Corresponds to the number of frequency bins and time frames. | `256` |
| `--num-channels <int>` | Parameter for the pre-trained diffusion model | `128` |
| `--num-res-blocks <int>` | Parameter for the pre-trained diffusion model | `3` |
| `--diffusion-steps <int>` | the number of diffusion steps when the model is trained. | `4000` |
| `--noise-scheduler <str>` | Type of noise scheduler used in model training | `linear` |
| `--use-fp16` | change inference on fp32 to fp16 | `False` |

## Parameters for `run_refiner.py` (Diffiner/Diffiner+) 

| Command line Argument      | Description                                                                     | Default         |
|----------------------------|---------------------------------------------------------------------------------|-----------------|
|`--simple-diffiner`  | switch the running mode from `Diffiner+` (default) to `Diffiner` | `False` |
|`--root-noisy <str>` | path to a directory storing the target noisy (=unprocessed) speeches |  |
|`--root-proc <str>`  | path to a directory storing the target pre-processed speeches (aiming to refine) |  |
|`--max-dur <float>`  | expected maximum duration of input speech. A longer speech than this duration will be automatically cut | `10.0` |
|`--model-path <str>`  | path to the pretrained model used to run refiner | |
| `--etas list(str)` | a list of all etas (Diffiner; $\eta_a$ and $\eta_b$, Diffiner+; $\eta_a$, $\eta_b$ $\eta_c$ and $\eta_d$). The default sets were same as those of our experiments (used in our paper) | {Diffiner; $\eta_a=0.9$ and $\eta_b=0.9$}, {Diffiner+; $\eta_a=0.4$, $\eta_b=0.6$ $\eta_c=0.0$ and $\eta_d=0.6$} |
| `--clip-denoised` | When ```--clip-denoised=True```, the clip function is applied at each diffsuion step. This technique is commonly used in the image domain and should be set to ```False``` in the complex spectrogram case.| `False` |
| `--timestep-respacing <str>` | Time step selection for execution during sampling from training time steps. See ```guided_diffusion/respace.py``` for more details. | `ddim200` |
| `--batch-size <int>` | the number of wav files to refine simultaneously. You should decide this depending on the memory size of GPU | `8` |
| `--no-gpu` | run refiner w/o GPU | `False` |

## Parameters for `stft_speech_train.py`
| Command line Argument      | Description                                                                     | Default         |
|----------------------------|---------------------------------------------------------------------------------|-----------------|
|`--weight-decay <float>`| Regularization term to prevent overfitting by penalizing large weights   | `0.0` |
|`--lr <float>` | learning rate of the optimizer (Adam)| `1e-4`|
|`--batch-size <int>`|Number of samples processed in one iteration of training|`1`|
|`--ema-rate <float>`| Rate for Exponential Moving Average (EMA)| `0.9999`|
|`--log-interval <int>`| Interval (in steps) at which training progress is logged | `10`|
|`--save-interval <int>`| Interval (in steps) at which the model checkpoints are saved| `10000`|
|`--resume-checkpoint <str>`| Path to a previous checkpoint to resume training from, if any | |
|`--seq-dur <float>`| sequence duration in seconds for the input speech | `4.2`|