# Evaluating Robot Policies in a World Model [\[paper\]](https://arxiv.org/abs/2506.00613) [\[website\]](https://world-model-eval.github.io/abstract) [\[demo\]](https://world-model-eval.github.io/) 

<!-- GIF gallery -->
<div style="display: flex; gap: 10px;">
  <img src="media/sweep_z.gif" alt="sweep z" width="200"/>
  <img src="media/sweep_y.gif" alt="sweep y" width="200"/>
  <img src="media/sweep_x.gif" alt="sweep x" width="200"/>
  <img src="media/gripper.gif" alt="gripper" width="200"/>
</div>

[Julian Quevedo](https://julian-q.github.io/)<sup>1</sup>, [Percy Liang](https://cs.stanford.edu/~pliang/)<sup>1</sup>, [Sherry Yang](https://sherryy.github.io/)<sup>1,2,3</sup>

Stanford University<sup>1</sup> &nbsp;&nbsp; New York University<sup>2</sup> &nbsp;&nbsp; Google DeepMind<sup>3</sup>


## Overview

This repository contains the implementation accompanying the paper [**Evaluating Robot Policies in a World Model**](https://arxiv.org/abs/2506.00613).  

News:
- 6/24/25: Dataset download script and VLM reward script released
- 6/11/25: Initial training code released

TODO:
- [x] Release dataset preparation scripts
- [ ] Release instructions for training on OpenVLA

---

## Installation

```bash
# Install PyTorch (replace cu124 with your local CUDA version)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install diffusers accelerate fire einops pytorchvideo tqdm imageio matplotlib
```

---

## Quick Start
This is how you launch training. It will train on the tiny 10-example dataset in `sample_data/`.
```bash
# Replace N with the number of available GPUs
torchrun --nproc_per_node=N train.py
```

Checkpoints and generated GIF samples will be written to `outputs/<timestamp>/`.

## Train on OpenXEmbodiment Datasets
To train on the OpenXEmbodiment datasets we used in the paper:
```bash
# We'll need tensorflow datasets and tensorflow since this code is 
# based on the original OpenXEmbodiment repo.
pip install tensorflow tensorflow_datasets
# For example, download just the Bridge dataset:
python download_data.py --dataset_name bridge
# By default the data will be written to ./converted_datasets.
# To choose your own output directory:
python download_data.py --dataset_name bridge --output_dir <your output dir>

# See download_data.py for more dataset names to choose from.
```
Then launch training with the correct dataset path:
```bash
torchrun --nproc_per_node=N train.py --dataset_dir ./converted_datasets --subset_names bridge
# Replace ./converted_datasets if your path is different.
```
You can enter a comma separated list for `subset_names` to train on a mixture of multiple datasets. For example, after downloading the `bridge` and `rt_1` datasets, you can do `--subset_names bridge,rt_1` to train on both the Bridge and RT-1 datasets.

## VLM-based reward labeling
This script demonstrates how we use GPT-4o to judge the success of generated policy rollouts:
```bash
python vlm_reward.py --video_path <path to your .mp4> --task <rollout task instructions>
```


---

## Citation

If you find this work useful, please cite:

```text
@misc{quevedo2025evaluatingrobotpoliciesworld,
      title={Evaluating Robot Policies in a World Model}, 
      author={Julian Quevedo and Percy Liang and Sherry Yang},
      year={2025},
      eprint={2506.00613},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.00613}, 
}
```

---

## Acknowledgements
- [Boyuan Chen](https://boyuan.space/) and [Kiwhan Song](https://kiwhan.dev/) for [Diffusion Forcing](https://github.com/buoyancy99/diffusion-forcing)
- [DiT](https://github.com/facebookresearch/DiT)
- [Oasis](https://github.com/etched-ai/open-oasis)

