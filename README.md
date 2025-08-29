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

---

### New: [Policy Rollout Notebook](robot_evaluation.ipynb)

You can use this notebook to rollout OpenVLA in the Bridge environment.

---

## Overview

This repository contains the implementation accompanying the paper [**Evaluating Robot Policies in a World Model**](https://arxiv.org/abs/2506.00613).  

News:
- 6/24/25: Dataset download script and VLM reward script released
- 6/11/25: Initial training code released

TODO:
- [x] Release dataset preparation scripts

---

## Installation

```bash
# Install PyTorch (replace cu124 with your local CUDA version)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install transformers diffusers accelerate fire einops pytorchvideo tqdm imageio matplotlib gdown openai bs4
```

---

## Quick Start
This is how you launch training. It will train on the tiny 10-example dataset in `sample_data/`.
```bash
# Replace N with the number of available GPUs
torchrun --nproc_per_node=N train.py
```

Checkpoints and generated GIF samples will be written to `outputs/<timestamp>/`.

## Train on Open X-Embodiment Datasets
To train on the Open X-Embodiment datasets we used in the paper:
```bash
# We'll need tensorflow datasets and tensorflow since this code is 
# based on the original Open X-Embodiment repo.
pip install tensorflow tensorflow_datasets
# For example, download just the RT-1 dataset:
python download_data.py --dataset_name rt_1
# By default the data will be written to ./converted_datasets.
# To choose your own output directory:
python download_data.py --dataset_name rt_1 --output_dir <your output dir>
```
See `download_data.py` for more dataset names to choose from.


Then launch training with the correct dataset path:
```bash
torchrun --nproc_per_node=N train.py --dataset_dir ./converted_datasets --subset_names rt_1
# Replace ./converted_datasets if your path is different.
```
You can enter a comma separated list for `subset_names` to train on a mixture of multiple datasets. For example, after downloading the `rt_1` and `bridge_v2` datasets, you can do `--subset_names rt_1,bridge_v2` to train on both the RT-1 and Bridge V2 datasets.

#### Training on Bridge V2
Since Bridge V2 was not included in the original Open X-Embodiment dataset, you'll need to first download the TFDS dataset to your machine like this:
```
wget -r -np -R "index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/
```
Then, convert the dataset to our format with `python download_data.py --dataset_name bridge_v2`, changing `BRIDGE_V2_PATH` at the top of the script if necessary. Since Bridge V2 is a superset of Bridge V1, choose between either downloading `bridge` or `bridge_v2`.

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

