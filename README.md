# Arcee Selective Scan Mamba Layer

> **Arcee Selective Scan Mamba Layer: **\
> Jitesh Chavan, Anand Kamat*, Rohit Lal*, Mengjia Xu\
> Paper: https://arxiv.org/abs/2511.11243

![Arcee Mamba](assets/arcee_selective_scan.png "Arcee Selective Scan")
*(a)* In a vanilla Mamba block, the selective scan is strictly causal: the state is initialized with $h^{(k)}(0)=0$, the terminal state $h^{(k)}(T)$ is discarded after producing $y$, and the next block again starts from zero. Darker cells indicate positions that have accumulated more context (later timesteps have seen a larger prefix of the sequence).  
*(b)* Arcee extends the scan to a two-port block: the terminal SSR $h^{(k)}(T)$ is reused as the initial state $h^{(k+1)}(0)$ of the next block via a differentiable boundary map, creating a recurrent state chain across depth with a valid gradient path and no change to the intra-block dynamics.


## Overview
This repo is an **atomic, layer-only implementation** of the Arcee-style **extended selective scan** with a **cross-block recurrent state chain**, introduced in [Arcee: Differentiable Recurrent State Chain for Generative Vision Modeling with Mamba SSMs](https://arxiv.org/abs/2511.11243) paper.

- Focused on the **core Mamba SSM layer + extended selective scan interface**  
- Drop-in block for your own training
- For the complete codebase including training scaffolding repository please visit [Official Arcee repository](https://github.com/JiteshChavan/Arcee)

## Installation

```bash
# 1) Create and activate env (Python 3.10.8)
conda create -n arcee-mamba python=3.10.8
conda activate arcee-mamba

# 2) Install CUDA 12.8 toolkit inside the env
conda install nvidia/label/cuda-12.8.0::cuda-toolkit

# 3) Install Python requirements
pip install -r req.txt

# 4) Install PyTorch 2.8 with CUDA 12.8 wheels (make sure cuda-toolkit and torch version match)
pip3 install torch torchvision

# 5) Build local extensions
cd causal_conv1d
pip install -e . --no-build-isolation -vvv

cd ../ArceeMamba
pip install -e . --no-build-isolation -vvv

# 6) From repo root, verify install
cd ..
bash run_test.sh   # if this runs, setup is successful

```

## Usage
![demo](assets/usage.png "Arcee Selective Scan")

## Citation

If you use this codebase, or otherwise find our work valuable, please cite:

```bibtex
@article{chavan2025arcee,
  title         = {Arcee: Differentiable Recurrent State Chain for Generative Vision Modeling with Mamba SSMs},
  author        = {Jitesh Chavan, Rohit Lal, Anand Kamat, Mengjia Xu},
  journal       = {arXiv preprint arXiv:2511.11243},
  year          = {2025},
  archivePrefix = {arXiv},
  eprint        = {2511.11243},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2511.11243},
  url           = {https://arxiv.org/abs/2511.11243}
}
```