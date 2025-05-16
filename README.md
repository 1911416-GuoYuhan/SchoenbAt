# SchoenbAt

*SchoenbAt* is a method for approximating dot-product kernelized attention under Schoenberg's theorem. It can act as a drop-in replacement for dot-product kernelized attention (e.g., exp(x·y) for Softmax attention) in any model architecture. This repository provides the official implementation of SchoenbAt and its application to Transformers.

## Installation

### Preparing the Code
To install requirements in a conda environment:
<!-- https://medium.com/@crismunozv/installing-custom-python-version-in-vertex-ai-eb9b1463e023 -->
<!-- Can also use python=3.12 -->
```
conda create -y -n schoenbat python=3.12
conda activate schoenbat
conda install torchquad -c conda-forge
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

<!-- If cannot install transformers -->
<!-- https://github.com/huggingface/transformers/issues/2831 -->
<!-- curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
Then reinstall transformers -->

Note: Specific requirements for data preprocessing are not included here.

### Preparing the Dataset

Processed dataset can be downloaded [here](https://drive.google.com/drive/folders/1rE0SjpeFKPFtgmWWjYCoIMz91UozHWWC?usp=sharing) from Skyformer[1].

Note: most source code comes from [LRA repo](https://github.com/google-research/long-range-arena)[2], and Random Maclaurin Feature is implemented based on [dp-rfs](https://github.com/joneswack/dp-rfs)[3]

## Usage

We provide a multi-head attention version of **SchoenbAt** for use in Transformers. To instantiate the `SchoenbAt` class, the following parameters are required:

- `dim`:  
  The total feature dimension.

- `head_dim`:  
  The feature dimension for each attention head.

- `num_head`:  
  The number of attention heads.

- `dropout`:  
  Dropout rate used in the attention mechanism.

- `rmfa_config`:  
  Configuration dictionary for RMFA (Random Maclaurin Feature Attention), which includes:
  
  - `nb_features`:  
    The dimensionality of the random projection.
  
  - `dotf`:  
    The type of dot-product kernel function. Optional choices include:  
    `'exp'`, `'inverse'`, `'logi'`, `'trigh'`, `'sqrt'`.

Example usage:
```python
schoenbat = SchoenbAt(
    dim=512,
    head_dim=64,
    num_head=8,
    dropout=0.1,
    rmfa_config={
        'nb_features': 256,
        'dotf': 'exp'
    }
)
output = schoenbat(Q, K, V, mask)
```

**References**

[1] Yifan Chen, Qi Zeng, Heng Ji, and Yun Yang. Skyformer: Remodel self-attention with gaussian kernel and Nystrom method. In NeurIPS 34, 2021.

[2] Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, and Don Metzler. Long range arena : A benchmark for efficient transformers. In ICLR, 2021.

[3] Wacker, Jonas and Filippone, Maurizio. Local random feature approximations of the Gaussian kernel. Procedia Computer Science, 2022, 207: 987-996.
