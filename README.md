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

Note: most source code comes from [LRA repo](https://github.com/google-research/long-range-arena)[2].

## Usage

Modify the configuration in `config.py` and run
```
python main.py --mode train --attn rmfa --task lra-text --device cuda:0 --former maclauformer --dotf exp --save True
```
for trying SchoenbAt approximating Softmax attention.
- mode: `train`, `eval`
- attn: `softmax`, `kernelized`, `rmfa`
- former: `maclauformer` which is with ppSBN, `transformer` which is without ppSBN
- dotf: `exp`, `inverse`, `log`, `trigh`, `sqrt` means different dot-product kernels
- task: `lra-text`, `lra-listops`, `lra-retrieval` 
- save: `True`, `False` deciding whether save the model after training 

**References**

[1] Yifan Chen, Qi Zeng, Heng Ji, and Yun Yang. Skyformer: Remodel self-attention with gaussian kernel and Nystrom method. In NeurIPS 34, 2021.

[2] Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, and Don Metzler. Long range arena : A benchmark for efficient transformers. In ICLR, 2021.
