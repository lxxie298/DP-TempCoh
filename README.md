# Discrete Prior-based Temporal-coherent Content Prediction for Blind Face Video Restoration

<img src='assets/model.png' />

A pytorch implementation for the paper: DP-TempCoh<br />  

### Discrete Prior-based Temporal-coherent Content Prediction for Blind Face Video Restoration

Lianxin Xie, Bingbing Zheng, Wen Xue, Yunfei Zhang, Le Jiang, Ruotao Xu, Si Wu, Hau-San Wong<br />  


-----

<a href='https://arxiv.org/abs/2501.09960'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a> 
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/fTtrLmOVq28)

-----
## 🎉 News 
- [x] [2025.01.18] Release the checkpoint used in DP-TempCoh.
- [x] [2025.01.16] 🚀🚀 Release the code of DP-TempCoh.

-----------

## Introduction

<p style="text-align: justify">
Blind face video restoration aims to restore high-fidelity details from videos subjected to complex and unknown degradations. This task poses a significant challenge of managing temporal heterogeneity while at the same time maintaining stable face attributes. In this paper, we introduce a Discrete Prior-based Temporal-Coherent content prediction transformer to address the challenge, and our model is referred to as DP-TempCoh. Specifically, we incorporate a spatial-temporal-aware content prediction module to synthesize high-quality content from discrete visual priors, conditioned on degraded video tokens. To further enhance the temporal coherence of the predicted content, a motion statistics modulation module is designed to adjust the content, based on discrete motion priors in terms of cross-frame mean and variance. As a result, the statistics of the predicted content can match with that of real videos over time. By performing extensive experiments, we verify the effectiveness of the design elements and demonstrate the superior performance of our DP-TempCoh in both synthetically and naturally degraded video restoration.
</p>


### Demo Video
https://youtu.be/fTtrLmOVq28


https://github.com/user-attachments/assets/b6bc333f-bf08-4ecc-9f70-d3e8e82e111d


-----------
<span id='Usage'/>

## Getting Started

<span id='Environment'/>

### 1.Environment</a>
Please first clone the repo and install the required environment, which can be done by running the following commands:
```shell
conda create -n DPT python=3.9.13
conda activate DPT

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
conda install -c conda-forge ffmpeg
```

### 2. Inference</a>
Download the checkpoint file from <a href="https://huggingface.co/xcc98/DP-TempCoh/tree/main">huggingface</a> and put it into directory [weights].

```
python scripts/inference_v1.py --ckpt_path weights/net_g_v1.pth -i ./examples -o results/examples
```

## Citation

If you find DP-TempCoh useful in your research or applications, please kindly cite:

```
@misc{xie2025discretepriorbasedtemporalcoherentcontent,
      title={Discrete Prior-based Temporal-coherent Content Prediction for Blind Face Video Restoration}, 
      author={Lianxin Xie and Bingbing Zheng and Wen Xue and Yunfei Zhang and Le Jiang and Ruotao Xu and Si Wu and Hau-San Wong},
      year={2025},
      eprint={2501.09960},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.09960}, 
}
```



## Acknowledgments
This project is built upon the open-source framework CodeFormer. We would like to express our sincere gratitude to the authors and contributors of CodeFormer for providing such a valuable resource. Their work has greatly influenced and supported the development of this project. You can find the original CodeFormer repository here: [CodeFormer GitHub Repository](https://github.com/sczhou/CodeFormer).
