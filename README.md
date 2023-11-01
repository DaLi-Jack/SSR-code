# Single-view 3D Scene Reconstruction with High-fidelity Shape and Texture

<p align="left">
    <a href=''>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper PDF'>
    </a>
    <a href=''>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='https://dali-jack.github.io/SSR/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
</p>


[Yixin Chen*](https://yixchen.github.io/),
[Junfeng Ni*](https://dali-jack.github.io/Junfeng-Ni/),
[Nan Jiang](https://pku.ai/author/nan-jiang/),
[Yaowei Zhang](),
[Yixin Zhu](https://yzhu.io/),
[Siyuan Huang](https://siyuanhuang.com/)

This repository is the official implementation of paper "Single-view 3D Scene Reconstruction with High-fidelity Shape and Texture".

We propose a novel framework that simultaneously recovers high-fidelity object shapes and textures from single-view images.

<div align=center>
<img src='./figures/teaser-compressed.jpg' width=80%>
</div>

## Abstract

We propose a novel framework for simultaneous high-fidelity recovery of object shapes and textures from single-view images. Our approach utilizes SSR, Single-view neural implicit Shape and Radiance field representations, leveraging explicit 3D shape supervision and volume rendering of color, depth, and surface normal images. To overcome shape-appearance ambiguity under partial observations, we introduce a two-stage learning curriculum that incorporates both 3D and 2D supervisions. A distinctive feature of our framework is its ability to generate fine-grained textured meshes while seamlessly integrating rendering capabilities into the single-view 3D reconstruction model. Beyond individual objects, our approach facilitates composing object-level representations into flexible scene representations, thereby enabling applications such as holistic scene understanding and 3D scene editing.

## Setup
```bash
conda create -n ssr python=3.8
conda activate ssr
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## Data & Checkpoints
### 1.Data
Please download [prepared data]() and unzip in the `data` folder, the resulting folder structure should be:
```
└── SSR-code
  └── data
    ├── FRONT3D
    ├── FRONT3D-demo
    ├── Pix3D
    ├── SUNRGBD
```
Because the processed by FRONT3D dataset is large, we selected some data from the test set as examples, namely **FRONT3D-demo**, to help us quickly test the results of the method.

### 2.Checkpoints
Please download our [pre-trained model]() unzip in the `output` folder, the resulting folder structure should be:
```
└── SSR-code
  └── output
    └── front3d_ckpt
      ├── model_latest.pth
      ├── out_config.yaml
    └── pix3d_ckpt
      ├── model_latest.pth
      ├── out_config.yaml
```


## Training
```bash
# NOTE: set show_rendering=False

# for 3D-FRONT dataset
python train.py --config configs/train_front3d.yaml

# for Pix3D dataset
python train.py --config configs/train_pix3d.yaml
```

## Inference
### 1.export mesh with appearance
```bash
# NOTE: set show_rendering=False, eval.export_mesh=True, eval.export_color_mesh=True

# for 3D-FRONT dataset
python inference.py --config configs/train_front3d.yaml

# for Pix3D dataset
python inference.py --config configs/train_pix3d.yaml

# for SUNRGB-D dataset
python inference_sunrgbd.py --config configs/train_sunrgbd.yaml
```

### 2.novel view synthesis
```bash
# NOTE: set show_rendering=True
python inference_rot_angle.py --config configs/train_front3d.yaml
```

### 3.scene fusion
```bash
# NOTE: set show_rendering=True, eval.fusion_scene=True
# and remember to change data.batch_size.test to object number !!!
# please carefully compare the differences between train_front3d.yaml and train_front3d_fusion.yaml
python inference_rot_angle.py --config configs/train_front3d_fusion.yaml
```

## Evaluation
```bash
# NOTE: set eval.export_mesh=True, eval.export_color_mesh=False
bash eval/evaluate.sh configs/train_front3d.yaml
```

## Citation

If you find our project useful, please consider citing us:

```tex
@inproceedings{chen2023ssr,
               title={Single-view 3D Scene Reconstruction with High-fidelity Shape and Texture},
               author={Chen, Yixin and Ni, Junfeng and Jiang, Nan and Zhang, Yaowei and Zhu, Yixin and Huang, Siyuan},
               booktitle=ThreeDV,
               year={2024}
}
```


## Acknowledgements
The code structure, some data processing and evaluation code refers to the [InstPIFu](https://github.com/GAP-LAB-CUHK-SZ/InstPIFu/tree/main). To use the pixel-align feature, we partially refer to the [PixelNeRF](https://github.com/sxyu/pixel-nerf). To leverage geometric cues, we mainly refer to the [MonoSDF](https://github.com/autonomousvision/monosdf). We thank all the authors for their great work and repos. 
