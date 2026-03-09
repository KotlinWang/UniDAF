# :star2: Universal rapid building damage assessment: From global scale to local application:star2:

[![Zenodo Dataset](https://img.shields.io/badge/Zenodo-Dataset-blue)](https://zenodo.org/records/18918459) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=KotlinWang.UniDAF&left_color=%2363C7E6&right_color=%23CEE75F)

https://github.com/user-attachments/assets/dc248175-5aa0-4c4b-ba4a-d3aeb3b28761

## :newspaper:News

* **` Mar. 09th, 2025`**:  We uploaded the [weights](#dartmodel-zoo) and code of UniDAF!
* **` Mar. 06th, 2025`**: Our UniDAF project is created and all the code is uploaded! :smiley:

## :star:Overview

![overview](./assets/unidaf.jpg)
- UniDAF is the first multi-modal change detection framework for **timely post-disaster imagery** acquisition. 
- Given the interference in the imagery acquired in time after a disaster, UniDAF mitigate this problem by combining the fine-grained information of **optical imagery** and the all-weather observation ability of **SAR imagery**.
- Given the highly unpredictable nature of disasters, UniDAF autonomously learns the characteristic distribution of sudden disaster events based on **domain adaptation**.

##  :dart:Model Zoo

| Method | Background↑ | Intact↑ | Damaged↑ | Destroyed↑ | mIoU ↑ | Weights |
|------|-----------|--------|---------|-----------|--------|----------|
| UniDAF (LA-Wildfire) | 94.50 | 57.94 | 4.59 | 57.78 | 53.70 | [Download](https://drive.google.com/file/d/1Wly0jafTLFumS5n3ZD8OZPGlXPudx7jB/view?usp=sharing) |
| DamageNet (MobileNet) | 94.62       | 55.47   | 27.35    | 40.18      | 54.41 | [Download](https://drive.google.com/file/d/1nA4hW5w-32yXguO21BsJpyzk9gA5N7Nq/view?usp=sharing) |
| DamageNet (ResNet-18) | 94.78 | 53.56 | 24.98 | 45.64 | 54.74 | [Download](https://drive.google.com/file/d/1_j58YiQntcirjrq2nZp6fCVD8WTbfs5O/view?usp=sharing) |
| DamageNet (PVTv2-b3) | 95.93 | 60.42 | 32.83 | 46.83 | 59.00 | [Download](https://drive.google.com/file/d/1jbhKC7sTcDCaHPbHx38j6b53C0HwpEex/view?usp=sharing) |
| UniDAF (MobileNet)    | 94.11       | 52.34   | 38.20    | 42.20      | 56.71  | [Download](https://drive.google.com/file/d/1zhX2R8poxp2z9Gd5s8AOCon41Dzt-JfX/view?usp=sharing) |
| UniDAF (ResNet-18)    | 94.98       | 56.02   | 41.68    | 44.11      | 59.20  | [Download](https://drive.google.com/file/d/1gTMxqo1-yAZsJgUjV3X44uAyaKo7nsgm/view?usp=sharing) |
| UniDAF (PVTv2-b3)     | **96.10**  | **66.89** | **49.05** | **56.67**  | **67.18** | [Download](https://drive.google.com/file/d/1UFvcJ5R0q5AjaaLt-r5BYAjlqE2Tv0cp/view?usp=sharing) |

## :satellite:Dataset Preparation
<details open>
<div align="center">
<img src="./assets/area.jpg" height="70%" width="70%" />
</div>

- The dataset consists of 13 global-scale disaster events and one Southern California wildfire case study. It covers five types of natural disasters and two types of human-induced disasters. 
- Notably, the global-scale disaster events used in this study are constructed based on the [BRIGHT](https://essd.copernicus.org/articles/17/6217/2025/essd-17-6217-2025.html).

Please download the study datasetfrom [Zenodo](https://zenodo.org/records/18918459). After the data has been prepared, please make them have the following folder/file structure:

```
${DATASET_ROOT}   # Dataset root directory, e.g. /home/username/dataset/
│
├── DisasterSet
│   │
│   ├── pre-event
│   │    ├── bata-explosion_00000000_pre_disaster.tif
│   │    ├── bata-explosion_00000001_pre_disaster.tif
│   │    ├── bata-explosion_00000002_pre_disaster.tif
│   │    ...
│   │
│   ├── post-event-opt
│   │    ├── bata-explosion_00000000_post_disaster_opt.tif
│   │    ...
│		│
│   ├── post-event-sar
│   │    ├── bata-explosion_00000000_post_disaster_sar.tif
│   │    ...
│   │
│   └── target
│        ├── bata-explosion_00000000_building_damage.tif
│        ...
│
└── LA-WildFire
    ├── ...
```
</details>

## :computer:Installation

<details open>

**Step 0**: Clone this project and create a conda environment:

   ```shell
   git clone https://github.com/KotlinWang/UniDAF.git
   cd UniDAF
   
   conda create -n unidaf python=3.12
   conda activate unidaf
   ```

**Step 1**: Install pytorch and torchvision matching your CUDA version:

   ```shell
   pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
   ```

**Step 2**: Install requirements:

   ```shell
   pip install -r requirements.txt
   ```

</details>

## :running: Training

```shell
bash train_unidaf.sh configs/unidaf_sk_resnet18.yaml
```

If the pre-trained weights download fails, please use: 
```shell
HF_ENDPOINT=https://hf-mirror.com bash train_unidaf.sh configs/unidaf_sk_resnet18.yaml
```

## :mag: Testing

"-existing_weight_path" represents the addition of the weights to be tested.

"-inferece_saved_path" represents the path where the test result images are saved, including both the color and the original images.

```
python script/infer_unidaf.py -existing_weight_path ../your weights path -inferece_saved_path ./your save path
```

## :rocket:Supported Networks:

<details open>

| CNNs | Transformer | Mamba | UAD |
|-----|-------------|-------|-----|
| [UNet](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) | [UNetFormer](https://doi.org/10.1016/j.isprsjprs.2022.06.008) | [UrbanSSF](https://doi.org/10.1016/j.isprsjprs.2025.01.017) | [Mean Teacher](https://papers.nips.cc/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html) |
| [DeepLabv3+](https://doi.org/10.1007/978-3-030-01234-2_49) | [DamageFormer](https://ieeexplore.ieee.org/document/9883139/) | [ChangeMamba](https://ieeexplore.ieee.org/document/10565926) | [AdaptSeg](https://openaccess.thecvf.com/content_cvpr_2018/html/Tsai_Learning_to_Adapt_CVPR_2018_paper.html) |
| [UANet](https://ieeexplore.ieee.org/document/10418227) | [ChangeFormer](https://doi.org/10.1109/IGARSS46834.2022.9883686) |  | [AdvEnt](https://openaccess.thecvf.com/content_CVPR_2019/html/Vu_ADVENT_Adversarial_Entropy_Minimization_for_Domain_Adaptation_in_Semantic_Segmentation_CVPR_2019_paper.html) |
| [CFDNet](https://doi.org/10.1016/j.isprsjprs.2025.06.028) | [DamageCAT](https://doi.org/10.1016/j.ijdrr.2025.105704) |  |  |
| [SiamCRNN](https://ieeexplore.ieee.org/document/8937755) | [SiamAttUNet](https://doi.org/10.1016/j.isprsjprs.2021.02.016) |  |  |
| [ACABFNet](https://ieeexplore.ieee.org/document/9961863) |  |  |  |

</details>

## :handshake:Acknowledgement

The authors would also like to give special thanks to [BRIGHT](https://github.com/ChenHongruixuan/BRIGHT) of Capella Space, [Capella Space's Open Data Gallery](https://www.capellaspace.com/earth-observation/gallery), [Maxar Open Data Program](https://www.maxar.com/open-data) and [GoogleEarth](https://earth.google.com/web) for providing the valuable data.

## Citation

If you find this project useful in your research, please consider citing:
```
```



