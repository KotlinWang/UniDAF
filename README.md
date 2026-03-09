# :star2: Universal rapid building damage assessment: From global scale to local application:star2:

https://github.com/user-attachments/assets/dc248175-5aa0-4c4b-ba4a-d3aeb3b28761

## :newspaper:News

* **` Mar. 09th, 2025`**: We updated the [weights](#:dart:Model Zoo) of all the models.
* **` Mar. 06th, 2025`**: Our UniDAF project was created and all the code is uploaded! :smiley:

## :star:Overview

![overview](./assets/unidaf.jpg)
- UniDAF is the first multi-modal change detection framework for timely post-disaster imagery acquisition. 
- In view of the interference usually accompanied by timely post-disaster imagery, we combined the fine-grained information of optical imagery and the all-weather observation ability of SAR imagery to establish DamageNet.
- The occurrence of disasters is highly uncertain, which makes the method of constructing data sets based on sudden disaster events unable to meet the requirements of timely emergency response. Based on this, our DomainStr gradually transfers to the assessment task of sudden disaster events by learning the representation of historical disaster events without additional data annotation.

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

## :see_no_evil:Visualization

<details open>

##### UAVid
<div align="center">
<img src="./assets/uavid.jpg" height="80%" width="80%" />
</div>

##### Vaihingen
<div align="center">
<img src="./assets/vaihingen.jpg" height="80%" width="80%" />
</div>

##### Potsdam
<div align="center">
<img src="./assets/potsdam.jpg" height="80%" width="80%" />
</div>
</details>

## :computer:Installation

<details open>

**Step 0**: Clone this project and create a conda environment:

   ```shell
   git clone https://github.com/KotlinWang/UrbanSSF.git
   cd UrbanSSF
   
   conda create -n urbanssf python=3.11
   conda activate urbanssf
   ```

**Step 1**: Install pytorch and torchvision matching your CUDA version:

   ```shell
   pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
   ```

**Step 2**: Install requirements:

   ```shell
   pip install -r requirements.txt
   ```

Change "collections.MutableMapping" in *xxx/envs/urbanssf/lib/python3.11/site-packages/catalyst/tools/registry.py* to "collections.abc.MutableMapping".

**Step 3**: Install Mamba:

   ```shell
   pip install mamba-ssm==1.2.0.post1
   
   pip install causal-conv1d==1.2.0.post2
   ```

**Step 4**: Replace the content of *xxx/envs/urbanssf/lib/python3.11/site-packages/mamba_ssm/ops/selective_scan_interface.py* with [selective_scan_interface.py](https://drive.google.com/file/d/1hHNJjNkV_-Uurqg07qCXaCPNhu-FESzB/view?usp=drive_link).

</details>

## :satellite:Dataset Preparation

<details open>

Download the [ISPRS Vaihingen, Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspxdatasets) and [UAVid](https://uavid.nl/) dateset.

**Vaihingen**

Generate the training set.
```shell
python tools/vaihingen_patch_split.py \
--img-dir "data/vaihingen/train_images" \
--mask-dir "data/vaihingen/train_masks" \
--output-img-dir "data/vaihingen/train/images_1024" \
--output-mask-dir "data/vaihingen/train/masks_1024" \
--mode "train" --split-size 1024 --stride 512 
```
Generate the testing set.
```shell
python tools/vaihingen_patch_split.py \
--img-dir "data/vaihingen/test_images" \
--mask-dir "data/vaihingen/test_masks_eroded" \
--output-img-dir "data/vaihingen/test/images_1024" \
--output-mask-dir "data/vaihingen/test/masks_1024" \
--mode "val" --split-size 1024 --stride 1024 \
--eroded
```
Generate the masks_1024_rgb (RGB format ground truth labels) for visualization.

````shell
python tools/vaihingen_patch_split.py \
--img-dir "data/vaihingen/test_images" \
--mask-dir "data/vaihingen/test_masks" \
--output-img-dir "data/vaihingen/test/images_1024" \
--output-mask-dir "data/vaihingen/test/masks_1024_rgb" \
--mode "val" --split-size 1024 --stride 1024 \
--gt
````

**Potsdam**
````shell
python tools/potsdam_patch_split.py \
--img-dir "data/potsdam/train_images" \
--mask-dir "data/potsdam/train_masks" \
--output-img-dir "data/potsdam/train/images_1024" \
--output-mask-dir "data/potsdam/train/masks_1024" \
--mode "train" --split-size 1024 --stride 1024 --rgb-image 
`````
As for the validation set, you can select some images from the training set to build it.

````shell
python tools/potsdam_patch_split.py \
--img-dir "data/potsdam/test_images" \
--mask-dir "data/potsdam/test_masks_eroded" \
--output-img-dir "data/potsdam/test/images_1024" \
--output-mask-dir "data/potsdam/test/masks_1024" \
--mode "val" --split-size 1024 --stride 1024 \
--eroded --rgb-image
````

```shell
python tools/potsdam_patch_split.py \
--img-dir "data/potsdam/test_images" \
--mask-dir "data/potsdam/test_masks" \
--output-img-dir "data/potsdam/test/images_1024" \
--output-mask-dir "data/potsdam/test/masks_1024_rgb" \
--mode "val" --split-size 1024 --stride 1024 \
--gt --rgb-image
```

**UAVid**
```shell
python tools/uavid_patch_split.py \
--input-dir "data/uavid/uavid_train_val" \
--output-img-dir "data/uavid/train_val/images" \
--output-mask-dir "data/uavid/train_val/masks" \
--mode 'train' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```

```shell
python tools/uavid_patch_split.py \
--input-dir "data/uavid/uavid_train" \
--output-img-dir "data/uavid/train/images" \
--output-mask-dir "data/uavid/train/masks" \
--mode 'train' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```

```shell
python tools/uavid_patch_split.py \
--input-dir "data/uavid/uavid_val" \
--output-img-dir "data/uavid/val/images" \
--output-mask-dir "data/uavid/val/masks" \
--mode 'val' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```

</details>

## :running: Training

"-c" means the path of the config, use different **config** to train different models.

```shell
python train_supervision.py -c config/uavid/urbanss-s.py
```

If the pre-trained weights download fails, please use: 
```shell
HF_ENDPOINT=https://hf-mirror.com python train_supervision.py -c config/uavid/urbanssf-s.py
```

## :mag: Testing

"-c" denotes the path of the config, Use different **config** to test different models. 

"-o" denotes the output path 

"--rgb" denotes whether to output masks in RGB format

**Vaihingen**
```
python vaihingen_test.py -c config/vaihingen/urbanssf-s.py -o fig_results/vaihingen/urbanssf-s --rgb -t 'None'
```

**Potsdam**

```
python potsdam_test.py -c config/potsdam/urbanssf-s.py -o fig_results/potsdam/urbanssf-s --rgb -t 'None'
```

**UAVid**

```
python uavid_test.py -c config/uavid/urbanssf-s.py -o fig_results/uavid/urbanssf-s --rgb -t 'None'
```

## Acknowledgement

- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
- [UNetFormer](https://github.com/WangLibo1995/GeoSeg)
- [Vision Mamba](https://github.com/hustvl/Vim)

## Citation

If you find this project useful in your research, please consider citing：

```
@article{WANG2025824,
title = {Accurate semantic segmentation of very high-resolution remote sensing images considering feature state sequences: From benchmark datasets to urban applications},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {220},
pages = {824-840},
year = {2025},
issn = {0924-2716}
author = {Zijie Wang and Jizheng Yi and Aibin Chen and Lijiang Chen and Hui Lin and Kai Xu}
}
```

