import os

import numpy as np
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

def img_loader(path):
    img = np.array(Image.open(path), np.float32)
    return img

class MultimodalDamageAssessmentDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, type='train', data_loader=img_loader, suffix='.tif', mapping=False):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type
        self.suffix = suffix
        self.crop_size = crop_size
        self.mapping = mapping

    def __transforms(self, aug, pre_img, post_opt_img, post_sar_img, label):
        if aug:
            train_transform = A.Compose([
                        A.RandomCrop(width=self.crop_size, height=self.crop_size, p=1.0),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        # A.GaussNoise(p=0.2),    # 将高斯噪声应用于输入图像。
                        A.RandomRotate90(p=0.5),
                        A.Normalize(),
                        ToTensorV2(),
                    ], additional_targets={
                        'image1': 'image',
                        'image2': 'image'
                    })
            aug = train_transform(image=pre_img, image1=post_opt_img, image2=post_sar_img, mask=label)
        else:
            val_transform = A.Compose([
                        A.Normalize(),
                        ToTensorV2(),
                    ], additional_targets={
                        'image1': 'image',
                        'image2': 'image'
                    })
            aug = val_transform(image=pre_img, image1=post_opt_img, image2=post_sar_img, mask=label)

        return aug['image'], aug['image1'], aug['image2'], aug['mask']

    def __getitem__(self, index):
        pre_img, post_opt_img, post_sar_img, clf_label = self.load_img_and_mask(index)

        if 'train' in self.data_pro_type:
            pre_img, post_opt_img, post_sar_img, clf_label = self.__transforms(True, pre_img, post_opt_img, post_sar_img, clf_label)
        else:
            pre_img, post_opt_img, post_sar_img, clf_label = self.__transforms(False, pre_img, post_opt_img, post_sar_img, clf_label)
        loc_label = clf_label.clone()
        loc_label[loc_label == 2] = 1
        loc_label[loc_label == 3] = 1

        data_idx = self.data_list[index]
        return pre_img, post_opt_img, post_sar_img, loc_label, clf_label, data_idx

    def __len__(self):
        return len(self.data_list)
        
    def load_img_and_mask(self, index):
        if not self.mapping:
            pre_path = os.path.join(self.dataset_path, 'DisasterSet-Clear', 'pre-event', self.data_list[index] + '_pre_disaster' + self.suffix)
            post_sar_path = os.path.join(self.dataset_path, 'DisasterSet-Clear', 'post-event-sar', self.data_list[index] + '_post_disaster_sar'  + self.suffix)
            post_opt_path = os.path.join(self.dataset_path, 'DisasterSet-Clear', 'post-event-opt', self.data_list[index] + '_post_disaster_opt'  + self.suffix)
            label_path = os.path.join(self.dataset_path, 'DisasterSet-Clear', 'target', self.data_list[index] + '_building_damage'  + self.suffix)
        else:
            pre_path = os.path.join(self.dataset_path, 'LA_WildFire', 'pre-event', self.data_list[index] + '_pre_disaster' + self.suffix)
            post_sar_path = os.path.join(self.dataset_path, 'LA_WildFire', 'post-event-sar', self.data_list[index] + '_post_disaster_sar'  + self.suffix)
            post_opt_path = os.path.join(self.dataset_path, 'LA_WildFire', 'post-event-opt', self.data_list[index] + '_post_disaster_opt'  + self.suffix)
            label_path = os.path.join(self.dataset_path, 'LA_WildFire', 'target', self.data_list[index] + '_building_damage'  + self.suffix)

        pre_img = self.loader(pre_path)[:,:,0:3] 
        post_opt_img = self.loader(post_opt_path)[:,:,0:3]
        post_sar_img = self.loader(post_sar_path)  
        
        # pre_img = np.stack((pre_img,)*3, axis=-1)
        post_sar_img = np.stack((post_sar_img,)*3, axis=-1)
        clf_label = self.loader(label_path)

        return pre_img, post_opt_img, post_sar_img, clf_label