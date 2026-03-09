import os

import numpy as np
import random


import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.make_data_loader_baseline import MultimodalDamageAssessmentDatset
from model.unidaf.change import Change

from util_func.metrics import Evaluator
import util_func.lovasz_loss as L


class Trainer(object):
    """
    Trainer class that encapsulates model, optimizer, and data loading.
    It can train the model and evaluate its performance on a holdout set.
    """

    def __init__(self, cfg, fabric, model_save_path):
        """
        Initialize the Trainer with arguments from the command line or defaults.

        :param args: Argparse namespace containing:
            - dataset, train_dataset_path, val_dataset_path, etc.
            - model_type, model_param_path, resume path for checkpoint
            - learning rate, weight decay, etc.
        """
        self.source_data_name_list = cfg['source_data_name_list']
        self.holdout_data_name_list = cfg['holdout_data_name_list']

        self.cfg = cfg
        self.fabric = fabric
        self.model_save_path =model_save_path
        self.epochs = cfg['disaster_epochs']
        self.is_mapping = cfg['mapping']

        # Initialize evaluator for metrics such as accuracy, IoU, etc.
        self.evaluator = Evaluator(num_class=cfg['clf_classes'])

        # Create the deep learning model. Here we show how to use UNet or SiamCRNN.
        with self.fabric.device:
            # caformer_s36.sail_in22k_ft_in1k
            # pvt_v2_b5.in1k
            # resnet18.fb_swsl_ig1b_ft_in1k
            self.deep_model = Change(cfg['backbone'], cfg['loc_classes'], cfg['clf_classes'], cfg['hidden_dim']) 

        if cfg['resume']:
            if not os.path.isfile(cfg['resume_path']):
                raise RuntimeError("=> no checkpoint found at '{}'".format(cfg['resume_path']))
            self._load_model(self.deep_model, cfg['resume_path'])

        self.optim = optim.AdamW(self.deep_model.parameters(),
                                lr=cfg['learning_rate'],
                                weight_decay=cfg['weight_decay'])
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=15, T_mult=2)
        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optim, T_max=self.epochs)
        self.deep_model = self.fabric.setup_module(self.deep_model)
        self.optim = self.fabric.setup_optimizers(self.optim)

    def _load_model(self, model, path=None):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in checkpoint.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)

    def seg_loss(self, pred, label):
        ce_loss_clf = F.cross_entropy(pred, label, ignore_index=255)
        lovasz_loss_clf = L.lovasz_softmax(F.softmax(pred, dim=1), label, ignore=255)
        return ce_loss_clf + 0.75 * lovasz_loss_clf

    def training(self):
        """
        Main training loop that iterates over the training dataset for several steps (max_iters).
        Prints intermediate losses and evaluates on holdout dataset periodically.
        """
        best_mIoU = 0.0
        best_round = []
        torch.cuda.empty_cache()
        train_dataset = MultimodalDamageAssessmentDatset(self.cfg['root_dir'], self.source_data_name_list, crop_size=self.cfg['crop_size'], type='train')
        train_data_loader = DataLoader(train_dataset, batch_size=self.cfg['disaster_train_batch_size'], shuffle=True, pin_memory=True, num_workers=self.cfg['num_workers'], drop_last=True)
        train_data_loader = self.fabric.setup_dataloaders(train_data_loader)

        for epoch in range(self.epochs):
            # 只在主进程显示进度条
            if self.fabric.global_rank == 0:
                loop = tqdm(train_data_loader, total=len(train_data_loader))
            else:
                loop = train_data_loader
            total_loss = 0.0

            for i, data in enumerate(loop):
                pre_change_imgs, post_opt_change_imgs, post_sar_change_imgs, labels_loc, labels_clf, _ = data

                self.optim.zero_grad()

                labels1 = labels_clf.clone()
                labels1[labels1 != 1] = 0
                labels2 = labels_clf.clone()
                labels2[labels2 != 2] = 0
                labels3 = labels_clf.clone()
                labels3[labels3 != 3] = 0

                pre_change_imgs = self.fabric.to_device(pre_change_imgs)
                post_opt_change_imgs = self.fabric.to_device(post_opt_change_imgs)
                post_sar_change_imgs = self.fabric.to_device(post_sar_change_imgs)
                labels_loc = self.fabric.to_device(labels_loc).long()
                labels_clf = self.fabric.to_device(labels_clf).long()
                labels1 = self.fabric.to_device(labels1).long()
                labels2 = self.fabric.to_device(labels2).long()
                labels3 = self.fabric.to_device(labels3).long()


                valid_labels_clf = (labels_clf != 255).any()
                if not valid_labels_clf:
                    continue
                
                inputs = torch.cat([pre_change_imgs, post_opt_change_imgs, post_sar_change_imgs], dim=0)
                loc_main, clf_main, out_1, out_2, out_3 = self.deep_model(inputs) # If you use SiamCRNN 
                
                # loss loc
                loss_loc = self.seg_loss(loc_main, labels_loc)
                # loss clf
                loss_clf = self.seg_loss(clf_main, labels_clf)
                # loss clf aux
                ce_loss_clf_1 = F.mse_loss(out_1, labels1.unsqueeze(1).float())
                ce_loss_clf_2 = F.mse_loss(out_2, labels2.unsqueeze(1).float())
                ce_loss_clf_3 = F.mse_loss(out_3, labels3.unsqueeze(1).float())

                final_loss = loss_loc + loss_clf + 0.5 * ce_loss_clf_1 + ce_loss_clf_2 + 0.75 * ce_loss_clf_3

                self.fabric.backward(final_loss)
                self.optim.step()

                total_loss += self.fabric.all_reduce(final_loss, reduce_op='mean').item()

                # 更新主进程进度条
                if self.fabric.global_rank == 0:
                    loop.set_description(f'Epoch [{epoch+1}/{self.epochs}]')
                    avg_loss = total_loss / (i+1)
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    loop.set_postfix(loss=avg_loss, lr=current_lr)
               
            self.lr_scheduler.step()
            self.deep_model.eval()
            val_mIoU, final_OA, IoU_of_each_class = self.validation()

            if val_mIoU > best_mIoU:
                self.fabric.save(os.path.join(self.model_save_path, f'best_change_model.pth'), self.deep_model.state_dict())
                best_mIoU = val_mIoU
                best_round = {
                    'best iter': epoch + 1,
                    'best mIoU': str(np.around(val_mIoU * 100, 2)),
                    'best OA': str(np.around(final_OA * 100, 2)),
                    'best sub class IoU': str(np.around(IoU_of_each_class * 100, 2))
                }
            else:
                self.fabric.save(os.path.join(self.model_save_path, f'last_change_model.pth'), self.deep_model.state_dict())
            self.fabric.print(best_round)

            # 更新主进程进度条
            if self.fabric.global_rank == 0:
                loss_logger = {
                    'Disaster Epoch': epoch + 1,
                    'Disaster mIoU': val_mIoU * 100,
                    'Disaster OA': final_OA * 100,
                    'Disaster Intact': IoU_of_each_class[0] * 100,
                    'Disaster Damaged': IoU_of_each_class[1] * 100,
                    'Disaster Destroyed': IoU_of_each_class[2] * 100,
                    'Disaster Loss': avg_loss
                }
                self.fabric.log_dict(loss_logger)
            self.deep_model.train()
            self.fabric.barrier()


    def validation(self):
        self.fabric.print('---------starting validation-----------')
        self.evaluator.reset()
        dataset = MultimodalDamageAssessmentDatset(self.cfg['root_dir'], self.holdout_data_name_list, crop_size=self.cfg['crop_size'], type='test', mapping=self.is_mapping)
        val_data_loader = DataLoader(dataset, batch_size=self.cfg['disaster_eval_batch_size'], pin_memory=True, num_workers=self.cfg['num_workers'], drop_last=False)
        val_data_loader = self.fabric.setup_dataloaders(val_data_loader)
        torch.cuda.empty_cache()

        with torch.no_grad():
            # 主进程显示进度条
            if self.fabric.global_rank == 0:
                loader = tqdm(val_data_loader, desc="Validating")
            else:
                loader = val_data_loader
            for data in loader:
                pre_change_imgs, post_opt_change_imgs, post_sar_change_imgs, labels_loc, labels_clf, _ = data

                pre_change_imgs = self.fabric.to_device(pre_change_imgs)
                post_opt_change_imgs = self.fabric.to_device(post_opt_change_imgs)
                post_sar_change_imgs = self.fabric.to_device(post_sar_change_imgs)
                labels_loc = self.fabric.to_device(labels_loc).long()
                labels_clf = self.fabric.to_device(labels_clf).long()
                
                inputs = torch.cat([pre_change_imgs, post_opt_change_imgs, post_sar_change_imgs], dim=0)
                _, output_clf, _, _, _ = self.deep_model(inputs)

                output_clf = output_clf.data.cpu().numpy()
                output_clf = np.argmax(output_clf, axis=1)
                labels_clf = labels_clf.cpu().numpy()

                self.evaluator.add_batch(labels_clf, output_clf)

        final_OA = self.evaluator.Pixel_Accuracy()
        IoU_of_each_class = self.evaluator.Intersection_over_Union()
        mIoU = self.evaluator.Mean_Intersection_over_Union()

        final_OA = self.fabric.all_reduce(final_OA, reduce_op='mean').item()
        IoU_of_each_class[0] = self.fabric.all_reduce(IoU_of_each_class[0], reduce_op='mean').item()
        IoU_of_each_class[1] = self.fabric.all_reduce(IoU_of_each_class[1], reduce_op='mean').item()
        IoU_of_each_class[2] = self.fabric.all_reduce(IoU_of_each_class[2], reduce_op='mean').item()
        IoU_of_each_class[3] = self.fabric.all_reduce(IoU_of_each_class[3], reduce_op='mean').item()
        mIoU = self.fabric.all_reduce(mIoU, reduce_op='mean').mean().item()

        self.fabric.print(f'Val: OA is {np.around(100 * final_OA, 2)}, mIoU is {np.around(100 * mIoU, 2)}, sub class IoU is {np.around(100 * IoU_of_each_class, 2)}')
        
        return mIoU, final_OA, IoU_of_each_class


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

def change_main(cfg, fabric, model_save_path):
    trainer = Trainer(cfg, fabric, model_save_path)
    trainer.training()
