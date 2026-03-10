import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

import argparse
import os
import random

import numpy as np


import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.make_data_loader_baseline import MultimodalDamageAssessmentDatset
from model.MeanTeacher.network import MTNetwork
from datetime import datetime
from model.MeanTeacher.utils.init_func import init_weight, group_weight
from model.MeanTeacher.engine.lr_policy import WarmUpPolyLR

from util_func.metrics import Evaluator
import util_func.lovasz_loss as L
import torch.nn as nn

import lightning as lg
from lightning.fabric.loggers import CSVLogger


class Trainer(object):
    """
    Trainer class that encapsulates model, optimizer, and data loading.
    It can train the model and evaluate its performance on a holdout set.
    """

    def __init__(self, args):
        """
        Initialize the Trainer with arguments from the command line or defaults.

        :param args: Argparse namespace containing:
            - dataset, train_dataset_path, holdout_dataset_path, etc.
            - model_type, model_param_path, resume path for checkpoint
            - learning rate, weight decay, etc.
        """
        self.args = args
        gpu_ids = args.gpu_ids.split(',')
        num_devices = len(gpu_ids)
        self.epochs = args.epochs
        
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_path = os.path.join(
            args.model_param_path,
            args.dataset,
            args.model_type + '_' + now_str
        )

        self.fabric = lg.Fabric(
            accelerator="auto",
            devices=num_devices,
            strategy='ddp' if num_devices > 1 else 'auto',
            precision='bf16-mixed',
            loggers=[CSVLogger(self.model_save_path,
                               name=f"{args.dataset}-{args.model_type}",
                               flush_logs_every_n_steps=1)]
        )
        self.fabric.launch()
        self.fabric.seed_everything(3047 + self.fabric.global_rank)

        if self.fabric.global_rank == 0:
            os.makedirs(self.model_save_path, exist_ok=True)

        # Initialize evaluator for metrics such as accuracy, IoU, etc.
        self.evaluator = Evaluator(num_class=4)

        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
        self.criterion_csst = nn.MSELoss(reduction='mean')
        # Create the deep learning model. Here we show how to use UNet or SiamCRNN.
        with self.fabric.device:
            self.deep_model = MTNetwork()

        # group weight and config optimizer
        base_lr = self.args.learning_rate_seg
        
        self.optimizer_l = optim.AdamW(self.deep_model.branch1.parameters(),
                                 lr=base_lr,
                                 weight_decay=args.weight_decay)
        self.optimizer_r = optim.AdamW(self.deep_model.branch2.parameters(),
                                 lr=base_lr,
                                 weight_decay=args.weight_decay)

        # config lr policy
        self.lr_scheduler_l = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_l, T_0=15, T_mult=2
        )
        self.lr_scheduler_r = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_r, T_0=15, T_mult=2
        )
        
        self.deep_model = self.fabric.setup_module(self.deep_model)
        self.optimizer_l = self.fabric.setup_optimizers(self.optimizer_l)
        self.optimizer_r = self.fabric.setup_optimizers(self.optimizer_r)

    def training(self):
        """
        Main training loop that iterates over the training dataset for several steps (max_iters).
        Prints intermediate losses and evaluates on holdout dataset periodically.
        """
        best_mIoU = 0.0
        best_mIoU_fs = 0.0
        best_mIoU_mix = 0.0

        best_round = []
        torch.cuda.empty_cache()
        train_dataset_source = MultimodalDamageAssessmentDatset(
            self.args.root_path, self.args.source_data_name_list,
            crop_size=self.args.crop_size, type='train'
        )
        train_dataset_target = MultimodalDamageAssessmentDatset(
            self.args.root_path, self.args.target_data_name_list,
            crop_size=self.args.crop_size, type='train'
        )

        loader_src = self.fabric.setup_dataloaders(
            DataLoader(train_dataset_source,
                       batch_size=self.args.train_batch_size,
                       shuffle=True, num_workers=self.args.num_workers)
        )
        loader_tgt = self.fabric.setup_dataloaders(
            DataLoader(train_dataset_target,
                       batch_size=self.args.train_batch_size,
                       shuffle=True, num_workers=self.args.num_workers)
        )

        for epoch in range(self.epochs):
            self.deep_model.train()

            src_iter = iter(loader_src)
            tgt_iter = iter(loader_tgt)
            iters = max(len(loader_src), len(loader_tgt))

            pbar = tqdm(range(iters)) if self.fabric.global_rank == 0 else range(iters)

            running_losses = {'sup': 0.0, 'csst': 0.0, 'sup_t': 0.0}

            for i in pbar:
                src = next(src_iter)
                try:
                    tgt = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(loader_tgt)
                    tgt = next(tgt_iter)

                self.optimizer_l.zero_grad()
                self.optimizer_r.zero_grad()

                pre_s, post_s, sar_s, loc_gt, clf_gt, _ = src
                pre_t, post_t, sar_t, _, _, _ = tgt

                pre_s = self.fabric.to_device(pre_s)
                post_s = self.fabric.to_device(post_s)
                sar_s = self.fabric.to_device(sar_s)
                loc_gt = self.fabric.to_device(loc_gt).long()
                clf_gt = self.fabric.to_device(clf_gt).long()

                pre_t = self.fabric.to_device(pre_t)
                post_t = self.fabric.to_device(post_t)
                sar_t = self.fabric.to_device(sar_t)

                valid_labels_clf = (clf_gt != 255).any()
                if not valid_labels_clf:
                    continue
                
                input_data = torch.cat([pre_s, post_s, sar_s], dim=0)
                input_data_unsup = torch.cat([pre_t, post_t, sar_t], dim=0)
                
                s_sup_pred, t_sup_pred = self.deep_model(input_data, step=1, cur_iter=epoch)
                out_loc1_sup, out_clf1_sup, o11_sup, o21_sup, o31_sup = s_sup_pred
                out_loc2_sup, out_clf2_sup, o12_sup, o22_sup, o32_sup = t_sup_pred
                    
                ### Mean Teacher loss ###
                # Perform semi-supervised learning after initial 10000 iterations
                if epoch < 30:
                    csst_loss = 0
                else:
                    s_unsup_pred, t_unsup_pred = self.deep_model(input_data_unsup, step=2, cur_iter=epoch)
                    out_loc1_unsup, out_clf1_unsup, o11_unsup, o21_unsup, o31_unsup = s_unsup_pred
                    out_loc2_unsup, out_clf2_unsup, o12_unsup, o22_unsup, o32_unsup = t_unsup_pred
                    
                    s_loc = torch.cat([out_loc1_sup, out_loc1_unsup], dim=0)
                    s_clf = torch.cat([out_clf1_sup, out_clf1_unsup], dim=0)
                    s_aux = torch.cat([o11_sup, o21_sup, o31_sup, o11_unsup, o21_unsup, o31_unsup], dim=0)
                    t_loc = torch.cat([out_loc2_sup, out_loc2_unsup], dim=0)
                    t_clf = torch.cat([out_clf2_sup, out_clf2_unsup], dim=0)
                    t_aux = torch.cat([o12_sup, o22_sup, o32_sup, o12_unsup, o22_unsup, o32_unsup], dim=0)

                    softmax_s_loc = F.softmax(s_loc, 1)
                    softmax_s_clf = F.softmax(s_clf, 1)
                    softmax_s_aux = F.softmax(s_aux, 1)
                    
                    softmax_t_loc = F.softmax(t_loc, 1)
                    softmax_t_clf = F.softmax(t_clf, 1)
                    softmax_t_aux = F.softmax(t_aux, 1)
                    
                    csst_loss_loc = self.criterion_csst(softmax_s_loc, softmax_t_loc.detach())
                    csst_loss_clf = self.criterion_csst(softmax_s_clf, softmax_t_clf.detach())
                    csst_loss_aux = self.criterion_csst(softmax_s_aux, softmax_t_aux.detach())
                    csst_loss = csst_loss_loc + csst_loss_clf + 0.75*csst_loss_aux
                csst_loss = csst_loss * 100

                ### Supervised loss For Student ###
                loss_sup_loc = self.criterion(out_loc1_sup, loc_gt) +  \
                    0.75 * L.lovasz_softmax(F.softmax(out_loc1_sup, dim=1), loc_gt, ignore=255)
                loss_sup_clf = self.criterion(out_clf1_sup, clf_gt) +  \
                    0.75 * L.lovasz_softmax(F.softmax(out_clf1_sup, dim=1), clf_gt, ignore=255)
                loss_sup_aux = (
                        0.5 * F.mse_loss(o11_sup, (clf_gt == 1).float().unsqueeze(1)) +
                        1.0 * F.mse_loss(o21_sup, (clf_gt == 2).float().unsqueeze(1)) +
                        0.75 * F.mse_loss(o31_sup, (clf_gt == 3).float().unsqueeze(1))
                    )
                loss_sup = loss_sup_loc + loss_sup_clf + loss_sup_aux
                
                ### Supervised loss For Teracher. No Backward ###
                loss_sup_t_loc = self.criterion(out_loc2_sup, loc_gt) +  \
                    0.75 * L.lovasz_softmax(F.softmax(out_loc2_sup, dim=1), loc_gt, ignore=255)
                loss_sup_t_clf = self.criterion(out_clf2_sup, clf_gt) +  \
                    0.75 * L.lovasz_softmax(F.softmax(out_clf2_sup, dim=1), clf_gt, ignore=255)
                loss_sup_t_aux = (
                        0.5 * F.mse_loss(o12_sup, (clf_gt == 1).float().unsqueeze(1)) +
                        1.0 * F.mse_loss(o22_sup, (clf_gt == 2).float().unsqueeze(1)) +
                        0.75 * F.mse_loss(o32_sup, (clf_gt == 3).float().unsqueeze(1))
                    )
                loss_sup_t = loss_sup_t_loc + loss_sup_t_clf + loss_sup_t_aux

                loss = loss_sup + csst_loss
                self.fabric.backward(loss)
                self.optimizer_l.step()

                running_losses['sup'] += loss_sup.detach().cpu().item()
                running_losses['csst'] += csst_loss.detach().cpu().item() if epoch >= 30 else 0
                running_losses['sup_t'] += loss_sup_t.detach().cpu().item()
                if self.fabric.global_rank == 0:
                    avg_sup = running_losses['sup'] / (i + 1)
                    avg_csst = running_losses['csst'] / (i + 1)
                    avg_sup_t = running_losses['sup_t'] / (i + 1)
                    lr = self.lr_scheduler_l.get_last_lr()[0]
                    pbar.set_description(f'Epoch [{epoch+1}/{self.epochs}]')
                    pbar.set_postfix(sup_loss=avg_sup, csst_loss=avg_csst, sup_t_loss=avg_sup_t, lr=lr)
            
            self.lr_scheduler_l.step()

            val_mIoU, val_OA, val_IoU_of_each_class = self.validation()

            if val_mIoU > best_mIoU:
                self.fabric.save(os.path.join(self.model_save_path, f'best_model.pth'), self.deep_model.state_dict())
                best_mIoU = val_mIoU
                best_round = {
                    'best iter': epoch + 1,
                    'best mIoU': str(np.around(val_mIoU * 100, 2)),
                    'best OA': str(np.around(val_OA * 100, 2)),
                    'best sub class IoU': str(np.around(val_IoU_of_each_class * 100, 2))
                }
            else:
                self.fabric.save(os.path.join(self.model_save_path, f'last_model.pth'), self.deep_model.state_dict())
            self.fabric.print(best_round)
            # self.deep_model.train()
            self.fabric.barrier()


    def validation(self):
        print('---------starting validation-----------')
        self.evaluator.reset()
        dataset = MultimodalDamageAssessmentDatset(self.args.root_path, self.args.holdout_data_name_list, crop_size=1024, type='test')
        val_data_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size, num_workers=1, drop_last=False)
        torch.cuda.empty_cache()

        self.deep_model.eval()
        with torch.no_grad():
            # 主进程显示进度条
            if self.fabric.global_rank == 0:
                loader = tqdm(val_data_loader, desc="Validating")
            else:
                loader = val_data_loader

            for data in loader:
                pre_opt_change_imgs, post_opt_change_imgs, post_sar_change_imgs, labels_loc, labels_clf, _ = data

                pre_opt_change_imgs = pre_opt_change_imgs.cuda()
                post_opt_change_imgs = post_opt_change_imgs.cuda()
                post_sar_change_imgs = post_sar_change_imgs.cuda()
                labels_loc = labels_loc.cuda().long()
                labels_clf = labels_clf.cuda().long()
                
                inputs = torch.cat([pre_opt_change_imgs, post_opt_change_imgs, post_sar_change_imgs], dim=0)
                output_clf = self.deep_model(inputs)  # if you use UNet    

                output_clf = output_clf.data.cpu().numpy()
                output_clf = np.argmax(output_clf, axis=1)
                labels_clf = labels_clf.cpu().numpy()

                self.evaluator.add_batch(labels_clf, output_clf)

        
        final_OA = self.evaluator.Pixel_Accuracy()
        IoU_of_each_class = self.evaluator.Intersection_over_Union()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        print(f'OA is {100 * final_OA}, mIoU is {100 * mIoU}, sub class IoU is {100 * IoU_of_each_class}')
        return mIoU, final_OA, IoU_of_each_class
    

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_data_with_prefix(data_list, prefix_list):
    return [data_name for data_name in data_list if any(data_name.startswith(prefix) for prefix in prefix_list)]

def remove_data_with_prefix(data_list, prefix_list):
    return [data_name for data_name in data_list if not any(data_name.startswith(prefix) for prefix in prefix_list)]
    
def remake_dataset(args):
    with open(args.train_data_list_path, 'r') as f:
        train_data_name_list = [data_name.strip() for data_name in f]
    with open(args.val_data_list_path, "r") as f:
        val_data_name_list = [data_name.strip() for data_name in f]

    new_source_data = remove_data_with_prefix(train_data_name_list, args.target_event_list) + \
                    remove_data_with_prefix(val_data_name_list, args.target_event_list)
    new_target_data = get_data_with_prefix(train_data_name_list, args.target_event_list) + \
                    get_data_with_prefix(val_data_name_list, args.target_event_list)
    new_holdout_data = random.sample(new_target_data, int(len(new_target_data) * 1))
    args.source_data_name_list = new_source_data
    args.target_data_name_list = new_target_data
    args.holdout_data_name_list = new_holdout_data

    # 只在主进程打印信息
    if os.environ.get('LOCAL_RANK', '0') == '0':
        print(f'Target event is {args.target_event_list}')
        print(f'Source dataset length: {len(new_source_data)}; Holdout dataset length: {len(new_holdout_data)}; Target dataset length: {len(new_target_data)}')

def main():
    parser = argparse.ArgumentParser(description="Training on AegisDA dataset")
    parser.add_argument('--dataset', type=str, default='AegisDA')

    parser.add_argument('--root_path', type=str, default='/shared/kotlin/DATASET/AegisDA')
    parser.add_argument('--train_data_list_path', type=str, default='/shared/kotlin/DATASET/AegisDA/DisasterSet/train.txt')
    parser.add_argument('--val_data_list_path', type=str, default='/shared/kotlin/DATASET/AegisDA/DisasterSet/val.txt')
    parser.add_argument('--target_event_list', type=list, default=['hawaii', 'libya', 'marshall'])

    
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--crop_size', type=int, default=640)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--val_data_name_list', type=list)

    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--gpu_ids', type=str, default="2")
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=105)
    parser.add_argument('--model_type', type=str, default='adaptation/meanteacher')
    parser.add_argument('--model_param_path', type=str, default='./saved_weights')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate_seg', type=float, default=1e-4)
    parser.add_argument('--learning_rate_dis', type=float, default=1e-4)

    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    seed_everything(1260)

    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    remake_dataset(args)

    trainer = Trainer(args)
    trainer.training()

if __name__ == '__main__':
    main()