import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

import argparse
import os
import time

import numpy as np


import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.make_data_loader_baseline import MultimodalDamageAssessmentDatset
from model.unidaf3.change import Change
from datetime import datetime
from model.ADVENT.advent import Discriminator, DomainAdversarialEntropyLoss

from util_func.metrics import Evaluator
import util_func.lovasz_loss as L

import lightning as lg
from lightning.fabric.loggers import CSVLogger

import random


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

        # Create a directory to save model weights, organized by timestamp.
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_path = os.path.join(args.model_param_path, args.dataset, args.model_type + '_' + now_str)

        # from lightning.fabric.strategies import DDPStrategy
        self.fabric = lg.Fabric(
            accelerator="auto",
            devices=num_devices,
            # strategy='ddp',
            precision='bf16-mixed',
            loggers=[CSVLogger(self.model_save_path, name=f"{args.dataset}-{args.model_type}", flush_logs_every_n_steps=1)]
        )
        self.fabric.launch()
        self.fabric.seed_everything(3047 + self.fabric.global_rank)

        if self.fabric.global_rank == 0:
            os.makedirs(self.model_save_path, exist_ok=True)

        # Initialize evaluator for metrics such as accuracy, IoU, etc.
        self.evaluator = Evaluator(num_class=4)


        # Create the deep learning model. Here we show how to use UNet or SiamCRNN.
        # Create the deep learning model. Here we show how to use UNet or SiamCRNN.
        with self.fabric.device:
            self.deep_model = Change('resnet18.fb_swsl_ig1b_ft_in1k', 2, 4, 128) 
            self.discriminator = Discriminator(num_classes=2, ndf=128)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.optim_seg = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate_seg,
                                 weight_decay=args.weight_decay)
        
        self.optim_dis = optim.AdamW(self.discriminator.parameters(),
                                 lr=args.learning_rate_dis,
                                 weight_decay=args.weight_decay)
        self.lr_scheduler_seg = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim_seg, T_0=15, T_mult=2)
        self.lr_scheduler_dis = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim_dis, T_0=15, T_mult=2)
        
        self.dann = DomainAdversarialEntropyLoss(self.fabric.setup_module(self.discriminator))

        self.deep_model = self.fabric.setup_module(self.deep_model)
        self.optim_seg = self.fabric.setup_optimizers(self.optim_seg)
        self.optim_dis = self.fabric.setup_optimizers(self.optim_dis)

    def training(self):
        """
        Main training loop that iterates over the training dataset for several steps (max_iters).
        Prints intermediate losses and evaluates on holdout dataset periodically.
        """
        best_mIoU = 0.0
        best_round = []

        torch.cuda.empty_cache()
        source_label = 0
        target_label = 1
    
        train_dataset_source = MultimodalDamageAssessmentDatset(self.args.root_path, self.args.source_data_name_list, crop_size=self.args.crop_size, type='train')
        train_dataset_target = MultimodalDamageAssessmentDatset(self.args.root_path, self.args.target_data_name_list, crop_size=self.args.crop_size, type='train')

        train_data_loader_source = DataLoader(train_dataset_source, batch_size=self.args.train_batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=False)
        train_data_loader_target = DataLoader(train_dataset_target, batch_size=self.args.train_batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=False)

        train_data_loader_source = self.fabric.setup_dataloaders(train_data_loader_source)
        train_data_loader_target = self.fabric.setup_dataloaders(train_data_loader_target)
        for epoch in range(self.args.epochs):
            self.deep_model.train()

            src_iter = iter(train_data_loader_source)
            tgt_iter = iter(train_data_loader_target)
            iters = max(len(train_data_loader_source), len(train_data_loader_target))

            pbar = tqdm(range(iters), total=iters) if self.fabric.global_rank == 0 else range(iters)
            running_losses = {'seg': 0.0, 'adv': 0.0, 'disc': 0.0}

            for i in pbar:
                # fetch batches
                src = next(src_iter)
                try:
                    tgt = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(train_data_loader_target)
                    tgt = next(tgt_iter)
            
                self.optim_seg.zero_grad() 

                pre_s, post_s, sar_s, loc_gt, clf_gt, _ = src
                pre_s = self.fabric.to_device(pre_s)
                post_s = self.fabric.to_device(post_s)
                sar_s = self.fabric.to_device(sar_s)
                loc_gt = self.fabric.to_device(loc_gt).long()
                clf_gt = self.fabric.to_device(clf_gt).long()
                

                valid_labels_clf = (clf_gt != 255).any()
                if not valid_labels_clf:
                    continue
                
                # Training on the source domain
                inputs_s = torch.cat([pre_s, post_s, sar_s], dim=0)
                out_loc, out_clf, o1, o2, o3 = self.deep_model(inputs_s)

                loss_loc = F.cross_entropy(out_loc, loc_gt) + \
                           0.75 * L.lovasz_softmax(
                               F.softmax(out_loc, dim=1), loc_gt, ignore=255
                           )
                loss_clf = F.cross_entropy(out_clf, clf_gt) + \
                           0.75 * L.lovasz_softmax(
                               F.softmax(out_clf, dim=1), clf_gt, ignore=255
                           )

                loss_aux = (
                    0.5 * F.mse_loss(o1, (clf_gt == 1).float().unsqueeze(1)) +
                    1.0 * F.mse_loss(o2, (clf_gt == 2).float().unsqueeze(1)) +
                    0.75 * F.mse_loss(o3, (clf_gt == 3).float().unsqueeze(1))
                )

                loss_seg = loss_loc + loss_clf + loss_aux
                self.fabric.backward(loss_seg)

                if epoch >= 30:
                    # prediction on the target domain
                    pre_t, post_t, sar_t, _, _, _ = tgt
                    pre_t = self.fabric.to_device(pre_t)
                    post_t = self.fabric.to_device(post_t)
                    sar_t = self.fabric.to_device(sar_t)

                    self.dann.eval()
                    self.optim_dis.zero_grad()

                    inputs_t = torch.cat([pre_t, post_t, sar_t], dim=0)
                    out_loc_t, _, _, _, _ = self.deep_model(inputs_t)
                    loss_transfer = self.dann(out_loc_t, 'source')
                    self.fabric.backward(0.001*loss_transfer)
                    self.optim_seg.step()

                    # Train the discriminator
                    self.dann.train()
                    self.optim_dis.zero_grad()

                    loss_discriminator = 0.5 * (self.dann(out_loc.detach(), 'source') + self.dann(out_loc_t.detach(), 'target'))
                    self.fabric.backward(loss_discriminator)
                    self.optim_dis.step()

                    # accumulate running statistics
                    running_losses['seg'] += loss_seg.detach().cpu().item()
                    running_losses['adv'] += (0.001*loss_transfer).detach().cpu().item()
                    running_losses['disc'] += loss_discriminator.detach().cpu().item()

                    if self.fabric.global_rank == 0:
                        avg_seg = running_losses['seg'] / (i + 1)
                        avg_adv = running_losses['adv'] / (i + 1)
                        avg_disc = running_losses['disc'] / (i + 1)
                        lr = self.lr_scheduler_seg.get_last_lr()[0]
                        pbar.set_description(f'Epoch [{epoch+1}/{self.epochs}]')
                        pbar.set_postfix(seg_loss=avg_seg, adv_loss=avg_adv, disc_loss=avg_disc, lr=lr)
                else:
                    self.optim_seg.step()
                    running_losses['seg'] += loss_seg.detach().cpu().item()
                    if self.fabric.global_rank == 0:
                        avg_seg = running_losses['seg'] / (i + 1)
                        lr = self.lr_scheduler_seg.get_last_lr()[0]
                        pbar.set_description(f'Epoch [{epoch+1}/{self.epochs}]')
                        pbar.set_postfix(seg_loss=avg_seg, lr=lr)
                

            self.lr_scheduler_seg.step(epoch)
            self.lr_scheduler_dis.step(epoch)

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
            self.deep_model.train()
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
                _, output_clf, _, _, _ = self.deep_model(inputs)  # if you use UNet    
                # _, output_clf = self.deep_model(pre_change_imgs, post_change_imgs) # If you use SiamCRNN


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

    print(f'Target event is {args.target_event_list}')
    print(f'Source dataset length: {len(new_source_data)}; Holdout dataset length: {len(new_holdout_data)}; Target dataset length: {len(new_target_data)}')

def main():
    parser = argparse.ArgumentParser(description="Training on AegisDA dataset")
    parser.add_argument('--dataset', type=str, default='AegisDA')

    parser.add_argument('--root_path', type=str, default='/shared/kotlin/DATASET/AegisDA')
    parser.add_argument('--train_data_list_path', type=str, default='/shared/kotlin/DATASET/AegisDA/DisasterSet/train.txt')
    parser.add_argument('--val_data_list_path', type=str, default='/shared/kotlin/DATASET/AegisDA/DisasterSet/val.txt')
    parser.add_argument('--target_event_list', type=list, default=['hawaii', 'libya', 'marshall'])

    
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--crop_size', type=int, default=640)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--val_data_name_list', type=list)

    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--gpu_ids', type=str, default="3")
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=105)
    parser.add_argument('--model_type', type=str, default='adaptation/advent')
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