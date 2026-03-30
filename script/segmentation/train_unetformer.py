import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

import argparse
import os

import numpy as np
import random


import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.make_data_loader_baseline import MultimodalDamageAssessmentDatset
from model.UNetFormer import UNetFormer
from datetime import datetime

from util_func.metrics import Evaluator
import util_func.lovasz_loss as L

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
            - dataset, train_dataset_path, val_dataset_path, etc.
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
            if not os.path.exists(self.model_save_path):
                os.makedirs(self.model_save_path)

        # Initialize evaluator for metrics such as accuracy, IoU, etc.
        self.evaluator = Evaluator(num_class=4)

        # Create the deep learning model. Here we show how to use UNet or SiamCRNN.
        with self.fabric.device:
            # caformer_s36.sail_in22k_ft_in1k
            # pvt_v2_b5.in1k
            # resnet18.fb_swsl_ig1b_ft_in1k
            self.deep_model = UNetFormer(backbone_name='resnet18.fb_swsl_ig1b_ft_in1k', encoder_channels=9, num_classes=4) 

        if args.resume:
            if not os.path.isfile(args.resume_path):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume_path))
            self._load_model(self.deep_model, args.resume_path)

        self.optim = optim.AdamW(self.deep_model.parameters(),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
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
        ce_loss_clf = F.cross_entropy(pred, label)
        lovasz_loss_clf = L.lovasz_softmax(F.softmax(pred, dim=1), label, ignore=255)
        loss_seg = ce_loss_clf + 0.75 * lovasz_loss_clf # iuf you use UNet
        return loss_seg

    def training(self):
        """
        Main training loop that iterates over the training dataset for several steps (max_iters).
        Prints intermediate losses and evaluates on holdout dataset periodically.
        """
        best_mIoU = 0.0
        best_round = []
        torch.cuda.empty_cache()
        train_dataset = MultimodalDamageAssessmentDatset(self.args.root_path, self.args.train_data_name_list, crop_size=self.args.crop_size, type='train')
        train_data_loader = DataLoader(train_dataset, batch_size=self.args.train_batch_size, shuffle=True, pin_memory=True, num_workers=self.args.num_workers, drop_last=True)
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

                pre_change_imgs = self.fabric.to_device(pre_change_imgs)
                post_opt_change_imgs = self.fabric.to_device(post_opt_change_imgs)
                post_sar_change_imgs = self.fabric.to_device(post_sar_change_imgs)
                labels_loc = self.fabric.to_device(labels_loc).long()
                labels_clf = self.fabric.to_device(labels_clf).long()

                valid_labels_clf = (labels_clf != 255).any()
                if not valid_labels_clf:
                    continue
                input = torch.cat([pre_change_imgs, post_opt_change_imgs, post_sar_change_imgs], dim=1)
                clf_output, aux_output = self.deep_model(input) # If you use SiamCRNN 
                
                final_loss = self.seg_loss(clf_output, labels_clf) + 0.75*self.seg_loss(aux_output, labels_clf)

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
                self.fabric.save(os.path.join(self.model_save_path, f'best_model.pth'), self.deep_model.state_dict())
                best_mIoU = val_mIoU
                best_round = {
                    'best iter': epoch + 1,
                    'best mIoU': str(np.around(val_mIoU * 100, 2)),
                    'best OA': str(np.around(final_OA * 100, 2)),
                    'best sub class IoU': str(np.around(IoU_of_each_class * 100, 2))
                }
            else:
                self.fabric.save(os.path.join(self.model_save_path, f'last_model.pth'), self.deep_model.state_dict())
            self.fabric.print(best_round)
            self.deep_model.train()
            self.fabric.barrier()


    def validation(self):
        self.fabric.print('---------starting validation-----------')
        self.evaluator.reset()
        dataset = MultimodalDamageAssessmentDatset(self.args.root_path, self.args.val_data_name_list, crop_size=1024, type='test')
        val_data_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size, pin_memory=True, num_workers=4, drop_last=False)
        val_data_loader = self.fabric.setup_dataloaders(val_data_loader)
        torch.cuda.empty_cache()

        with torch.no_grad():
            # 主进程显示进度条
            if self.fabric.global_rank == 0:
                loader = tqdm(val_data_loader, desc="Validating")
            else:
                loader = val_data_loader
            for data in loader:
                pre_change_imgs, post_opt_change_imgs, post_sar_change_imgs, _, labels_clf, _ = data

                pre_change_imgs = self.fabric.to_device(pre_change_imgs)
                post_opt_change_imgs = self.fabric.to_device(post_opt_change_imgs)
                post_sar_change_imgs = self.fabric.to_device(post_sar_change_imgs)
                labels_clf = self.fabric.to_device(labels_clf).long()

                
                input = torch.cat([pre_change_imgs, post_opt_change_imgs, post_sar_change_imgs], dim=1)
                clf_output = self.deep_model(input)

                output_clf = clf_output.data.cpu().numpy()
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
    args.train_data_name_list = new_source_data
    args.val_data_name_list = new_holdout_data

    print(f'Target event is {args.target_event_list}')
    print(f'Source dataset length: {len(new_source_data)}; Holdout dataset length: {len(new_holdout_data)}')
    

def main():
    parser = argparse.ArgumentParser(description="Training on BRIGHT dataset")

    parser.add_argument('--dataset', type=str, default='AegisDA')
    parser.add_argument('--root_path', type=str, default='/shared/kotlin/DATASET/AegisDA')
    parser.add_argument('--train_data_list_path', type=str, default='/shared/kotlin/DATASET/AegisDA/DisasterSet/train.txt')
    parser.add_argument('--val_data_list_path', type=str, default='/shared/kotlin/DATASET/AegisDA/DisasterSet/val.txt')
    parser.add_argument('--target_event_list', type=list, default=['hawaii', 'libya', 'marshall'])
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=1)

    parser.add_argument('--crop_size', type=int, default=512)

    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--val_data_name_list', type=list)

    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--gpu_ids', type=str, default="3")
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=105)
    parser.add_argument('--model_type', type=str, default='segmention/unetformer')
    parser.add_argument('--model_param_path', type=str, default='./saved_weights')
    
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
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


if __name__ == "__main__":
    main()