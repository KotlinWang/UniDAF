import os

import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# project imports - keep same relative imports as original environment
from dataset.make_data_loader_baseline import MultimodalDamageAssessmentDatset
from model.unidaf3.change import Change
from model.unidaf3.discriminator import FCDiscriminator
from util_func.metrics import Evaluator
import util_func.lovasz_loss as L


class Trainer(object):
    def __init__(self, cfg, fabric, model_save_path):
        self.source_data_name_list = cfg['source_data_name_list']
        self.holdout_data_name_list = cfg['holdout_data_name_list']
        self.target_data_name_list = cfg['target_data_name_list']

        self.cfg = cfg
        self.fabric = fabric
        self.model_save_path = model_save_path
        self.epochs = cfg['sk_epochs']
        self.is_mapping = cfg['mapping']

        self.evaluator = Evaluator(num_class=cfg['clf_classes'])

        # build models on device
        with self.fabric.device:
            self.tea_model = Change(cfg['backbone'], cfg['loc_classes'], cfg['clf_classes'], cfg['hidden_dim'], cfg['tea_eval_size']) 
            self.stu_model = Change(cfg['backbone'], cfg['loc_classes'], cfg['clf_classes'], cfg['hidden_dim'], cfg['stu_eval_size']) 
            self.discriminator = FCDiscriminator(num_classes=cfg['loc_classes'], ndf=128)

            # optionally load weights
            if cfg['resume'] and os.path.isfile(cfg['resume_path']):
                self._load_model(self.stu_model, cfg['resume_path'])
                teacher_path = cfg['resume_path'].replace('.pth', '_teacher.pth')
                if os.path.isfile(teacher_path):
                    self._load_model(self.tea_model, teacher_path)

            change_model_path = os.path.join(self.model_save_path, 'best_change_model.pth')
            if os.path.exists(change_model_path):
                # synchronize initial teacher with student
                self._load_model(self.stu_model, change_model_path)
                self._load_model(self.tea_model, None, copy_from=self.stu_model)

            for param in self.tea_model.parameters():
                param.requires_grad = False

        # optimizers
        self.optim_seg = optim.AdamW(self.stu_model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        self.optim_disc = optim.AdamW(self.discriminator.parameters(), lr=cfg['learning_rate'] * cfg['disc_lr_scale'], weight_decay=cfg['weight_decay'])

        # schedulers: warmup + cosine restarts
        self.lr_scheduler_seg = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim_seg, T_0=15, T_mult=2)
        self.lr_scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim_disc, T_0=15, T_mult=2)
        
        # fabric wrap
        # self.tea_model = self.fabric.setup_module(self.tea_model)
        self.stu_model = self.fabric.setup_module(self.stu_model)
        self.discriminator = self.fabric.setup_module(self.discriminator)
        self.optim_seg = self.fabric.setup_optimizers(self.optim_seg)
        self.optim_disc = self.fabric.setup_optimizers(self.optim_disc)

        # losses & params
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.ema_base = cfg['ema_decay']
        self.sup_only_epoch = cfg['sup_only_epoch']
        self.ema_update_freq = cfg['ema_update_freq']

        self.kd_temperature = cfg['kd_temperature']
        self.kd_alpha = cfg['kd_alpha']
        self.distillation_start_epoch = cfg['distillation_start_epoch']

        self.class_thresholds = np.array(cfg['class_thresholds'])
        assert len(self.class_thresholds) == cfg['clf_classes'], "class_thresholds should have length equal to number of classes"

        self.grad_acc_steps = max(1, cfg['gradient_accumulation_steps'])

    def _load_model(self, model, path=None, copy_from=None):
        if path and os.path.isfile(path):
            checkpoint = torch.load(path, map_location='cpu')
            # try to load matching keys
            sd = model.state_dict()
            ck = {k: v for k, v in checkpoint.items() if k in sd}
            sd.update(ck)
            model.load_state_dict(sd)
        elif copy_from is not None:
            model.load_state_dict({k: v.clone() for k, v in copy_from.state_dict().items()})

    def seg_loss(self, pred, label):
        ce_loss_clf = F.cross_entropy(pred, label, ignore_index=255)
        lovasz_loss_clf = L.lovasz_softmax(F.softmax(pred, dim=1), label, ignore=255)
        return ce_loss_clf + 0.75 * lovasz_loss_clf

    def kd_loss(self, student_logits, teacher_logits, temperature=1):
        # 对 logits 进行温度缩放
        teacher_logits_scaled = teacher_logits / temperature
        student_logits_scaled = student_logits / temperature

        # 使用 MSE 计算学生网络与教师网络的输出之间的差异
        loss = F.mse_loss(student_logits_scaled, teacher_logits_scaled)
        return loss


    def update_ema(self, epoch, step):
        # anneal ema decay to be slightly smaller early training (allow faster adaptation)
        if epoch < self.sup_only_epoch or (step % self.ema_update_freq != 0):
            return
        # dynamic decay: ramp to base
        progress = (epoch - self.sup_only_epoch) / max(1, (self.epochs - self.sup_only_epoch))
        ema_decay = min(0.9999, self.ema_base + 0.5 * (1 - np.exp(-5 * progress)))
        with torch.no_grad():
            for t_param, s_param in zip(self.tea_model.parameters(), self.stu_model.parameters()):
                t_param.data.mul_(ema_decay).add_(s_param.data * (1.0 - ema_decay))

    def _generate_pseudo_labels(self, teacher_logits, temperature=1.0, entropy_thresh=1.5):
        """
        teacher_logits: (N, C, H, W)
        """

        with torch.no_grad():
            # 1️⃣ temperature-scaled softmax
            probs = F.softmax(teacher_logits / temperature, dim=1)

            # 2️⃣ entropy as uncertainty
            entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
            # (N, H, W)

            # 3️⃣ entropy-based confidence mask
            keep_mask = entropy <= entropy_thresh

            # 4️⃣ hard pseudo label
            hard_targets = probs.argmax(dim=1)
            hard_targets.masked_fill_(~keep_mask, 255)

            # 5️⃣ uncertainty-aware soft label weight
            weight = (1.0 - entropy / entropy_thresh).clamp(min=0.0)
            weight = weight.unsqueeze(1)

            soft_targets = probs * weight

        return soft_targets, hard_targets, keep_mask


    def training(self):
        best_mIoU = 0.0
        torch.cuda.empty_cache()

        train_dataset_source = MultimodalDamageAssessmentDatset(self.cfg['root_dir'], self.source_data_name_list, crop_size=self.cfg['crop_size'], type='train')
        train_dataset_target = MultimodalDamageAssessmentDatset(self.cfg['root_dir'], self.target_data_name_list, crop_size=self.cfg['crop_size'], type='train', mapping=self.is_mapping)

        src_loader = DataLoader(train_dataset_source, batch_size=self.cfg['disaster_train_batch_size'], shuffle=True, num_workers=self.cfg['num_workers'], drop_last=True)
        tgt_loader = DataLoader(train_dataset_target, batch_size=self.cfg['disaster_train_batch_size'], shuffle=True, num_workers=self.cfg['num_workers'], drop_last=True)

        src_loader = self.fabric.setup_dataloaders(src_loader)
        tgt_loader = self.fabric.setup_dataloaders(tgt_loader)

        source_label = 0.9  # label smoothing for adversarial
        target_label = 0.1

        for epoch in range(self.epochs):
            self.stu_model.train()
            self.tea_model.eval()

            src_iter = iter(src_loader)
            tgt_iter = iter(tgt_loader)
            iters = min(len(src_loader), len(tgt_loader))

            pbar = tqdm(range(iters), total=iters) if self.fabric.global_rank == 0 else range(iters)

            running_losses = {'seg': 0.0, 'adv': 0.0, 'disc': 0.0}

            for i in pbar:
                # fetch batches
                src = next(src_iter)
                try:
                    tgt = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(tgt_loader)
                    tgt = next(tgt_iter)

                pre_s, post_opt_s, post_sar_s, labels_loc_s, labels_clf_s, _ = src
                pre_t, post_opt_t, post_sar_t, _, _, _ = tgt

                # move to device
                pre_s = self.fabric.to_device(pre_s)
                post_opt_s = self.fabric.to_device(post_opt_s)
                post_sar_s = self.fabric.to_device(post_sar_s)
                labels_clf_s = self.fabric.to_device(labels_clf_s).long().squeeze(1)
                labels_loc_s = self.fabric.to_device(labels_loc_s).long().squeeze(1)

                labels_clf_s1 = labels_clf_s.clone()
                labels_clf_s1[labels_clf_s1 != 1] = 0
                labels_clf_s2 = labels_clf_s.clone()
                labels_clf_s2[labels_clf_s2 != 2] = 0
                labels_clf_s3 = labels_clf_s.clone()
                labels_clf_s3[labels_clf_s3 != 3] = 0

                pre_t = self.fabric.to_device(pre_t)
                post_opt_t = self.fabric.to_device(post_opt_t)
                post_sar_t = self.fabric.to_device(post_sar_t)

                # zero grads for accumulation
                if (i % self.grad_acc_steps) == 0:
                    self.optim_seg.zero_grad()

                # ---------- Teacher predictions on target (no grad) ----------
                with torch.no_grad():
                    inputs_t = torch.cat([pre_t, post_opt_t, post_sar_t], dim=0)
                    t_loc_tgt, t_clf_tgt, t_clf_1, t_clf_2, t_clf_3 = self.tea_model(inputs_t)

                # generate pseudo labels (soft + hard + mask)
                soft_targets_clf, hard_targets_clf, keep_mask_clf = self._generate_pseudo_labels(t_clf_tgt)
                soft_targets_loc, hard_targets_loc, keep_mask_loc = self._generate_pseudo_labels(t_loc_tgt)

                # ---------- Student predictions on combined batch (src + tgt) ----------
                # combine half-half to keep batch balance and reduce memory
                bs = pre_s.size(0)
                half = bs // 2
                
                pre_comb = torch.cat([pre_s[:half], pre_t[:half]], dim=0)
                post_opt_comb = torch.cat([post_opt_s[:half], post_opt_t[:half]], dim=0)
                post_sar_comb = torch.cat([post_sar_s[:half], post_sar_t[:half]], dim=0)

                inputs_comb = torch.cat([pre_comb, post_opt_comb, post_sar_comb], dim=0)
                s_loc, s_clf, s_clf_1, s_clf_2, s_clf_3 = self.stu_model(inputs_comb)

                # ----- segmentation losses -----
                # source part: supervised
                # clf loss
                src_pred_clf = s_clf[:half]
                src_label_clf = labels_clf_s[:half]
                loss_src_clf = self.seg_loss(src_pred_clf, src_label_clf)
                # loc loss
                src_pred_loc = s_loc[:half]
                src_label_loc = labels_loc_s[:half]
                loss_src_loc = self.seg_loss(src_pred_loc, src_label_loc)
                # clf aux loss
                src_pred_clf_1 = s_clf_1[:half]
                src_pred_clf_2 = s_clf_2[:half]
                src_pred_clf_3 = s_clf_3[:half]
                loss_src_clf_1 = F.mse_loss(src_pred_clf_1, labels_clf_s1[:half].unsqueeze(1).float())
                loss_src_clf_2 = F.mse_loss(src_pred_clf_2, labels_clf_s2[:half].unsqueeze(1).float())
                loss_src_clf_3 = F.mse_loss(src_pred_clf_3, labels_clf_s3[:half].unsqueeze(1).float())
                loss_src_clf_aux =  loss_src_clf_1 + loss_src_clf_2 + loss_src_clf_3



                # target part: use pseudo hard labels and KD soft targets
                # clf loss
                tgt_pred_clf = s_clf[half:]
                tgt_hard_clf = hard_targets_clf[:half]
                # loc loss
                tgt_pred_loc = s_loc[half:]
                tgt_hard_loc = hard_targets_loc[:half]
                # convert to device
                tgt_hard_clf = self.fabric.to_device(tgt_hard_clf).long()
                tgt_hard_loc = self.fabric.to_device(tgt_hard_loc).long()
                # supervised CE on confident pixels only
                if (tgt_hard_clf != 255).any():
                    # loss_tgt_ce_clf = F.cross_entropy(tgt_pred_clf, tgt_hard_clf, ignore_index=255)
                    loss_tgt_ce_clf = self.seg_loss(tgt_pred_clf, tgt_hard_clf)
                else:
                    loss_tgt_ce_clf = torch.tensor(0.0, device=self.fabric.device)
                # loc loss
                if (tgt_hard_loc != 255).any():
                    # loss_tgt_ce_loc = F.cross_entropy(tgt_pred_loc, tgt_hard_loc, ignore_index=255)
                    loss_tgt_ce_loc = self.seg_loss(tgt_pred_loc, tgt_hard_loc)
                else:
                    loss_tgt_ce_loc = torch.tensor(0.0, device=self.fabric.device)
                # clf aux loss
                tgt_pred_clf_1 = s_clf_1[half:]
                tgt_pred_clf_2 = s_clf_2[half:]
                tgt_pred_clf_3 = s_clf_3[half:]
                loss_tgt_ce_clf_1 = self.kd_loss(tgt_pred_clf_1, t_clf_1[half:], 2)
                loss_tgt_ce_clf_2 = self.kd_loss(tgt_pred_clf_2, t_clf_2[half:], 2)
                loss_tgt_ce_clf_3 = self.kd_loss(tgt_pred_clf_3, t_clf_3[half:], 2)
                loss_tgt_ce_clf_aux = loss_tgt_ce_clf_1 + loss_tgt_ce_clf_2 + loss_tgt_ce_clf_3
                # 

                # loss_seg = loss_src_clf + loss_tgt_ce_clf + loss_kd + loss_src_loc + loss_tgt_ce_loc
                loss_seg = loss_src_clf + loss_tgt_ce_clf + loss_src_loc + loss_tgt_ce_loc + 0.5*loss_src_clf_aux + 0.75*loss_tgt_ce_clf_aux

                # adversarial loss (student tries to fool discriminator)
                for p in self.discriminator.parameters():
                    p.requires_grad = False
                adv_out = self.discriminator(F.softmax(s_loc, dim=1))
                # adversarial target: source-like labels for student
                adv_target = torch.full_like(adv_out, fill_value=source_label)
                loss_adv = 0.001 * self.bce_loss(adv_out, adv_target)

                # backward with gradient accumulation
                self.fabric.backward(loss_seg + loss_adv)

                if ((i + 1) % self.grad_acc_steps) == 0:
                    self.optim_seg.step()

                # ---------- discriminator update (use detached student features) ----------
                for p in self.discriminator.parameters():
                    p.requires_grad = True
                with torch.no_grad():
                    s_loc_det = s_loc.detach()
                self.optim_disc.zero_grad()
                D_out = self.discriminator(F.softmax(s_loc_det, dim=1))

                # build labels: first half source, second half target
                D_src = torch.full_like(D_out[:half], fill_value=source_label)
                D_tgt = torch.full_like(D_out[half:], fill_value=target_label)
                D_label = torch.cat([D_src, D_tgt], dim=0)

                loss_D = 0.5 * self.bce_loss(D_out, D_label)

                self.fabric.backward(loss_D)
                self.optim_disc.step()

                # update EMA teacher
                self.update_ema(epoch, i)

                # accumulate running statistics
                running_losses['seg'] += loss_seg.detach().cpu().item()
                running_losses['adv'] += loss_adv.detach().cpu().item()
                running_losses['disc'] += loss_D.detach().cpu().item()

                if self.fabric.global_rank == 0:
                    avg_seg = running_losses['seg'] / (i + 1)
                    avg_adv = running_losses['adv'] / (i + 1)
                    avg_disc = running_losses['disc'] / (i + 1)
                    lr = self.lr_scheduler_seg.get_last_lr()[0]
                    pbar.set_description(f'Epoch [{epoch+1}/{self.epochs}]')
                    pbar.set_postfix(seg_loss=avg_seg, adv_loss=avg_adv, disc_loss=avg_disc, lr=lr)

            # step schedulers
            self.lr_scheduler_seg.step(epoch)
            self.lr_scheduler_disc.step(epoch)

            # validation
            self.stu_model.eval()
            val_mIoU, final_OA, IoU_of_each_class = self.validation()

            if val_mIoU > best_mIoU:
                self.fabric.save(os.path.join(self.model_save_path, f'best_sk_model2.pth'), self.stu_model.state_dict())
                best_mIoU = val_mIoU
                best_round = {
                    'best iter': epoch + 1,
                    'best mIoU': str(np.around(val_mIoU * 100, 2)),
                    'best OA': str(np.around(final_OA * 100, 2)),
                    'best sub class IoU': str(np.around(IoU_of_each_class * 100, 2))
                }
            else:
                self.fabric.save(os.path.join(self.model_save_path, f'last_sk_model2.pth'), self.stu_model.state_dict())
            self.fabric.print(best_round)

            # 更新主进程进度条
            if self.fabric.global_rank == 0:
                loss_logger = {
                    'Disaster Epoch': epoch + 1,
                    'Disaster mIoU': val_mIoU * 100,
                    'Disaster OA': final_OA * 100,
                    'Disaster Background': IoU_of_each_class[0] * 100,
                    'Disaster Intact': IoU_of_each_class[1] * 100,
                    'Disaster Damaged': IoU_of_each_class[2] * 100,
                    'Disaster Destroyed': IoU_of_each_class[3] * 100,
                    'Disaster SegLoss': avg_seg,
                    'Disaster AdvLoss': avg_adv,
                    'Disaster DiscLoss': avg_disc,
                }
                self.fabric.log_dict(loss_logger)
            self.fabric.barrier()

            
    def validation(self):
        self.fabric.print('---------starting validation-----------')
        self.evaluator.reset()
        dataset = MultimodalDamageAssessmentDatset(self.cfg['root_dir'], self.holdout_data_name_list, crop_size=1024, type='test', mapping=self.is_mapping)
        val_loader = DataLoader(dataset, batch_size=self.cfg['disaster_eval_batch_size'], num_workers=self.cfg['num_workers'], drop_last=False)
        val_loader = self.fabric.setup_dataloaders(val_loader)

        with torch.no_grad():
            loader = tqdm(val_loader, desc="Validating") if self.fabric.global_rank == 0 else val_loader
            for data in loader:
                pre, post_opt, post_sar, _, labels_clf, _ = data
                pre = self.fabric.to_device(pre)
                post_opt = self.fabric.to_device(post_opt)
                post_sar = self.fabric.to_device(post_sar)
                labels_clf = self.fabric.to_device(labels_clf).long().squeeze(1)

                inputs = torch.cat([pre, post_opt, post_sar], dim=0)
                _, out_clf, _, _, _ = self.stu_model(inputs)
                out_clf = out_clf.cpu().numpy()
                preds = np.argmax(out_clf, axis=1)
                labels = labels_clf.cpu().numpy()
                self.evaluator.add_batch(labels, preds)

        final_OA = self.evaluator.Pixel_Accuracy()
        IoU_of_each_class = self.evaluator.Intersection_over_Union()
        mIoU = self.evaluator.Mean_Intersection_over_Union()

        # distributed reduce
        final_OA = self.fabric.all_reduce(final_OA, reduce_op='mean').item()
        for i in range(len(IoU_of_each_class)):
            IoU_of_each_class[i] = self.fabric.all_reduce(IoU_of_each_class[i], reduce_op='mean').item()
        mIoU = self.fabric.all_reduce(mIoU, reduce_op='mean').item()

        self.fabric.print(f'Val: OA {100 * final_OA:.2f}, mIoU {100 * mIoU:.2f}, IoU per class {100 * np.array(IoU_of_each_class)}')
        return mIoU, final_OA, IoU_of_each_class


# ----------------- Helpers -----------------
class WarmupCosineScheduler:
    """Simple warmup -> cosine scheduler wrapper for python optimizers."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * float(epoch + 1) / float(max(1, self.warmup_epochs))
        else:
            # cosine annealing
            t = (epoch - self.warmup_epochs) / max(1, (self.total_epochs - self.warmup_epochs))
            lr = 0.5 * (1 + np.cos(np.pi * t)) * self.base_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_last_lr(self):
        return [g['lr'] for g in self.optimizer.param_groups]


# ----------------- Entrypoint -----------------

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def sk_main(cfg, fabric, model_save_dir):
    trainer = Trainer(cfg, fabric, model_save_dir)
    trainer.training()

