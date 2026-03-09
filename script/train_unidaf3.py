import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse

import random
import numpy as np

import torch
import lightning as lg
from lightning.fabric.loggers import CSVLogger

from datetime import datetime

from UniDAF.train_change import change_main
from UniDAF.train_sk import sk_main

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

def remake_dataset(cfg):
    with open(os.path.join(cfg['root_dir'], cfg['disaster_train_list_dir']), 'r') as f:
        train_data_name_list = [data_name.strip() for data_name in f]
    with open(os.path.join(cfg['root_dir'], cfg['disaster_val_list_dir']), 'r') as f:
        val_data_name_list = [data_name.strip() for data_name in f]

    # new_train_data = remove_data_with_prefix(train_data_name_list, cfg['target_event_list'])
    # new_val_data = remove_data_with_prefix(val_data_name_list, cfg['target_event_list'])

    new_target_data = get_data_with_prefix(train_data_name_list, cfg['target_event_list']) + \
                    get_data_with_prefix(val_data_name_list, cfg['target_event_list'])
    new_source_data = remove_data_with_prefix(train_data_name_list, cfg['target_event_list']) + \
                    remove_data_with_prefix(val_data_name_list, cfg['target_event_list'])
    new_holdout_data = random.sample(new_target_data, int(len(new_target_data) * 1))
    
    
    # cfg['train_data_name_list'] = new_train_data
    # cfg['val_data_name_list'] = new_val_data

    cfg['source_data_name_list'] = new_source_data
    cfg['holdout_data_name_list'] = new_holdout_data
    cfg['target_data_name_list'] = new_target_data

    print(f'Target event is {cfg["target_event_list"]}')
    print(f'Source dataset length: {len(new_source_data)}; Holdout dataset length: {len(new_holdout_data)}; Target dataset length: {len(new_target_data)}')

def main():
    parser = argparse.ArgumentParser(description="Training on BuildingSet")
    parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
    args = parser.parse_args()
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    seed_everything(1260)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu_ids']

    # Create a directory to save model weights, organized by timestamp.
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = os.path.join(cfg['model_param_path'], cfg['dataset'], cfg['model_type'] + '_' + now_str)
    # model_save_dir = './saved_weights/AegisDA/unidaf6_pvtv2'

    remake_dataset(cfg)

    gpu_ids = cfg['gpu_ids'].split(',')
    num_devices = len(gpu_ids)
    # from lightning.fabric.strategies import DDPStrategy
    fabric = lg.Fabric(
        accelerator="auto",
        devices=num_devices,
        strategy='ddp' if num_devices > 1 else 'auto',
        precision='bf16-mixed',
        loggers=[CSVLogger(model_save_dir, name=f"{cfg['dataset']}-{cfg['model_type']}", flush_logs_every_n_steps=1)]
    )
    fabric.launch()
    fabric.seed_everything(3047 + fabric.global_rank)
    if fabric.global_rank == 0:
        os.makedirs(model_save_dir, exist_ok=True)

    change_main(cfg, fabric, model_save_dir)
    sk_main(cfg, fabric, model_save_dir)


if __name__ == "__main__":
    main()