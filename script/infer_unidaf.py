import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.make_data_loader_baseline import MultimodalDamageAssessmentDatset

from PIL import Image
from model.unidaf3.change import Change
import argparse
from datetime import datetime

from util_func.metrics import Evaluator
import ttach as tta


def get_data_with_prefix(data_list, prefix_list):
    return [data_name for data_name in data_list if any(data_name.startswith(prefix) for prefix in prefix_list)]

def remove_data_with_prefix(data_list, prefix_list):
    return [data_name for data_name in data_list if not any(data_name.startswith(prefix) for prefix in prefix_list)]

def remake_dataset(args):
    with open(args.train_data_list_path, 'r') as f:
        train_data_name_list = [data_name.strip() for data_name in f]
    with open(args.val_data_list_path, "r") as f:
        val_data_name_list = [data_name.strip() for data_name in f]

    new_target_data = get_data_with_prefix(train_data_name_list, args.target_event_list) + \
                    get_data_with_prefix(val_data_name_list, args.target_event_list)

    args.val_data_name_list = new_target_data

    print(f'Target event is {args.target_event_list}')
    print(f'Val dataset length: {len(new_target_data)}')


class Inference:
    def __init__(self, args):
        self.args = args

        self.evaluator = Evaluator(num_class=4)

        # Load dataset
        dataset = MultimodalDamageAssessmentDatset(args.dataset_path, args.val_data_name_list, 1024, 'test', suffix='.tif', mapping=False)
        self.val_loader = DataLoader(dataset, batch_size=1, num_workers=1, drop_last=False)
        
        # Load model
        # pvt_v2_b3.in1k
        # resnet18.fb_swsl_ig1b_ft_in1k
        # mobilenetv4_conv_small.e1200_r224_in1k
        self.model = Change('mobilenetv4_conv_small.e1200_r224_in1k', 2, 4, 128) 
        

        self.model = self.model.cuda()
        self.model.eval()
        
       # TTA transforms configuration - optimized for balance between accuracy and speed
        if self.args.use_tta:
            try:
                self.tta_transforms = tta.Compose([
                    tta.HorizontalFlip(),
                    tta.VerticalFlip(),
                    tta.Rotate90(angles=[0, 90, 180, 270]),
                    # tta.Scale(scales=[0.75, 1, 1.25, 1.5, 1.75]),
                    # tta.Multiply(factors=[0.8, 1, 1.2])
                ])
            except ImportError:
                print("Warning: ttach library not found. TTA will be disabled.")
                self.args.use_tta = False

        self.color_map = {
            0: (255, 255, 255),   # No damage - white
            1: (35,77,161),    # Intact - green
            2: (249,202,94),   # Damaged - orange
            3: (218,80,37)      # Destroyed - red
        }


        self.now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(os.path.join(self.args.inferece_saved_path, self.now_str)):
            os.makedirs(os.path.join(self.args.inferece_saved_path, self.now_str))
            os.makedirs(os.path.join(self.args.inferece_saved_path, self.now_str, 'raw'))
            os.makedirs(os.path.join(self.args.inferece_saved_path, self.now_str, 'color'))

        if args.existing_weight_path is not None:
            if not os.path.isfile(args.existing_weight_path):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.existing_weight_path))
            checkpoint = torch.load(args.existing_weight_path, weights_only=True)
            model_dict = {}
            state_dict = self.model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.model.load_state_dict(state_dict)
            print('Loaded existing weights from {}'.format(args.existing_weight_path))

    def run_inference(self):
        if self.args.use_tta:
            print('Starting inference with TTA...')
            # Wrap model with TTA
            tta_model = tta.SegmentationTTAWrapper(
                self.model, 
                self.tta_transforms, 
                merge_mode='mean'
            )
        else:
            print('Starting inference without TTA...')

        self.evaluator.reset()
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.val_loader)):
                pre_change_imgs, post_opt_change_imgs, post_sar_change_imgs, labels_loc, labels_clf, file_name = data
                pre_change_imgs = pre_change_imgs.cuda()
                post_opt_change_imgs = post_opt_change_imgs.cuda()
                post_sar_change_imgs = post_sar_change_imgs.cuda()
                labels_clf = labels_clf.cuda().long()
                file_name = file_name[0]  # Get the filename as a string

                # input_data = torch.cat([post_opt_change_imgs, post_sar_change_imgs], dim=1) # if you use UNet
                if self.args.use_tta:
                    # Use TTA for inference
                    img_data = torch.cat([pre_change_imgs, post_opt_change_imgs, post_sar_change_imgs], dim=0)
                    output = tta_model(img_data)
                else:
                    # Standard inference
                    img_data = torch.cat([pre_change_imgs, post_opt_change_imgs, post_sar_change_imgs], dim=0)
                    _, output, _, _, _ = self.model(img_data)  # if you use UNet

                output_clf = output.clone()
                output_clf = output_clf.data.cpu().numpy()
                output_clf = np.argmax(output_clf, axis=1)
                labels_clf = labels_clf.cpu().numpy()

                self.evaluator.add_batch(labels_clf, output_clf)

                output = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)

                self.save_prediction_map(output, file_name)

        final_OA = self.evaluator.Pixel_Accuracy()
        final_F1 = self.evaluator.Mean_F1_socore()
        IoU_of_each_class = self.evaluator.Intersection_over_Union()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        print(f'OA is {100 * final_OA}, F1 is {100 * final_F1}, mIoU is {100 * mIoU}, sub class IoU is {100 * IoU_of_each_class}')

    def save_prediction_map(self, prediction, file_name):
        """Saves the raw and colored prediction maps"""
        color_map_img = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
        
        for cls, color in self.color_map.items():
            color_map_img[prediction == cls] = color

        raw_output_path = os.path.join(self.args.inferece_saved_path, self.now_str, 'raw', file_name + '_building_damage.png') # upload this to leaderboard
        color_output_path = os.path.join(self.args.inferece_saved_path, self.now_str, 'color', file_name + '_building_damage.png')  # this is for your visualization
        Image.fromarray(prediction).save(raw_output_path)
        Image.fromarray(color_map_img).save(color_output_path)

   
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    parser = argparse.ArgumentParser(description="Inference on BRIGHT dataset")

    parser.add_argument('--dataset_path', type=str, default='/shared/kotlin/DATASET/AegisDA')
    parser.add_argument('--train_data_list_path', type=str, default='/shared/kotlin/DATASET/AegisDA/DisasterSet3/train.txt')
    parser.add_argument('--val_data_list_path', type=str, default='/shared/kotlin/DATASET/AegisDA/DisasterSet3/val.txt')
    parser.add_argument('--target_event_list', type=list, default=['hawaii'])
    parser.add_argument('--val_data_name_list', type=list)
    parser.add_argument('--existing_weight_path', type=str,
                        default='./saved_weights/AegisDA/unidaf3_mobile_cross/best_sk_model.pth')
    parser.add_argument('--inferece_saved_path', type=str, default='./inference_results')
    parser.add_argument('--use_tta', action='store_true', help='Enable Test Time Augmentation', default=False)

    args = parser.parse_args()
    
    remake_dataset(args)
    
    inference = Inference(args)
    inference.run_inference()

