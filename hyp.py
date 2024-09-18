import os
import torch
import sys

sys.path.append('.')
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

raw_dataset_dir = '<fill it>'

detection_dataset_dir = '<fill it>'
anno_path = '<fill it>'
ckpt_dir = './ckpts'
os.makedirs(ckpt_dir, exist_ok=True)
lownet_path = os.path.join(ckpt_dir, 'lownet.pth')
midnet_path = os.path.join(ckpt_dir, 'midnet.pth')
highnet_path = os.path.join(ckpt_dir, 'highnet.pth')
pseudo_dataset = './pseudo_dataset'
os.makedirs(pseudo_dataset, exist_ok=True)

ir152_backbone_path = '<fill it>/Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth'

ir152_save_path = './results_classifier/ir152.pth'
