from cvpiclr.dataset.dataset import ClassificationDataset
from cvpiclr.models.detect import Detector
from tqdm import tqdm

from PIL import Image, ImageDraw, ImageFont
import os
import torch
import sys
from cvpiclr.utils import box as boxutils

sys.path.append('.')
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import hyp

trainset = ClassificationDataset(hyp.anno_path, hyp.raw_dataset_dir, mode='train')

# detector = Detector(P_PATH, R_PATH, O_PATH, 'cuda')
detector = Detector(hyp.lownet_path, hyp.midnet_path, hyp.highnet_path, 'cuda')

save_dir = hyp.pseudo_dataset

cnt = 0
for i, (img, target, _) in enumerate(tqdm(trainset)):
    pred_boxes = detector.detect(img)
    if len(pred_boxes) == 0:
        continue
    h, w = img.size

    for box in pred_boxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        cls = box[4]
        newim = img.crop((x1, y1, x2, y2))

        save_lable_dir = os.path.join(save_dir, f'{target}')
        os.makedirs(save_lable_dir, exist_ok=True)
        newim.save(os.path.join(save_lable_dir, f'{i}.png'))

        break
