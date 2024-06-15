import os
from tqdm import tqdm

import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw
from torchvision import transforms as T

import hyp
from cvpiclr.dataset.dataset import ClassificationDataset
from cvpiclr.models.detect import Detector
from cvpiclr.models.classifier import auto_classifier_from_pretrained
from cvpiclr.utils import box as boxutils

default_iou_thresholds = np.linspace(0, 100, 100)

trans = T.Compose(
    [
        T.ToTensor(),
        T.Resize((64, 64), antialias=True),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def compute_ap(precision, recall):
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap


@torch.no_grad()
def evaluation_nontarget(mode, iou_thresholds=default_iou_thresholds):

    dataset = ClassificationDataset(
        hyp.anno_path, hyp.raw_dataset_dir, mode=mode, labeled=False
    )
    detector = Detector(hyp.lownet_path, hyp.midnet_path, hyp.highnet_path, hyp.device)

    # if use_classifier:
    #     classifier = auto_classifier_from_pretrained(hyp.ir152_save_path).to(hyp.device)

    TP = np.zeros_like(iou_thresholds)
    FP = np.zeros_like(iou_thresholds)

    all_iou = []

    for img, target, label_box in tqdm(dataset):
        pred_boxes = detector.detect(img)
        if len(pred_boxes) == 0:
            continue
        ious = boxutils.iou(label_box, pred_boxes)

        for box, iou in zip(pred_boxes, ious):
            mask = iou_thresholds >= iou
            all_iou.append(iou)
            TP[mask] += 1
            FP[~mask] += 1
            break

    precision = TP / (TP + FP)
    recall = TP / len(dataset)
    eps = 1e-7
    f1 = 2 / (
        1 / precision.clip(min=eps, max=None) + 1 / recall.clip(min=eps, max=None)
    )
    ap = compute_ap(precision, recall)
    meaniou = np.array(all_iou).mean()

    return precision, recall, f1, ap, meaniou


if __name__ == '__main__':
    precision, recall, f1, ap, meaniou = evaluation_nontarget('val')
    print(ap, meaniou)
    os.makedirs('./results_eval')
    torch.save([precision, recall, f1, ap, meaniou], 'val_nontarget.pt')
    # precision, recall, f1, ap, meaniou = evaluation_target('val')
    # print(ap, meaniou)
    # os.makedirs('./results_eval', exist_ok=True)
    # torch.save([precision, recall, f1, ap, meaniou], 'val_target.pt')

    precision, recall, f1, ap, meaniou = evaluation_nontarget('test')
    print(ap, meaniou)
    torch.save([precision, recall, f1, ap, meaniou], 'test_nontarget.pt')
    # precision, recall, f1, ap, meaniou = evaluation_target('test')
    # print(ap, meaniou)
    # torch.save([precision, recall, f1, ap, meaniou], 'test_target.pt')

    import pandas as pd

    res = []  # pd.DataFrame(columns=['precision', 'recall', 'f1', 'ap', 'meaniou'])

    for name in ['val_nontarget', 'test_nontarget', 'val_target', 'test_target']:
        data = torch.load(f'{name}.pt', map_location='cpu')
        print(len(data))
        data = [np.mean(d) if isinstance(d, np.ndarray) else d for d in data]
        data = [np.round(d, 3) for d in data]
        res.append(data)
    # print(torch.load('test_target.pt')[0])
    # exit()

    df = pd.DataFrame(res, columns=['precision', 'recall', 'f1', 'ap', 'meaniou'])

    df.to_csv('result.csv', index=None)
