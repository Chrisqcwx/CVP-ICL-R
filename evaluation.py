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
def evaluation_nontarget(mode, iou_thresholds=default_iou_thresholds, alpha=0.709):

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
        pred_boxes = detector.detect(img, low_pred_scaler=alpha)
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


@torch.no_grad()
def evaluation_target(mode, iou_thresholds=default_iou_thresholds, alpha=0.709):

    dataset = ClassificationDataset(hyp.anno_path, hyp.raw_dataset_dir, mode=mode)
    detector = Detector(hyp.lownet_path, hyp.midnet_path, hyp.highnet_path, hyp.device)

    # if use_classifier:
    classifier = auto_classifier_from_pretrained(hyp.ir152_save_path).to(hyp.device)
    classifier.eval()

    # print(len(dataset.classes))
    TP = np.zeros((len(dataset.classes), len(iou_thresholds)))
    FP = np.zeros((len(dataset.classes), len(iou_thresholds)))

    ALL = np.zeros((len(dataset.classes),))

    all_iou = []

    for img, target, label_box in tqdm(dataset):
        pred_boxes = detector.detect(img, low_pred_scaler=alpha)
        if len(pred_boxes) == 0:
            # FP += 1
            continue
        # print(boxutils.iou(label_box, pred_boxes))
        # exit()
        ious = boxutils.iou(label_box, pred_boxes)
        ALL[target] += 1
        for box, iou in zip(pred_boxes, ious):
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            cls = box[4]

            # if use_classifier:
            newim = img.crop((x1, y1, x2, y2))
            newim_tensor = trans(newim).unsqueeze(0).to(hyp.device)
            pred = classifier(newim_tensor)[0].argmax(dim=-1).cpu().item()
            mask = (iou_thresholds >= iou) & (pred == target)

            all_iou.append(iou)
            # print(TP.shape, pred)
            TP[pred][mask] += 1
            FP[pred][~mask] += 1
            break

    precision = TP / (TP + FP)
    precision = np.where(TP + FP == 0, 0, precision)
    recall = TP / ALL.reshape(-1, 1)
    eps = 1e-7
    f1 = 2 / (
        1 / precision.clip(min=eps, max=None) + 1 / recall.clip(min=eps, max=None)
    )

    aps = []
    for i in range(len(precision)):
        aps.append(compute_ap(precision[i], recall[i]))
    aps = np.array(aps).mean()
    meaniou = np.array(all_iou).mean()

    return precision.mean(axis=0), recall.mean(axis=0), f1.mean(axis=0), aps, meaniou


import time

if __name__ == '__main__':

    os.makedirs('./results_eval', exist_ok=True)
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
        # t = time.time()
        # precision, recall, f1, ap, meaniou = evaluation_nontarget('val')
        # print(ap, meaniou)
        # torch.save(
        #     [precision, recall, f1, ap, meaniou, time.time() - t],
        #     f'./results_eval/val_nontarget_{alpha}.pt',
        # )
        t = time.time()
        precision, recall, f1, ap, meaniou = evaluation_target('val', alpha=alpha)
        print(ap, meaniou)
        torch.save(
            [precision, recall, f1, ap, meaniou, time.time() - t],
            f'./results_eval/val_target_{alpha}.pt',
        )

        # t = time.time()
        # precision, recall, f1, ap, meaniou = evaluation_nontarget('test')
        # print(ap, meaniou)
        # torch.save(
        #     [precision, recall, f1, ap, meaniou, time.time() - t],
        #     f'./results_eval/test_nontarget_{alpha}.pt',
        # )
        t = time.time()
        precision, recall, f1, ap, meaniou = evaluation_target('test', alpha=alpha)
        print(ap, meaniou)
        torch.save(
            [precision, recall, f1, ap, meaniou, time.time() - t],
            f'./results_eval/test_target_{alpha}.pt',
        )

    # import pandas as pd

    # res = []  # pd.DataFrame(columns=['precision', 'recall', 'f1', 'ap', 'meaniou'])

    # for name in ['val_nontarget', 'test_nontarget', 'val_target', 'test_target']:
    #     data = torch.load(f'{name}.pt', map_location='cpu')
    #     print(len(data))
    #     data = [np.mean(d) if isinstance(d, np.ndarray) else d for d in data]
    #     data = [np.round(d, 3) for d in data]
    #     res.append(data)
    # # print(torch.load('test_target.pt')[0])
    # # exit()

    # df = pd.DataFrame(res, columns=['precision', 'recall', 'f1', 'ap', 'meaniou'])

    # df.to_csv('result.csv', index=None)
