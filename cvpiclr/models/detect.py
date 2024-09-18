import torch
from torch import nn
from torchvision.transforms.functional import resize, crop
import numpy as np
from ..utils.box import nms, convert_to_square
from torchvision.transforms import InterpolationMode


class LowNet(nn.Module):
    def __init__(self):
        super(LowNet, self).__init__()
        self.resolution = 12
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=10,
                kernel_size=3,
            ),  # 10*10*10
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),  # 5*5*10
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=16,
                kernel_size=3,
            ),  # 3*3*16
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
            ),  # 1*1*32
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv4 = nn.Conv2d(
            in_channels=32,
            out_channels=5,
            kernel_size=1,
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        category = torch.sigmoid(y[:, 0:1])
        offset = y[:, 1:]
        return category, offset


class MidNet(nn.Module):
    def __init__(self):
        super(MidNet, self).__init__()
        self.resolution = 24
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=28,
                kernel_size=3,
            ),  # 22*22*28
            nn.BatchNorm2d(28),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), 2, 1),  # 11*11*28
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=28,
                out_channels=48,
                kernel_size=3,
            ),  # 9*9*48
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), 2, 0),  # 4*4*48
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=48,
                out_channels=64,
                kernel_size=2,
            ),  # 3*3*64
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(3 * 3 * 64, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        y = self.conv1(x)
        # print(y.shape)
        y = self.conv2(y)
        # print(y.shape)
        y = self.conv3(y)
        # print(y.shape)
        y = torch.reshape(y, [y.size(0), -1])
        # print(y.shape)
        y = self.fc1(y)
        # print(y.shape)
        y = self.fc2(y)
        # print(y.shape)

        category = torch.sigmoid(y[:, 0:1])
        offset = y[:, 1:]
        return category, offset


class HighNet(nn.Module):
    def __init__(self):
        super(HighNet, self).__init__()
        self.resolution = 48
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
            ),  # 46*46*32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), 2, 1),  # 23*23*32
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
            ),  # 21*21*64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), 2),  # 10*10*64
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
            ),  # 8*8*64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),  # 4*4*64
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=2,
            ),  # 3*3*128
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(3 * 3 * 128, 256), nn.BatchNorm1d(256), nn.ReLU()
        )
        self.fc2 = nn.Linear(256, 5)

    def forward(self, x):
        y = self.conv1(x)
        # print(y.shape)
        y = self.conv2(y)
        # print(y.shape)
        y = self.conv3(y)
        # print(y.shape)
        y = self.conv4(y)
        # print(y.shape, "==========")
        y = torch.reshape(y, [y.size(0), -1])
        # print(y.shape)

        y = self.fc1(y)
        # print(y.shape)
        y = self.fc2(y)
        # print(y.shape)
        category = torch.sigmoid(y[:, 0:1])
        offset = y[:, 1:]
        return category, offset


import os
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim

from ..dataset.dataset import DetectionDataset

# from tensorboardX import SummaryWriter
from tqdm import tqdm


class DetectTrainer:
    def __init__(self, net, save_path, dataset_path, device):
        self.device = device
        self.net = net
        self.save_path = save_path
        # os.makedirs(save_path, exist_ok=True)
        self.dataset_path = dataset_path
        self.resolution = net.resolution
        self.cls_loss_fn = nn.BCELoss()
        self.offset_loss_fn = nn.MSELoss()

        self.optimizer = optim.Adam(self.net.parameters())

        if os.path.exists(self.save_path):  # 是否有已经保存的参数文件
            net.load_state_dict(torch.load(self.save_path, map_location='cpu'))
        else:
            print("NO Param")

    def train(self, max_epoch):
        stop_value = 0
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [
                        0.5,
                    ],
                    [
                        0.5,
                    ],
                ),
            ]
        )
        faceDataset = DetectionDataset(
            self.dataset_path, 'train', self.resolution, transform=trans
        )  # 实例化对象
        dataloader = DataLoader(
            faceDataset, batch_size=512, shuffle=True, num_workers=0, drop_last=True
        )
        loss = 0
        self.net.train()
        # epoch_num = 0

        for epoch_num in range(1, max_epoch + 1):
            iterations = 0
            cla_label = []
            cla_out = []
            offset_label = []
            offset_out = []
            # epoch_num += 1
            pbar = tqdm(dataloader)
            for i, (img_data_, category_, offset_) in enumerate(pbar):
                img_data_ = img_data_.to(self.device)  # 得到的三个值传入到CPU或者GPU
                category_ = category_.to(self.device)
                offset_ = offset_.to(self.device)

                _output_category, _output_offset = self.net(
                    img_data_
                )  # 输出置信度和偏移值

                output_category = _output_category.view(-1, 1)  # 转化成NV结构
                output_offset = _output_offset.view(-1, 4)

                # 正样本和负样本用来训练置信度
                category_mask = torch.lt(category_, 2)
                category = torch.masked_select(category_, category_mask)
                output_category = torch.masked_select(output_category, category_mask)
                cls_loss = self.cls_loss_fn(output_category, category)

                offset_mask = torch.gt(category_, 0)
                offset = torch.masked_select(offset_, offset_mask)
                output_offset = torch.masked_select(output_offset, offset_mask)
                offset_loss = self.offset_loss_fn(output_offset, offset)

                loss = cls_loss + offset_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                cls_loss = cls_loss.cpu().item()
                offset_loss = offset_loss.cpu().item()
                loss = loss.cpu().item()
                if iterations % 10 == 0:
                    pbar.set_description(
                        f'epoch: {epoch_num} iter {iterations} loss {loss:.3f} cls loss {cls_loss:.3f} offset loss {offset_loss:.3f}'
                    )
                iterations += 1

                cla_out.extend(output_category.detach().cpu())
                cla_label.extend(category.detach().cpu())
                offset_out.extend(output_offset.detach().cpu())
                offset_label.extend(offset.detach().cpu())
                cla_out = []
                cla_label.clear()
                offset_out.clear()
                offset_label.clear()

            torch.save(self.net.state_dict(), self.save_path)
            print("save success")

            if loss < stop_value:
                break


from torchvision.transforms import transforms


class Detector:

    def __init__(self, lownet_path, midnet_path, highnet_path, device) -> None:
        self.device = device
        self.lownet = LowNet()
        self.midnet = MidNet()
        self.highnet = HighNet()

        for net, path in zip(
            [self.lownet, self.midnet, self.highnet],
            [lownet_path, midnet_path, highnet_path],
        ):
            state_dict = torch.load(path, map_location='cpu')
            net.load_state_dict(state_dict)
            net.to(device)
            net.eval()

        self.trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    @torch.no_grad()
    def detect(
        self,
        image,
        conf_thresholds=[0.6, 0.6, 0.97],
        nms_thresholds=[0.3, 0.3, 0.3],
        low_pred_scaler=0.709,
    ):
        # print("===================")
        lownet_boxes = self._lownet_detect(
            image, conf_thresholds[0], nms_thresholds[0], low_pred_scaler
        )
        # print("***********************")
        if lownet_boxes.shape[0] == 0:
            return np.array([])

        # return lownet_boxes

        midnet_boxes = self._midnet_detect(
            image, lownet_boxes, conf_thresholds[1], nms_thresholds[1]
        )  # p网络输出的框和原图像输送到R网络中，O网络将框扩为正方形再进行裁剪，再缩放
        # print( midnet_boxes)
        if midnet_boxes.shape[0] == 0:
            return np.array([])

        # return midnet_boxes

        highnet_boxes = self._highnet_detect(
            image, midnet_boxes, conf_thresholds[2], nms_thresholds[2]
        )
        if highnet_boxes.shape[0] == 0:
            return np.array([])

        return highnet_boxes

    def _lownet_detect(self, img, conf_threshold, nms_threshold, scaler):

        boxes = []
        w, h = img.size
        min_side_len = min(w, h)

        scale = 1

        while min_side_len >= 12:
            img_data = self.trans(img).to(self.device)
            # img_data = img_data.unsqueeze_(0)
            img_data.unsqueeze_(0)

            _cls, _offest = self.lownet(img_data)  # NCHW
            # print(_cls.shape)    #torch.Size([1, 1, 1290, 1938])
            # print(_offest.shape)    #torch.Size([1, 4, 1290, 1938])

            cls, offest = _cls[0][0].cpu().data, _offest[0].cpu().data
            # _cls[0][0].cpu().data去掉NC，  _offest[0]去掉N
            # print(_cls.shape)       #torch.Size([1, 1, 1290, 1938])
            # print(_offest.shape)     #torch.Size([1, 4, 1290, 1938])

            idxs = torch.nonzero(
                torch.gt(cls, conf_threshold)
            )  # 取出置信度大于0.6的索引
            # print(idxs.shape)   #N2     #torch.Size([4639, 2])

            for idx in idxs:  # idx里面就是一个h和一个w
                # print(idx)    #tensor([ 102, 1904])
                # print(offest)
                boxes.append(self._box(idx, offest, cls[idx[0], idx[1]], scale))  # 反算
            scale *= scaler
            _w = int(w * scale)
            _h = int(h * scale)

            img = img.resize((_w, _h))
            # print(min_side_len)
            min_side_len = np.minimum(_w, _h)
        return nms(np.array(boxes), nms_threshold)

    def _box(
        self, start_index, offset, cls, scale, stride=2, side_len=12
    ):  # side_len=12建议框大大小

        _x1 = int(start_index[1] * stride) / scale  # 宽，W，x
        _y1 = int(start_index[0] * stride) / scale  # 高，H,y
        _x2 = int(start_index[1] * stride + side_len) / scale
        _y2 = int(start_index[0] * stride + side_len) / scale

        ow = _x2 - _x1  # 偏移量
        oh = _y2 - _y1

        _offset = offset[:, start_index[0], start_index[1]]  # 通道层面全都要[C, H, W]

        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return [x1, y1, x2, y2, cls]

    def _midnet_detect(self, image, lownet_boxes, conf_threshold, nms_threshold):

        _img_dataset = []
        _lownet_boxes = convert_to_square(lownet_boxes)
        for _box in _lownet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((24, 24))
            img_data = self.trans(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset).to(self.device)

        _cls, _offset = self.midnet(img_dataset)
        _cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()
        boxes = []

        idxs, _ = np.where(_cls > conf_threshold)
        for idx in idxs:  # 只是取出合格的
            _box = _lownet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]
            cls = _cls[idx][0]

            boxes.append([x1, y1, x2, y2, cls])
            # print(len(nms(np.array(boxes), 0.3)))
        # print("""""" """""" """""" """""" """""" """""" """""")

        return nms(np.array(boxes), nms_threshold)

    def _highnet_detect(self, image, midnet_boxes, conf_threshold, nms_threshold):

        _img_dataset = []
        _midnet_boxes = convert_to_square(midnet_boxes)
        for _box in _midnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self.trans(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset).to(self.device)

        _cls, _offset = self.highnet(img_dataset)

        _cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()

        boxes = []
        idxs, _ = np.where(_cls > conf_threshold)
        for idx in idxs:
            _box = _midnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]
            cls = _cls[idx][0]

            boxes.append([x1, y1, x2, y2, cls])

        return nms(np.array(boxes), nms_threshold, isMin=True)
