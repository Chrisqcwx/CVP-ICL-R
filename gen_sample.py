import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision import transforms
import time
import os
import os
import torch
import sys

sys.path.append('.')
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
save_ckpt_dir = '/data/yhy/Model-Inversion-Attack-ToolBox/test/recognition/learn/ckpts'
P_PATH = os.path.join(save_ckpt_dir, 'P.pth')
R_PATH = os.path.join(save_ckpt_dir, 'R.pth')
O_PATH = os.path.join(save_ckpt_dir, 'O.pth')

trans = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

from tqdm import tqdm
from cvpiclr.models.detect import Detector

if __name__ == '__main__':
    x = time.time()
    with torch.no_grad() as grad:
        # src_path = r"/data/yhy/datasets/ffhq128/00000/"  # 遍历文件夹内的图片
        src_path = (
            '/data/yhy/Model-Inversion-Attack-ToolBox/test/recognition/learn/sample'
        )
        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            dst_path = f'./results/{alpha}'
            os.makedirs(dst_path, exist_ok=True)
            for name in tqdm(os.listdir(src_path)):
                img = os.path.join(src_path, name)
                image_file = img
                # image_file = r"1.jpg"
                # print(image_file)
                detector = Detector(P_PATH, R_PATH, O_PATH, 'cuda')

                with Image.open(image_file) as im:
                    # im_tensor = trans(im)
                    im = im.convert('RGB')
                    boxes = detector.detect(im, low_pred_scaler=alpha)
                    # print(im,"==========================")
                    # print(boxes.shape)
                    imDraw = ImageDraw.Draw(im)
                    for box in boxes:
                        x1 = int(box[0])
                        y1 = int(box[1])
                        x2 = int(box[2])
                        y2 = int(box[3])

                        # print(x1)
                        # print(y1)
                        # print(x2)
                        # print(y2)

                        # print(box[4])
                        cls = box[4]
                        imDraw.rectangle((x1, y1, x2, y2), outline='red', width=2)

                        newim = im.crop((x1, y1, x2, y2))
                        newim.save('crop.png')
                        # font = ImageFont.truetype(r"C:\Windows\Fonts\simhei", size=20)
                        # font = ImageFont.
                        # imDraw.text((x1, y1), "{:.3f}".format(cls), fill="red", font=font)
                    y = time.time()
                    # print(y - x)
                    # im.show()
                    im.save(os.path.join(dst_path, name))

                    # im = im.convert('RGB')
