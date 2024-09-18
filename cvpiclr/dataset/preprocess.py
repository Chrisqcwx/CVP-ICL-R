import os
from PIL import Image
import numpy as np
import traceback
from ..utils import box
from tqdm import tqdm

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def walk_imgs(path):
    """Traverse all images in the specified path.

    Args:
        path (_type_): The specified path.

    Returns:
        List: The list that collects the paths for all the images.
    """

    img_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(IMG_EXTENSIONS):
                img_paths.append(os.path.join(root, file))
    return img_paths


float_num = [
    0.1,
    0.5,
    0.5,
    0.5,
    0.9,
    0.9,
    0.9,
    0.9,
    0.9,
]  # 控制正负样本比例，（控制比例？）


def gen_sample(face_size, anno_map, mode, save_dir, image_dir):
    stop_value = 1e9
    print("gen size:{} image".format(face_size))
    save_path = os.path.join(save_dir, mode)
    image_dir = os.path.join(image_dir, mode)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    positive_image_dir = os.path.join(
        save_path, str(face_size), "positive"
    )  # 仅仅生成路径名
    negative_image_dir = os.path.join(save_path, str(face_size), "negative")
    part_image_dir = os.path.join(save_path, str(face_size), "part")

    for dir_path in [
        positive_image_dir,
        negative_image_dir,
        part_image_dir,
    ]:  # 生成路径
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    positive_anno_filename = os.path.join(save_path, str(face_size), "positive.txt")
    negative_anno_filename = os.path.join(save_path, str(face_size), "negative.txt")
    part_anno_filename = os.path.join(save_path, str(face_size), "part.txt")

    positive_count = 0
    negative_count = 0
    part_count = 0

    try:  # 抛出异常
        positive_anno_file = open(positive_anno_filename, "w")
        negative_anno_file = open(negative_anno_filename, "w")
        part_anno_file = open(part_anno_filename, "w")

        for i, image_file in enumerate(
            tqdm(walk_imgs(image_dir))
        ):  # txt开头的两行文件不是路径和标签，需要跳过
            # if i < 2:
            #     continue
            try:
                image_filename = str(os.path.split(image_file)[1])
                # image_file = os.path.join(img_dir, image_filename)

                with Image.open(image_file) as img:
                    img_w, img_h = img.size  # 原图
                    # x1 = float(strs[1].strip())
                    # y1 = float(strs[2].strip())
                    # w = float(strs[3].strip())  # 人脸框
                    # h = float(strs[4].strip())
                    x1, y1, w, h = anno_map[int(image_filename[:-4])]
                    x2 = float(x1 + w)
                    y2 = float(y1 + h)

                    if x1 < 0 or y1 < 0 or w < 0 or h < 0:  # 跳过坐标值为负数的
                        continue

                    boxes = [
                        [x1, y1, x2, y2]
                    ]  # 当前真实框四个坐标（根据中心点偏移）， 二维数组便于IOU计算

                    # 求中心点坐标
                    cx = x1 + w / 2
                    cy = y1 + h / 2
                    side_len = max(w, h)
                    seed = float_num[
                        np.random.randint(0, len(float_num))
                    ]  # 取0到9之间的随机数作为索引     #len(float_num) = 9 #float_num = [0.1, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9]
                    count = 0
                    for _ in range(4):
                        _side_len = side_len + np.random.randint(
                            int(-side_len * seed), int(side_len * seed)
                        )  # 生成框
                        _cx = cx + np.random.randint(
                            int(-cx * seed), int(cx * seed)
                        )  # 中心点作偏移
                        _cy = cy + np.random.randint(int(-cy * seed), int(cy * seed))

                        _x1 = _cx - _side_len / 2  # 左上角
                        _y1 = _cy - _side_len / 2
                        _x2 = _x1 + _side_len  # 右下角
                        _y2 = _y1 + _side_len

                        if (
                            _x1 < 0 or _y1 < 0 or _x2 > img_w or _y2 > img_h
                        ):  # 左上角的点是否偏移到了框外边，右下角的点大于图像的宽和高
                            continue

                        offset_x1 = (x1 - _x1) / _side_len  # 得到四个偏移量
                        offset_y1 = (y1 - _y1) / _side_len
                        offset_x2 = (x2 - _x2) / _side_len
                        offset_y2 = (y2 - _y2) / _side_len

                        crop_box = [_x1, _y1, _x2, _y2]
                        face_crop = img.crop(crop_box)  # 图片裁剪
                        face_resize = face_crop.resize(
                            (face_size, face_size)
                        )  # 对裁剪后的图片缩放

                        iou = box.iou(crop_box, np.array(boxes))[0]
                        if iou > 0.65:  # 可以自己修改
                            positive_anno_file.write(
                                "positive/{0}.jpg {1} {2} {3} {4} {5}\n".format(
                                    positive_count,
                                    1,
                                    offset_x1,
                                    offset_y1,
                                    offset_x2,
                                    offset_y2,
                                )
                            )
                            positive_anno_file.flush()  # flush() 方法是用来刷新缓冲区的，即将缓冲区中的数据立刻写入文件，同时清空缓冲区
                            face_resize.save(
                                os.path.join(
                                    positive_image_dir, "{0}.jpg".format(positive_count)
                                )
                            )
                            # print("positive_count",positive_count)
                            positive_count += 1
                        elif 0.65 > iou > 0.4:
                            part_anno_file.write(
                                "part/{0}.jpg {1} {2} {3} {4} {5}\n".format(
                                    part_count,
                                    2,
                                    offset_x1,
                                    offset_y1,
                                    offset_x2,
                                    offset_y2,
                                )
                            )
                            part_anno_file.flush()
                            face_resize.save(
                                os.path.join(
                                    part_image_dir, "{0}.jpg".format(part_count)
                                )
                            )
                            # print("part_count", part_count)
                            part_count += 1
                        elif iou < 0.1:
                            negative_anno_file.write(
                                "negative/{0}.jpg {1} 0 0 0 0\n".format(
                                    negative_count, 0
                                )
                            )
                            negative_anno_file.flush()
                            face_resize.save(
                                os.path.join(
                                    negative_image_dir, "{0}.jpg".format(negative_count)
                                )
                            )
                            # print("negative_count", negative_count)
                            negative_count += 1

                        count = positive_count + part_count + negative_count
                        # print(count)
                    if count >= stop_value:
                        break

            except:
                traceback.print_exc()  # 返回错误类型
    finally:
        positive_anno_file.close()
        negative_anno_file.close()
        part_anno_file.close()


def parse_anno_file(filename):
    filename2box = {}
    with open(filename) as f:
        f.readline()
        f.readline()
        data = f.readlines()
    for line in data:
        strs = line.split(' ')
        strs = [s for s in strs if '' != s]
        if len(strs) < 5:
            continue

        fname = strs[0].strip()
        # print(strs, fname)
        x1 = float(strs[1].strip())
        y1 = float(strs[2].strip())
        w = float(strs[3].strip())  # 人脸框
        h = float(strs[4].strip())
        filename2box[int(fname[:-4])] = [x1, y1, w, h]
    return filename2box


def preprocess(anno_file, split_src, save_dir):
    anno_map = parse_anno_file(anno_file)

    for resolution in [12, 24, 48, 128]:
        for mode in ['train', 'val', 'test']:
            gen_sample(resolution, anno_map, mode, save_dir, split_src)
