from typing import Any, Callable, Dict, List, Tuple, Optional
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
import os


from torch.utils import data
import torch
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision import transforms
import numpy as np
from typing import Literal
from .preprocess import parse_anno_file


class LabelDatasetFolder(DatasetFolder):
    """A label data loader.

    The subfolder of the root should be named as the label number.

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any],
        extensions: Tuple[str, ...] | None = None,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        is_valid_file: Callable[[str], bool] | None = None,
    ) -> None:
        super().__init__(
            root, loader, extensions, transform, target_transform, is_valid_file
        )

        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(
            (
                entry.name
                for entry in os.scandir(directory)
                if entry.is_dir() and entry.name.isalnum()
            ),
            key=lambda x: int(x),
        )
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: int(cls_name) for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class LabelImageFolder(LabelDatasetFolder):
    """A generic data loader where the images are arranged in this way by default: ::

        root/0/xxx.png
        root/0/xxy.png

        root/1/123.png
        root/1/nsdf3.png

    This class inherits from :class:`LabelDatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples


class ClassificationDataset(LabelImageFolder):

    def __init__(
        self,
        anno_file_path,
        data_path,
        mode: Literal['train', 'val', 'test'] = 'train',
        labeled=True,
    ):
        self.anno_infos = parse_anno_file(anno_file_path)
        subfolder = 'labeled' if labeled else 'unlabeled'
        data_path = os.path.join(data_path, mode, subfolder)
        # trans = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             [
        #                 0.5,
        #             ],
        #             [
        #                 0.5,
        #             ],
        #         ),
        #     ]
        # )
        super().__init__(data_path, None)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        idx = int(os.path.split(path)[1][:-4])
        anno_info = self.anno_infos[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (
            sample,
            target,
            (
                anno_info[0],
                anno_info[1],
                anno_info[0] + anno_info[2],
                anno_info[1] + anno_info[3],
            ),
        )


class DetectionDataset(Dataset):
    def __init__(
        self,
        data_path,
        mode: Literal['train', 'val', 'test'] = 'train',
        resolution: int = 12,
        transform=None,
    ):
        data_path = os.path.join(data_path, mode, str(resolution))
        self.dataset = []
        l1 = open(os.path.join(data_path, "negative.txt")).readlines()
        for l1_filename in l1:
            self.dataset.append(
                [
                    os.path.join(data_path, l1_filename.split(" ")[0]),
                    l1_filename.split(" ")[1:6],
                ]
            )
            # print(self.dataset)
        # exit()
        l2 = open(os.path.join(data_path, "positive.txt")).readlines()
        for l2_filename in l2:
            self.dataset.append(
                [
                    os.path.join(data_path, l2_filename.split(" ")[0]),
                    l2_filename.split(" ")[1:6],
                ]
            )
        l3 = open(os.path.join(data_path, "part.txt")).readlines()
        for l3_filename in l3:
            self.dataset.append(
                [
                    os.path.join(data_path, l3_filename.split(" ")[0]),
                    l3_filename.split(" ")[1:6],
                ]
            )

        self.trans = transform
        # transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             [
        #                 0.5,
        #             ],
        #             [
        #                 0.5,
        #             ],
        #         ),
        #     ]
        # )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        # print(data[0])
        img_tensor = Image.open(data[0])
        if self.trans is not None:
            img_tensor = self.trans(img_tensor)
        category = torch.tensor(float(data[1][0])).reshape(-1)
        offset = torch.tensor(
            [float(data[1][1]), float(data[1][2]), float(data[1][3]), float(data[1][4])]
        )

        return img_tensor, category, offset
