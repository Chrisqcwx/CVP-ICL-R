import os
import inspect
import functools
import torch
from .evolve import evolve
from typing import *


def safe_save(obj, save_dir: str, save_name: Optional[str] = None):
    """Save the obj by using torch.save function.

    Args:
        obj (_type_): The objective to save.
        save_dir (str): The directory path.
        save_name (Optional[str], optional): The file name for the objective to save. Defaults to None.
    """

    if save_name is None:
        save_dir, save_name = os.path.split(save_dir)
    if save_dir.strip() != '':
        os.makedirs(save_dir, exist_ok=True)
    torch.save(obj, os.path.join(save_dir, save_name))


class ConfigMixin:
    """
       A Mixin to save parameters from `__init__` function. Inherit the `ConfigMixin` class and add the decorator `@register_to_config_init` to the `__init__` function.

       The workflow of the class are as follows.
                       +------------------------------+
                       |                              |
                       |      Initial Parameters      |
                       |                              |
                       +-----------+-----^------------+
                                   |     |
          register_to_config_init  |     |  __init__
                                   |     |
                       +-----------v-----+------------+
                       |                              |
                       |        Loaded Config         |
                       |                              |
                       +-----------+-----^------------+
                                   |     |
    preprocess_config_before_save  |     |  postprocess_config_after_load
                                   |     |
                       +-----------v-----+------------+
                       |                              |
                       |         Saved Config         |
                       |                              |
                       +------------------------------+
    """

    def preprocess_config_before_save(self, config):
        return config

    @staticmethod
    def postprocess_config_after_load(config):
        return config

    def register_to_config(self, **config_dict):
        self._config_mixin_dict = config_dict

    def save_config(self, save_path: str):
        # os.makedirs(save_path, exist_ok=True)
        safe_save(
            self.preprocess_config_before_save(self._config_mixin_dict), save_path
        )

    @staticmethod
    def load_config(config_path: str):
        if not os.path.exists(config_path):
            raise RuntimeError(f'config_path {config_path} is not existed.')

        kwargs = torch.load(config_path, map_location='cpu')
        return ConfigMixin.postprocess_config_after_load(kwargs)

    @staticmethod
    def register_to_config_init(init):
        """Decorator of `__init__` method of classses inherit from `ConfigMixin`. Automatically save the init parameters."""

        @functools.wraps(init)
        def inner_init(self, *args, **kwargs):

            # Ignore private kwargs in the init.
            init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
            config_init_kwargs = {k: v for k, v in kwargs.items() if k.startswith("_")}
            if not isinstance(self, ConfigMixin):
                raise RuntimeError(
                    f"`@register_to_config_init` was applied to {self.__class__.__name__} init method, but this class does "
                    "not inherit from `ConfigMixin`."
                )

            # Get positional arguments aligned with kwargs
            new_kwargs = {}
            signature = inspect.signature(init)
            parameters = {
                name: p.default
                for i, (name, p) in enumerate(signature.parameters.items())
                if i > 0
            }
            for arg, name in zip(args, parameters.keys()):
                new_kwargs[name] = arg

            # Then add all kwargs
            new_kwargs.update(
                {
                    k: init_kwargs.get(k, default)
                    for k, default in parameters.items()
                    if k not in new_kwargs
                }
            )

            # Take note of the parameters that were not present in the loaded config
            # if len(set(new_kwargs.keys()) - set(init_kwargs)) > 0:
            #     new_kwargs["_use_default_values"] = list(
            #         set(new_kwargs.keys()) - set(init_kwargs)
            #     )

            new_kwargs = {**config_init_kwargs, **new_kwargs}
            # getattr(self, "register_to_config")(**new_kwargs)
            self.register_to_config(**new_kwargs)
            init(self, *args, **init_kwargs)

        return inner_init


import os
import json

import torch
from torch.nn import Module


class ModelMixin(Module, ConfigMixin):

    # def save_config(self, save_path: str):
    #     os.makedirs(save_path, exist_ok=True)
    #     with open(save_path, 'w', encoding='utf8') as f:
    #         json.dump(f, self._config_mixin_dict)

    # @staticmethod
    # def load_config(config_path: str):
    #     if not os.path.exists(config_path):
    #         raise RuntimeError(f'config_path {config_path} is not existed.')

    #     with open(config_path, 'r', encoding='utf8') as f:
    #         kwargs = json.load(config_path)

    #     return kwargs

    def save_pretrained(self, path, **add_infos):
        save_result = {
            'state_dict': self.state_dict(),
            'config': self.preprocess_config_before_save(self._config_mixin_dict),
            **add_infos,
        }
        safe_save(save_result, path)

    @classmethod
    def from_pretrained(cls, data_or_path, **config_kwargs):

        if isinstance(data_or_path, str):
            data: dict = torch.load(data_or_path, map_location='cpu')
        else:
            data = data_or_path

        kwargs = cls.postprocess_config_after_load(data['config'])
        for k in config_kwargs:
            kwargs[k] = config_kwargs[k]
        init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        model = cls(**init_kwargs)

        if 'state_dict' in data:
            state_dict = data['state_dict']
            if state_dict is not None:
                # print(f'load state dict')
                model.load_state_dict(state_dict)

        return model


import importlib
from abc import abstractmethod
from copy import deepcopy
from typing import Callable, Optional, Any
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as tvmodel
import torchvision.transforms.functional as TF
from torchvision.models.inception import InceptionOutputs


HOOK_NAME_FEATURE = 'feature'
HOOK_NAME_HIDDEN = 'hidden'
HOOK_NAME_DEEPINVERSION_BN = 'deepinversion_bn'

BUILDIN_CLASSIFIERS = {}
CLASSNAME_TO_NAME_MAPPING = {}
TORCHVISION_MODEL_NAMES = tvmodel.list_models()


def register_model(name: Optional[str] = None):
    """Register model for construct.

    Args:
        name (Optional[str], optional): The key of the model. Defaults to None.
    """

    def wrapper(c):
        key = name if name is not None else c.__name__
        CLASSNAME_TO_NAME_MAPPING[c.__name__] = key
        if key in BUILDIN_CLASSIFIERS:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        BUILDIN_CLASSIFIERS[key] = c
        return c

    return wrapper


class ModelConstructException(Exception):
    pass


def construct_classifiers_by_name(name: str, **kwargs):

    if name in BUILDIN_CLASSIFIERS:
        return BUILDIN_CLASSIFIERS[name](**kwargs)

    if name in TORCHVISION_MODEL_NAMES:
        return TorchvisionClassifierModel(name, **kwargs)

    raise ModelConstructException(f'Module name {name} not found.')


def list_classifiers():
    """List all valid module names"""
    return sorted(BUILDIN_CLASSIFIERS.keys()) + TORCHVISION_MODEL_NAMES


def auto_classifier_from_pretrained(data_or_path, **kwargs):

    if isinstance(data_or_path, str):
        data = torch.load(data_or_path, map_location='cpu')
    else:
        data = data_or_path
    if 'model_name' not in data:
        raise RuntimeError('model_name is not contained in the data')

    cls: BaseImageClassifier = BUILDIN_CLASSIFIERS[data['model_name']]
    return cls.from_pretrained(data_or_path, **kwargs)


class BaseImageModel(ModelMixin):

    def __init__(self, resolution: int, feature_dim: int, **kwargs) -> None:
        nn.Module.__init__(self, **kwargs)

        self._resolution = resolution
        self._feature_dim = feature_dim
        self._inner_hooks = {}

    @property
    def resolution(self):
        return self._resolution

    @property
    def feature_dim(self):
        return self._feature_dim

    def _check_hook(self, name: str):
        if name not in self._inner_hooks:
            raise RuntimeError(f'The model do not have feature for `{name}`')

    def register_hook_for_forward(self, name: str, hook):
        self._inner_hooks[name] = hook

    @abstractmethod
    def _forward_impl(self, image: torch.Tensor, *args, **kwargs):
        raise NotImplementedError()

    def save_pretrained(self, path, **add_infos):
        return super().save_pretrained(
            path,
            model_name=CLASSNAME_TO_NAME_MAPPING[self.__class__.__name__],
            **add_infos,
        )

    def forward(self, image: torch.Tensor, *args, **kwargs):

        if image.shape[-1] != self.resolution or image.shape[-2] != self.resolution:
            image = TF.resize(image, (self.resolution, self.resolution), antialias=True)

        forward_res = self._forward_impl(image, *args, **kwargs)
        hook_res = {k: v.get_feature() for k, v in self._inner_hooks.items()}
        if isinstance(forward_res, tuple) and not isinstance(
            forward_res, InceptionOutputs
        ):
            if len(forward_res) != 2:
                raise RuntimeError(
                    f'The number of model output must be 1 or 2, but found {len(forward_res)}'
                )
            forward_res, forward_addition = forward_res
            if forward_addition is not None:
                for k, v in forward_addition.items():
                    if k in hook_res:
                        raise RuntimeError('hook result key conflict')
                    hook_res[k] = v
        return forward_res, hook_res


class BaseImageEncoder(BaseImageModel):

    def __init__(self, resolution: int, feature_dim: int, **kwargs) -> None:
        super().__init__(resolution, feature_dim, **kwargs)


class BaseImageClassifier(BaseImageModel):

    def __init__(
        self,
        resolution,
        feature_dim,
        num_classes,
        register_last_feature_hook=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(resolution, feature_dim, *args, **kwargs)
        self._num_classes = num_classes

        # self._feature_flag = False

        # self.register_last_feature_hook = register_last_feature_hook

    @property
    def num_classes(self):
        return self._num_classes

    # def get_last_feature_hook(self) -> BaseHook:
    #     return None

    def preprocess_config_before_save(self, config):
        config = deepcopy(config)
        if 'register_last_feature_hook' in config:
            del config['register_last_feature_hook']
        return super().preprocess_config_before_save(config)

    def forward(self, image: torch.Tensor, *args, **kwargs):
        # if not self._feature_flag and self.register_last_feature_hook:
        #     self._feature_flag = True
        #     hook = self.get_last_feature_hook()
        #     if hook is None:
        #         raise RuntimeError('The last feature hook is not set.')
        #     self.register_hook_for_forward(HOOK_NAME_FEATURE, hook=hook)
        return super().forward(image, *args, **kwargs)


def _operate_fc_impl(
    module: nn.Module, reset_num_classes: int = None, visit_fc_fn: Callable = None
):
    """Reset the output class num of nn.Linear and return the input feature_dim of nn.Linear.

    Args:
        module (nn.Module): The specific model structure.
        reset_num_classes (int, optional): The new output class num. Defaults to None.
        visit_fc_fn (Callable, optional): Other operations to the nn.Linear of the input module. Defaults to None.

    Returns:
        feature_dim (int): The input feature_dim of nn.Linear.
    """

    if isinstance(module, nn.Sequential):

        if len(module) == 0:
            raise ModelConstructException('fail to implement')

        if isinstance(module[-1], nn.Linear):
            feature_dim = module[-1].weight.shape[-1]

            if (
                reset_num_classes is not None
                and reset_num_classes != module[-1].weight.shape[0]
            ):
                module[-1] = nn.Linear(feature_dim, reset_num_classes)

            if visit_fc_fn is not None:
                visit_fc_fn(module[-1])

            return feature_dim
        else:
            return _operate_fc_impl(module[-1], reset_num_classes)

    children = list(module.named_children())
    if len(children) == 0:
        raise ModelConstructException('fail to implement')
    attr_name, child_module = children[-1]
    if isinstance(child_module, nn.Linear):
        feature_dim = child_module.weight.shape[-1]

        if (
            reset_num_classes is not None
            and reset_num_classes != child_module.weight.shape[0]
        ):
            setattr(module, attr_name, nn.Linear(feature_dim, reset_num_classes))

        if visit_fc_fn is not None:
            visit_fc_fn(getattr(module, attr_name))

        return feature_dim
    else:
        return _operate_fc_impl(child_module, reset_num_classes)


def operate_fc(
    module: nn.Module, reset_num_classes: int = None, visit_fc_fn: Callable = None
) -> int:
    return _operate_fc_impl(module, reset_num_classes, visit_fc_fn)


@register_model('torchvision')
class TorchvisionClassifierModel(BaseImageClassifier):

    @ModelMixin.register_to_config_init
    def __init__(
        self,
        arch_name: str,
        num_classes: int,
        resolution=224,
        weights=None,
        arch_kwargs={},
        register_last_feature_hook=False,
    ) -> None:
        # weights: None, 'IMAGENET1K_V1', 'IMAGENET1K_V2' or 'DEFAULT'

        self._feature_hook = None

        _output_transform = None
        if register_last_feature_hook:

            def _output_transform(m: nn.Linear):
                # self._feature_hook = FirstInputHook(m)
                def hook_fn(module, input, output):
                    return output, {HOOK_NAME_FEATURE: input[0]}

                m.register_forward_hook(hook_fn)

        tv_module = importlib.import_module('torchvision.models')
        factory = getattr(tv_module, arch_name, None)
        if factory is None:
            raise RuntimeError(f'torchvision do not support model {arch_name}')
        model = factory(weights=weights, **arch_kwargs)

        feature_dim = operate_fc(model, num_classes, _output_transform)

        super().__init__(
            resolution, feature_dim, num_classes, register_last_feature_hook
        )

        self.model = model

    def _forward_impl(self, image: torch.Tensor, *args, **kwargs):
        return self.model(image)

    # def get_last_feature_hook(self) -> BaseHook:
    #     return self._feature_hook


@register_model('resnest')
class ResNeSt(BaseImageClassifier):

    @ModelMixin.register_to_config_init
    def __init__(
        self,
        arch_name: str,
        num_classes: int,
        pretrained=False,
        arch_kwargs={},
        register_last_feature_hook=False,
    ) -> None:
        # weights: None, 'IMAGENET1K_V1', 'IMAGENET1K_V2' or 'DEFAULT'

        self._feature_hook = None

        _output_transform = None
        if register_last_feature_hook:

            def _output_transform(m: nn.Linear):
                # self._feature_hook = FirstInputHook(m)
                def hook_fn(module, input, output):
                    return output, {HOOK_NAME_FEATURE: input[0]}

                m.register_forward_hook(hook_fn)

        try:
            tv_module = importlib.import_module('resnest.torch')
        except ModuleNotFoundError as e:
            raise RuntimeError(
                'ResNeSt module not found. Please install the module by `pip install git+https://github.com/zhanghang1989/ResNeSt`'
            )
        factory = getattr(tv_module, arch_name, None)
        if factory is None:
            raise RuntimeError(f'ResNeSt do not support model {arch_name}')
        model = factory(pretrained=pretrained, **arch_kwargs)

        feature_dim = operate_fc(model, num_classes, _output_transform)

        super().__init__(224, feature_dim, num_classes, register_last_feature_hook)

        self.model = model

    def _forward_impl(self, image: torch.Tensor, *args, **kwargs):
        return self.model(image)


from copy import deepcopy

from torch import Tensor
import torchvision


@register_model('vgg16_64')
class VGG16_64(BaseImageClassifier):

    @ModelMixin.register_to_config_init
    def __init__(self, num_classes, pretrained=False, register_last_feature_hook=False):
        self.feat_dim = 512 * 2 * 2
        super(VGG16_64, self).__init__(
            64, self.feat_dim, num_classes, register_last_feature_hook
        )
        model = torchvision.models.vgg16_bn(pretrained=pretrained)
        self.feature = model.features

        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

    def preprocess_config_before_save(self, config):
        config = deepcopy(config)
        del config['pretrained']
        return super().preprocess_config_before_save(config)

    #     self.feature_hook = FirstInputHook(self.fc_layer)

    # def get_last_feature_hook(self) -> BaseHook:
    #     return self.feature_hook

    # def create_hidden_hooks(self) -> list:

    #     hiddens_hooks = []
    #     def _add_hook_fn(module):
    #         if isinstance(module, MaxPool2d):
    #             hiddens_hooks.append(OutputHook(module))
    #     traverse_module(self, _add_hook_fn, call_middle=False)
    #     return hiddens_hooks

    # def freeze_front_layers(self) -> None:

    #     freeze_num = 8
    #     i = 0
    #     for m in self.feature.children():

    #         if isinstance(m, nn.Conv2d):
    #             i += 1
    #             if i >= freeze_num:
    #                 break
    #         for p in m.parameters():
    #             p.requires_grad_(False)

    def _forward_impl(self, x: Tensor, *args, **kwargs):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)

        return res, {HOOK_NAME_FEATURE: feature}


# class Flatten(nn.Module):
#     def forward(self, input):
#         return input.view(input.size(0), -1)


@register_model(name='ir152_64')
class IR152_64(BaseImageClassifier):

    @ModelMixin.register_to_config_init
    def __init__(
        self,
        num_classes=1000,
        register_last_feature_hook=False,
        backbone_path: Optional[str] = None,
    ):
        self.feat_dim = 512
        super(IR152_64, self).__init__(
            64, self.feat_dim, num_classes, register_last_feature_hook
        )
        self.feature = evolve.IR_152_64((64, 64))
        if backbone_path is not None:
            state_dict = torch.load(backbone_path, map_location='cpu')
            for k in list(state_dict.keys()):
                if 'output_layer' in k:
                    del state_dict[k]
            self.feature.load_state_dict(state_dict)

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 512),
            nn.BatchNorm1d(512),
        )

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

        # self.feature_hook = FirstInputHook(self.fc_layer)

    def preprocess_config_before_save(self, config):
        config = deepcopy(config)
        del config['backbone_path']
        return super().preprocess_config_before_save(config)

    # def get_last_feature_hook(self) -> BaseHook:
    #     return self.feature_hook

    # def create_hidden_hooks(self) -> list:

    #     hiddens_hooks = []

    #     length_hidden = len(self.feature.body)

    #     num_body_monitor = 4
    #     offset = length_hidden // num_body_monitor
    #     for i in range(num_body_monitor):
    #         hiddens_hooks.append(OutputHook(self.feature.body[offset * (i+1) - 1]))

    #     hiddens_hooks.append(OutputHook(self.output_layer))
    #     return hiddens_hooks

    # def freeze_front_layers(self) -> None:
    #     length_hidden = len(self.feature.body)
    #     for i in range(int(length_hidden * 2 // 3)):
    #         self.feature.body[i].requires_grad_(False)

    def _forward_impl(self, image: Tensor, *args, **kwargs):
        feat = self.feature(image)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return out, {HOOK_NAME_FEATURE: feat}


@register_model(name='facenet64')
class FaceNet64(BaseImageClassifier):

    @ModelMixin.register_to_config_init
    def __init__(
        self,
        num_classes=1000,
        register_last_feature_hook=False,
        backbone_path: Optional[str] = None,
    ):
        super(FaceNet64, self).__init__(
            64, 512, num_classes, register_last_feature_hook
        )
        self.feature = evolve.IR_50_64((64, 64))
        if backbone_path is not None:
            state_dict = torch.load(backbone_path, map_location='cpu')
            for k in list(state_dict.keys()):
                if 'output_layer' in k:
                    del state_dict[k]
            self.feature.load_state_dict(state_dict)
        self.feat_dim = 512
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 512),
            nn.BatchNorm1d(512),
        )

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

        # self.feature_hook = FirstInputHook(self.fc_layer)

    # def get_last_feature_hook(self) -> BaseHook:
    #     return self.feature_hook

    def preprocess_config_before_save(self, config):
        config = deepcopy(config)
        del config['backbone_path']
        return super().preprocess_config_before_save(config)

    def _forward_impl(self, image: Tensor, *args, **kwargs):
        feat = self.feature(image)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)

        return out, {HOOK_NAME_FEATURE: feat}


@register_model(name='efficientnet_b0_64')
class EfficientNet_b0_64(BaseImageClassifier):

    @ModelMixin.register_to_config_init
    def __init__(self, num_classes=1000, pretrained=False):
        super(EfficientNet_b0_64, self).__init__(64, 1280, num_classes, False)
        model = torchvision.models.efficientnet.efficientnet_b0(pretrained=pretrained)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.feat_dim = 1280
        self.fc_layer = nn.Linear(self.feat_dim, num_classes)

    def preprocess_config_before_save(self, config):
        config = deepcopy(config)
        del config['pretrained']
        return super().preprocess_config_before_save(config)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return res, {HOOK_NAME_FEATURE: feature}

    def get_feature_dim(self) -> int:
        return self.feat_dim


@register_model(name='efficientnet_b1_64')
class EfficientNet_b1_64(BaseImageClassifier):

    @ModelMixin.register_to_config_init
    def __init__(self, num_classes=1000, pretrained=False):
        super(EfficientNet_b1_64, self).__init__(64, 1280, num_classes, False)
        model = torchvision.models.efficientnet.efficientnet_b1(pretrained=pretrained)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.feat_dim = 1280
        self.fc_layer = nn.Linear(self.feat_dim, num_classes)

    def preprocess_config_before_save(self, config):
        config = deepcopy(config)
        del config['pretrained']
        return super().preprocess_config_before_save(config)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return res, {HOOK_NAME_FEATURE: feature}

    def get_feature_dim(self) -> int:
        return self.feat_dim


@register_model(name='efficientnet_b2_64')
class EfficientNet_b2_64(BaseImageClassifier):

    @ModelMixin.register_to_config_init
    def __init__(self, num_classes=1408, pretrained=False):
        super(EfficientNet_b2_64, self).__init__(64, 1280, num_classes, False)
        model = torchvision.models.efficientnet.efficientnet_b2(pretrained=pretrained)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.feat_dim = 1408
        self.fc_layer = nn.Linear(self.feat_dim, num_classes)

    def preprocess_config_before_save(self, config):
        config = deepcopy(config)
        del config['pretrained']
        return super().preprocess_config_before_save(config)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return res, {HOOK_NAME_FEATURE: feature}

    def get_feature_dim(self) -> int:
        return self.feat_dim


import os
import importlib
from dataclasses import field, dataclass
from abc import abstractmethod, ABC
from collections import OrderedDict
from copy import deepcopy
from typing import Optional, Iterator, Tuple, Callable, Sequence
import math

import torch
from torch import nn, Tensor, LongTensor
from torch.nn import Module
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torchvision.models.inception import InceptionOutputs
from tqdm import tqdm


# from ...models import BaseImageClassifier
# from ...utils import (
#     unwrapped_parallel_module,
#     ClassificationLoss,
#     obj_to_yaml,
#     print_as_yaml,
#     print_split_line,
#     DictAccumulator,
# )

import importlib
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F


def max_margin_loss(out, iden):
    real = out.gather(1, iden.unsqueeze(1)).squeeze(1)
    tmp1 = torch.argsort(out, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == iden, tmp1[:, -2], tmp1[:, -1])
    margin = out.gather(1, new_y.unsqueeze(1)).squeeze(1)

    return (-1 * real).mean() + margin.mean()


def poincare_loss(outputs, targets, xi=1e-4):
    # Normalize logits
    u = outputs / torch.norm(outputs, p=1, dim=-1).unsqueeze(1)
    # Create one-hot encoded target vector
    v = torch.clip(torch.eye(outputs.shape[-1])[targets.detach().cpu()] - xi, 0, 1)
    v = v.to(u.device)
    # Compute squared norms
    u_norm_squared = torch.norm(u, p=2, dim=1) ** 2
    v_norm_squared = torch.norm(v, p=2, dim=1) ** 2
    diff_norm_squared = torch.norm(u - v, p=2, dim=1) ** 2
    # Compute delta
    delta = 2 * diff_norm_squared / ((1 - u_norm_squared) * (1 - v_norm_squared))
    # Compute distance
    loss = torch.arccosh(1 + delta)
    return loss.mean()


_LOSS_MAPPING = {
    'ce': F.cross_entropy,
    'poincare': poincare_loss,
    'max_margin': max_margin_loss,
}


class LabelSmoothingCrossEntropyLoss:
    """The Cross Entropy Loss with label smoothing technique. Used in the LS defense method."""

    def __init__(self, label_smoothing: float = 0.0) -> None:
        self.label_smoothing = label_smoothing

    def __call__(self, inputs, labels):
        ls = self.label_smoothing
        confidence = 1.0 - ls
        logprobs = F.log_softmax(inputs, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + ls * smooth_loss
        return torch.mean(loss, dim=0).sum()


class TorchLoss:
    """Find loss function from 'torch.nn.functional' and 'torch.nn'"""

    def __init__(self, loss_fn: str | Callable, *args, **kwargs) -> None:
        # super().__init__()
        self.fn = None
        if isinstance(loss_fn, str):
            if loss_fn.lower() in _LOSS_MAPPING:
                self.fn = _LOSS_MAPPING[loss_fn.lower()]
            else:
                module = importlib.import_module('torch.nn.functional')
                fn = getattr(module, loss_fn, None)
                if fn is not None:
                    self.fn = lambda *arg, **kwd: fn(*arg, *args, **kwd, **kwargs)
                else:
                    module = importlib.import_module('torch.nn')
                    t = getattr(module, loss_fn, None)
                    if t is not None:
                        self.fn = t(*args, **kwargs)
                if self.fn is None:
                    raise RuntimeError(f'loss_fn {loss_fn} not found.')
        else:
            self.fn = loss_fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


ClassificationLoss = TorchLoss

import copy
from collections import defaultdict, OrderedDict

import torch


class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        self.data = [0] * n
        self.num = 0

    def add(self, *args, add_num=1, add_type='mean'):
        """adding data to the data list"""
        assert len(args) == len(self.data)
        mul_coef = add_num if add_type == 'mean' else 1
        self.num += add_num
        for i, add_item in enumerate(args):
            if isinstance(add_item, torch.Tensor):
                add_item = add_item.item()
            self.data[i] += add_item * mul_coef

    def reset(self):
        """reset all data to 0"""
        self.data = [0] * len(self.data)
        self.num = 0

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def avg(self, idx=None):
        """Calculate average of the data specified by `idx`. If idx is None, it will calculate average of all data.

        Args:
            idx (int, optional): subscript for the data list. Defaults to None.

        Returns:
            int | list: list if idx is None else int
        """
        num = 1 if self.num == 0 else self.num
        if idx is None:
            return [d / num for d in self.data]
        else:
            return self.data[idx] / num


class DictAccumulator:
    def __init__(self) -> None:
        self.data = OrderedDict()  # defaultdict(lambda : 0)
        self.num = 0

    def reset(self):
        """reset all data to 0"""
        self.data = OrderedDict()  # defaultdict(lambda : 0)
        self.num = 0

    def add(self, add_dic: OrderedDict, add_num=1, add_type='mean'):
        mul_coef = add_num if add_type == 'mean' else 1
        self.num += add_num
        for key, val in add_dic.items():
            if isinstance(val, torch.Tensor):
                val = val.item()
            if key not in self.data.keys():
                self.data[key] = 0
            self.data[key] += val * mul_coef

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def avg(self, key=None):
        num = 1 if self.num == 0 else self.num
        if key is None:
            res = copy.deepcopy(self.data)
            for k in self.data:
                res[k] /= num
            return res
        else:
            return self.data[key] / num


from ..utils.format import *


@dataclass
class BaseTrainConfig:

    experiment_dir: str
    save_name: str
    device: torch.device

    model: BaseImageClassifier
    optimizer: Optimizer
    lr_scheduler: Optional[LRScheduler] = None
    clip_grad_norm: Optional[float] = None

    save_per_epochs: int = 10


class BaseTrainer(ABC):

    def __init__(self, config: BaseTrainConfig, *args, **kwargs) -> None:
        self.config = config
        os.makedirs(config.experiment_dir, exist_ok=True)
        self.save_path = os.path.join(config.experiment_dir, config.save_name)

        self._epoch = 0
        self._iteration = 0

    @property
    def epoch(self):
        return self._epoch

    @property
    def iteration(self):
        return self._iteration

    @property
    def model(self):
        return self.config.model

    @property
    def optimizer(self):
        return self.config.optimizer

    @property
    def lr_scheduler(self):
        return self.config.lr_scheduler

    @abstractmethod
    def calc_loss(self, inputs, result, labels: torch.LongTensor):
        raise NotImplementedError()

    @torch.no_grad()
    def calc_acc(self, inputs, result, labels: torch.LongTensor):
        res = result[0]
        if isinstance(res, InceptionOutputs):
            res, _ = res
        assert res.ndim <= 2

        pred = torch.argmax(res, dim=-1)
        # print((pred == labels).float())
        return (pred == labels).float().mean()

    def calc_train_acc(self, inputs, result, labels: torch.LongTensor):
        return self.calc_acc(inputs, result, labels)

    def calc_test_acc(self, inputs, result, labels):
        return self.calc_acc(inputs, result, labels)

    def _update_step(self, loss):
        self.optimizer.zero_grad()
        if self.config.clip_grad_norm is not None:
            clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.clip_grad_norm
            )
        loss.backward()
        self.optimizer.step()

    def prepare_input_label(self, batch):
        imgs, labels = batch
        imgs = imgs.to(self.config.device)
        labels = labels.to(self.config.device)

        return imgs, labels

    def _train_step(self, inputs, labels) -> OrderedDict:

        self.before_train_step()

        result = self.model(inputs)

        loss = self.calc_loss(inputs, result, labels)
        acc = self.calc_train_acc(inputs, result, labels)
        self._update_step(loss)

        return OrderedDict(loss=loss, acc=acc)

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_train_step(self):
        self.model.train()

    def before_test_step(self):
        self.model.eval()

    def _train_loop(self, dataloader: DataLoader):

        self.before_train()

        accumulator = DictAccumulator()

        # iter_times = 0
        for i, batch in enumerate(tqdm(dataloader, leave=False)):
            self._iteration = i
            # iter_times += 1
            inputs, labels = self.prepare_input_label(batch)
            step_res = self._train_step(inputs, labels)
            accumulator.add(step_res)

        self.after_train()

        return accumulator.avg()

    @torch.no_grad()
    def _test_step(self, inputs, labels):
        # self.model.eval()
        self.before_test_step()

        result = self.model(inputs)

        acc = self.calc_test_acc(inputs, result, labels)

        return OrderedDict(acc=acc)

    @torch.no_grad()
    def _test_loop(self, dataloader: DataLoader):

        accumulator = DictAccumulator()

        for i, batch in enumerate(tqdm(dataloader, leave=False)):
            self._iteration = i
            inputs, labels = self.prepare_input_label(batch)
            step_res = self._test_step(inputs, labels)
            accumulator.add(step_res)

        return accumulator.avg()

    def train(
        self, epoch_num: int, trainloader: DataLoader, testloader: DataLoader = None
    ):

        epochs = range(epoch_num)

        bestacc = 0
        bestckpt = None

        for epoch in epochs:

            self._epoch = epoch
            print_split_line()
            train_res = self._train_loop(trainloader)
            print_as_yaml({'epoch': epoch})
            print_as_yaml({'train': train_res})

            if testloader is not None:
                test_res = self._test_loop(testloader)
                if 'acc' in test_res and test_res['acc'] > bestacc:
                    bestckpt = deepcopy(self.model).cpu().eval()
                    bestacc = test_res['acc']
                print_as_yaml({'test': test_res})

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if (epoch + 1) % self.config.save_per_epochs == 0:
                self.save_state_dict()

        if bestckpt is None:
            self.save_state_dict()
        else:
            self.save_state_dict(bestckpt, test_acc=bestacc)

        print(f'best acc: {bestacc}')

    def save_state_dict(self, model=None, **kwargs):
        if model is None:
            model = self.model
        if isinstance(model, (DataParallel, DistributedDataParallel)):
            model = model.module

        model.save_pretrained(self.save_path, **kwargs)


@dataclass
class SimpleTrainConfig(BaseTrainConfig):

    loss_fn: str | Callable = 'cross_entropy'


class SimpleTrainer(BaseTrainer):

    def __init__(self, config: SimpleTrainConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

        self.loss_fn = ClassificationLoss(config.loss_fn)

    def calc_loss(self, inputs, result, labels: LongTensor):
        result = result[0]
        if isinstance(result, InceptionOutputs):
            output, aux = result
            return self.loss_fn(output, labels) + self.loss_fn(aux, labels)
        return self.loss_fn(result, labels)
