# CVP-ICL-R: Center-aligned Visual Pre-training and Identity Content Learning for face Recognition

Final project for Computer Vision Course.

> “人脸识别是这样的，CVP只需要框出人脸区域，可是ICL要考虑的类别就很多了。”

> “若无全Pipeline上下万众一心，分类精度再高又有何用？仰赖CVP框出人脸，ICL必不负所托。”

The code of `hyp.py` and `./cvpiclr` is comming soon.

## 使用说明

根据实际情况填写`hyp.py`中各项参数和路径，并进行如下几个步骤

### 数据集处理

使用如下脚本预处理CelebA数据集
```bash
python preprocess.py
```

### 目标检测器训练

需要训练LowNet, MidNet和HighNet三个目标检测模型，按如下脚本运行：
```bash
python train_low.py
python train_mid.py
python train_high.py
```


### 分类器训练数据集生成

使用目标检测模型提取人脸区域，生成中心对齐的人脸分类训练集
```bash
python generate_dataset.py
```

### 分类器训练

按一般的方法训练分类器
```bash
python train_classifier.py
```

## 评估与可视化

可视化操作：调整`gen_sample.py`中的文件路径，并运行
```bash
python gen_sample.py
```


无目标评估：
```bash
python evaluation_nontarget.py
```

有目标评估：
```bash
python evaluation.py
```








