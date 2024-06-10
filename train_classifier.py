import os
import torch
import torch.utils
import torch.utils.data
from torchvision import transforms as T

import hyp
from cvpiclr.models.classifier import IR152_64, SimpleTrainConfig, SimpleTrainer
from cvpiclr.dataset.dataset import LabelImageFolder

if __name__ == '__main__':
    model = IR152_64(num_classes=1000, backbone_path=hyp.ir152_backbone_path).to(
        hyp.device
    )

    dataset = LabelImageFolder(
        hyp.pseudo_dataset,
        transform=T.Compose(
            [
                T.ToTensor(),
                T.Resize((64, 64), antialias=True),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )

    dataloader = torch.utils.data.DataLoader(dataset, 128, shuffle=True, num_workers=8)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    config = SimpleTrainConfig(
        './results_classifier', 'ir152.pth', hyp.device, model, optimizer
    )

    trainer = SimpleTrainer(config)

    trainer.train(100, dataloader, None)
