import os
import json
import argparse
import numpy as np
from types import SimpleNamespace

import torch
import torchvision
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from simsiam.models import ResNet, LinearClassifier
from simsiam.transforms import load_transforms, augment_transforms2


def main(cfg: SimpleNamespace) -> None:

    model = ResNet(
        backbone=cfg.model.backbone,
        num_classes=cfg.data.num_classes,
        pretrained=False,
        freeze=cfg.model.freeze
    )

    # if cfg.model.weights_path:
    #     model.encoder.load_state_dict(torch.load("runs/May17_09-30-21_DESKTOP-CVQR3S4/pretrained.pt"))
    model.encoder.load_state_dict(torch.load("repretrain/pretrained/pretrained.pt"))
    model = model.to(cfg.device)

    opt = torch.optim.SGD(
        params=model.parameters(),
        lr=cfg.train.lr,
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay
    )

    loss_func = torch.nn.CrossEntropyLoss()

    dataset = torchvision.datasets.STL10(
        root="D:/kaihsiang/結構化機器學習/byol/data",
        split="train",
        transform=load_transforms(input_shape=cfg.data.input_shape),
        download=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=0
    )

    test_dataset = torchvision.datasets.STL10(
        root="D:/kaihsiang/結構化機器學習/byol/data",
        split="test",
        transform=load_transforms(input_shape=cfg.data.input_shape),
        download=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=0
    )

    transforms = augment_transforms2(
        input_shape=cfg.data.input_shape,
        device=cfg.device
    )

    writer = SummaryWriter('repretrain/train/no freeze/')
    print(f'freeze:{cfg.model.freeze}')
    n_iter = 0
    for epoch in range(cfg.train.epochs):

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch, (x, y) in pbar:

            opt.zero_grad()

            x, y = x.to(cfg.device), y.to(cfg.device)
            #x = transforms(x)
            y_pred = model(x)
            loss = loss_func(y_pred, y)
            loss.backward()
            opt.step()

            pbar.set_description("Epoch {}, Loss: {:.4f}".format(epoch, float(loss)))

            if n_iter % cfg.train.log_interval == 0:
                writer.add_scalar(tag="tune_loss", scalar_value=float(loss), global_step=n_iter)

            n_iter += 1

        total = 0
        if epoch % 10 == 9:
            correct = 0
            for x1, y1 in test_loader:
                x1 = x1.to(cfg.device)
                y1 = y1.to(cfg.device)

                #x1 = transforms(x1)
                logits = model(x1)
                predictions = torch.argmax(logits, dim=1)
                
                total += y1.size(0)
                correct += (predictions == y1).sum().item()
                
            acc = 100 * correct / total
            writer.add_scalar(tag="test_acc", scalar_value=float(loss), global_step=epoch+1)
            print(f"Testing accuracy: {np.mean(acc)}")

        # save checkpoint
        torch.save(model.state_dict(), os.path.join(writer.log_dir, cfg.model.name + ".pt"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to config json file")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = json.loads(f.read(), object_hook=lambda d: SimpleNamespace(**d))

    main(cfg)
