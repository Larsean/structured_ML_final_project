import os
import shutil

import torch
import yaml
import wandb

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def wandb_init(args, resume=False, init_caption=''):
    run = wandb.init(
        project="Contrastive-Learning-PyTorch"+'SimCLR',
        notes=f"self-supervised leaning on {args.dataset_name} dataset." + init_caption,
        tags=["Contrastive learning", "SimCLR"],
        config=args,
        name=f"{args.arch}, {args.out_dim}d, batch {args.batch_size}",
        resume=resume,
        )
    return run