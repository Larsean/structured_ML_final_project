import torch 
from torch import tensor
from torchmetrics import ConfusionMatrix


def accuracy(output, target, topk=(1,)):
    """
        output: model output, shape: (B, num_classes)
        target: ground truth label, shape: (B,)
    Computes the accuracy over the k top predictions for the specified values of k"""
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

def f1_score(output, target):    
    pred = torch.argmax(output, dim=1)
    cm = ConfusionMatrix(task="multiclass", num_classes=output.size(1))
    result = cm(pred, target)
    
    f1_scores = []
    for i in range(output.size(1)):
        tp = result[i, i]
        fp = torch.sum(result[i, :]) - tp
        fn = torch.sum(result[:, i]) - tp
        precision = tp / (tp + fp) if tp + fp > 0 else tensor(0)
        recall = tp / (tp + fn) if tp + fn > 0 else tensor(0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else tensor(0)
        f1_scores.append(f1)
        # print(f"Class {i}: precision: {precision}, recall: {recall}, f1: {f1}")
    return f1_scores
    
    

def eval_metric(output, target, topk=(1,)):
    """
        output: model output, shape: (B, num_classes)
        target: ground truth label, shape: (B,)
    Computes the top_k accuracy and f1_score for each class."""
    output = output.cpu()
    target = target.cpu()
    acc = accuracy(output, target, topk)
    f1 = f1_score(output, target)
    return acc, f1


if __name__ == '__main__':
    output = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    target = torch.tensor([3, 2])
    acc, f1 = eval_metric(output, target, topk=(1, 3))
    print(acc, f1)