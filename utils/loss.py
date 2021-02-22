import torch
from torch import nn
from torch.nn import functional as F


def create_loss_fn(args):
    if args.label_smoothing > 0:
        criterion = SmoothCrossEntropy(alpha=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion.to(args.device)


class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()
