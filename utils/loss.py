import torch

def l2_loss(pred, target):
    num_examples = pred.shape[0]
    diff_norms = torch.norm(pred.reshape(num_examples,-1) - target.reshape(num_examples,-1), 2, 1)
    y_norms = torch.norm(target.reshape(num_examples,-1), 2, 1)
    return torch.mean(diff_norms/y_norms)


@torch.jit.script
def l2_loss_opt(pred: torch.Tensor, target: torch.Tensor):
    num_examples = pred.shape[0]
    diff_norms = torch.norm(pred.reshape(num_examples,-1) - target.reshape(num_examples,-1), 2, 1)
    y_norms = torch.norm(target.reshape(num_examples,-1), 2, 1)
    return torch.mean(diff_norms/y_norms)

