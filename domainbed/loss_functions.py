import torch.nn.functional as F

LOSSES = [
    'cross_entropy',
    'cross_entropy_approx',
    'brier_score_loss',
    'brier_score_loss_with_logits'
]

# Default order of taylor expansion of cross entropy, is set dynamically by hparam ce_approx_order
CE_APPROX_ORDER = 2

def get_loss_fn(loss_name):
    """Return the loss function with the given name."""
    if loss_name not in LOSSES:
        raise NotImplementedError("Loss function not found: {}".format(loss_name))
    elif loss_name in globals():
        return globals()[loss_name]
    else:
        return getattr(F, loss_name)

def log_approx(input, order=2):
    """
    Taylor expansion of log function
    input - (N,C) where C = number of classes
    order - number of expansions
    """
    result = 0
    for n in range(1, order+1):
        result += (-1)**(n-1) * (input-1)**n / n
    return result

def cross_entropy_approx(input, target, order=CE_APPROX_ORDER, reduction='mean'):
    """
    input - (N,C) where C = number of classes
    target - (N) 
    order - number of expansions
    """
    # TODO: Use log-sum-exp trick
    return F.nll_loss(log_approx(F.softmax(input, dim=1), order=order), target, reduction=reduction)

def brier_score_loss_with_logits(input, target, reduction='mean'):
    return brier_score_loss(F.softmax(input, dim=1), target, reduction=reduction)

def brier_score_loss(input, target, reduction='mean'):
    """
    Computes the brier score: https://en.wikipedia.org/wiki/Brier_score
    input - (N,C) where C = number of classes
    target - (N) 
    reduction - reduction to apply to output, default: mean
    """
    num_classes = input[0].size()[0]
    target = F.one_hot(target, num_classes=num_classes)
    loss = (input - target).pow(2).sum(dim=1)
    if reduction == 'mean':
        loss = loss.mean()
    return loss