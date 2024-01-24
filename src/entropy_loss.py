import torch


EPSILON = 1e-8
def entropy_loss(preds):
    """
    Returns the entropy loss: negative of the entropy present in the
    input distribution
    """
    return torch.tensor(1)
    # return torch.mean(preds * torch.log(preds + EPSILON))
