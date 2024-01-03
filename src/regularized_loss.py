import torch


def bias_regularized_margin_ranking_loss(input1, input2, regularizer, bias):
    """
    implementation of the a regularized version of the multilable margine loss, which is used in the
    repbert paper.

    1-[\SUM REL(Q,D_J^+)-(REL(Q,D_J^-)+LAMBDA BIAS(D_J^-))]
    BIAS(D_J^-) = ABS(boolean bias)
    Args:
        input1: rel(q, d+)
        input2: rel(q, d-)
        regularizer: scalar
        bias: bias of batch
    Returns:

    """
    diff = input2 + regularizer * bias - input1 # input2 - input1 to cover the minus sign of target in the original formula
    input_loss = diff + torch.ones(input1.size()).to('cuda')
    max_0 = torch.nn.functional.relu(input_loss)
    return torch.mean(max_0)

def bias_regularized_margin_ranking_loss_2(score_pos, score_neg, regularizer_neg=None, 
                                             regularizer_pos=None, bias_neg=None, bias_pos=None):
    """
    implementation of the a regularized version of the multilable margine loss, which is used in the
    repbert paper.

    1-[\SUM REL(Q,D_J^+)-(REL(Q,D_J^-)+LAMBDA BIAS(D_J^-))]
    BIAS(D_J^-) = ABS(boolean bias)
    Args:
        input1: rel(q, d+)
        input2: rel(q, d-)
        regularizer: scalar
        bias: bias of batch
    Returns:

    """
    if bias_pos is not None and bias_neg is not None:
        # input2 - input1 to cover the minus sign of target in the original formula
        diff = score_neg + regularizer_neg * bias_neg - score_pos + regularizer_pos * bias_pos 
    elif bias_pos is None and bias_neg is not None:
        diff = score_neg + regularizer_neg * bias_neg - score_pos
    elif bias_neg is None and bias_pos is not None:
        diff = score_neg - score_pos+ regularizer_pos * bias_pos
    else:
        ValueError("something is wrong with the fairnesses")

    input_loss = diff + torch.ones(score_pos.size()).to('cuda')
    max_0 = torch.nn.functional.relu(input_loss)
    return torch.mean(max_0)

def fairness_regularized_margin_ranking_loss(score_pos, score_neg, regularizer_neg=1, 
                                             regularizer_pos=1, fairness_neg=None, fairness_pos=None):
    """
    implementation of the a regularized version of the multilable margine loss, which is used in the
    repbert paper.

    1-[\SUM REL(Q,D_J^+)-(REL(Q,D_J^-)-LAMBDA FAIRNESS(D_J^-))]
    FAIRNESS(D_J^-) = NFaiR(doc_neg)
    Args:
        input1: rel(q, d+)
        input2: rel(q, d-)
        regularizer: scalar
        fairness: NFaiR of each document
    Returns:

    """
    if fairness_pos is not None and fairness_neg is not None:
        # input2 - input1 to cover the minus sign of target in the original formula
        diff = score_neg - score_pos - regularizer_neg * fairness_neg - regularizer_pos * fairness_pos 
    elif fairness_pos is None and fairness_neg is not None:
        diff = score_neg - regularizer_neg * fairness_neg - score_pos
    elif fairness_neg is None and fairness_pos is not None:
        diff = score_neg - score_pos- regularizer_pos * fairness_pos
    else:
        ValueError("something is wrong with the fairnesses")

    input_loss = diff + torch.ones(score_pos.size()).to('cuda')
    max_0 = torch.nn.functional.relu(input_loss)
    return torch.mean(max_0)

# def refinement_regularized_margin_ranking_loss(score_positive, score_negative, score_queries):
#     """
#     implementation of the a regularized version of the multilable margine loss, which is used in the
#     repbert paper.
#
#     max(0, 1-[REL(Q,D_J^+)-(REL(Q,D_J^-)+ REL(Q, Q'))])
#
#     Args:
#         input1:
#
#     Returns:
#
#     """
#     diff = score_negative - score_positive - score_queries# input2 - input1 to cover the minus sign of target in the original formula
#     input_loss = diff + torch.ones(score_positive.size()).to('cuda')
#     max_0 = torch.nn.functional.relu(input_loss)
#     return torch.mean(max_0)
