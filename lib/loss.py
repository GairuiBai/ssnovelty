"""
Losses
"""
# pylint: disable=C0301,C0103,R0902,R0915,W0221,W0622


##
# LIBRARIES
import torch

##
def l1_loss(input, target):
    """ L1 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L1 distance between input and output
    """

    return torch.mean(torch.abs(input - target))

##
def l2_loss(input, target, size_average=True):
    """ L2 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L2 distance between input and output
    """
    if size_average:
        return torch.mean(torch.pow((input-target), 2))
    else:
        return torch.pow((input-target), 2)

def new_loss(x0,y0,x1,y1):

    # dis = torch.sqrt


    return (torch.pow((x0-x1),2) + torch.pow((y0-y1),2))

def rotate(x,y):

    if (x*y)>0:
        x_0 = -x
        y_0 = y
    elif (x*y)<0:
        x_0 = x
        y_0 = -y

    return x_0,y_0


def attention_loss(input):

    idx = int(len(input)/4)
    loss_total = None
    loss_this = None
    for i in range(idx):
        x0, y0 = input[i*4][0],input[i*4][1]
        x1, y1 = input[i * 4 + 1][0], input[i * 4 + 1][1]
        x2, y2 = input[i * 4 + 2][0], input[i * 4 + 2][1]
        x3, y3 = input[i * 4 + 2][0], input[i * 4 + 2][1]
        x_0, y_0 = rotate(x0,y0)
        x_1, y_1 = rotate(x1, y1)
        x_2, y_2 = rotate(x2, y2)
        x_3, y_3 = rotate(x3, y3)


        loss_this = (new_loss(x0,y0,x_3, y_3) + new_loss(x1,y1,x_0, y_0) + new_loss(x2,y2,x_1, y_1) + new_loss(x3,y3,x_2, y_2))/4

        loss_total = loss_this if (loss_total is None) else (loss_total + loss_this)


    return loss_total





