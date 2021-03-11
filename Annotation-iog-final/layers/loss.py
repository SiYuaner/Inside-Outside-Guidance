from __future__ import division
import numpy as np
import torch
   
def class_balanced_cross_entropy_loss(output, label, size_average=True, batch_average=True, void_pixels=None):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    size_average: return per-element (pixel) average loss
    batch_average: return per-batch average loss
    void_pixels: pixels to ignore from the loss
    Returns:
    Tensor that evaluates the loss
    """
    assert(output.size() == label.size())
    labels = torch.ge(label, 0.5).float()
    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))
    if void_pixels is not None:
        w_void = torch.le(void_pixels, 0.5).float()
        final_loss = torch.mul(w_void, loss_val)
    else:
        final_loss=loss_val        
    final_loss = torch.sum(-final_loss)
    
    # average loss
    if size_average:
        final_loss /= np.prod(label.size())
    elif batch_average:
        final_loss /= label.size()[0]
    return final_loss
   
    
