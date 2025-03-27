import torch
import torch.nn as nn

class AdaBinFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, alpha):
        ctx.save_for_backward(weight, alpha)
        binary_weight = torch.sign(weight) * alpha
        return binary_weight

    @staticmethod
    def backward(ctx, grad_output):
        weight, alpha = ctx.saved_tensors
        grad_weight = grad_output.clone()
        grad_alpha = (grad_output * torch.sign(weight)).sum().unsqueeze(0)
        return grad_weight, grad_alpha

class BinaryActivation(nn.Module):
    def forward(self, x):
        return torch.sign(x)