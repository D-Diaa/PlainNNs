import torch
from torch import nn


class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)  # Should save input for potential use in backward
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BiHalfEstimator(torch.autograd.Function):
    gamma = 6

    @staticmethod
    def forward(ctx, U):
        # Yunqiang for half and half (optimal transport)
        _, index = U.sort(0, descending=True)
        N, D = U.shape
        B_creat = torch.cat((torch.ones([int(N / 2), D]), -torch.ones([N - int(N / 2), D]))).to(U.device)
        B = torch.zeros(U.shape).to(U.device).scatter_(0, index, B_creat)
        ctx.save_for_backward(U, B)
        return B

    @staticmethod
    def backward(ctx, g):
        U, B = ctx.saved_tensors
        add_g = (U - B) / (B.numel())
        grad = g + BiHalfEstimator.gamma * add_g
        return grad


class BinaryActivation(nn.Module):
    def __init__(self, activation_type="tanh", bihalf_gamma=6):
        super(BinaryActivation, self).__init__()
        self.activation_type = activation_type
        self.bihalf_gamma = bihalf_gamma
        BiHalfEstimator.gamma = bihalf_gamma

    def forward(self, x):
        if self.activation_type == "tanh":
            return torch.tanh(x)
        elif self.activation_type == "ste":
            return StraightThroughEstimator.apply(x)
        elif self.activation_type == "sigmoid":
            return torch.sigmoid(x) - 0.5
        elif self.activation_type == "bihalf":
            return BiHalfEstimator.apply(x)
        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")