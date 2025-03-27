import torch

class DVLR:
    def __init__(self, init_alpha=1.0, lr=0.01):
        self.alpha = torch.tensor(init_alpha, requires_grad=True)
        self.lr = lr

    def update(self, grad_alpha):
        self.alpha = self.alpha - self.lr * grad_alpha
        self.alpha = self.alpha.clamp(0.1, 2.0)
        return self.alpha