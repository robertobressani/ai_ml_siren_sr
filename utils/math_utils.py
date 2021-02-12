from math import log10, sqrt
import torch


def laplace(y, x, no_graph = False):
    channels = y.shape[-1]
    grad = gradient(y, x)
    laplacian = divergence(grad, x, channels=channels, no_graph= no_graph)
    return laplacian/62.5


def divergence(y, x, channels=1,no_graph = False):
    if channels == 1:
        div = 0.
        y = y.squeeze(-1)
        for i in range(2):
            div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    else:
        div = []
        for channel in range(channels):
            div_tmp = 0.
            y_tmp = y[..., channel]
            for i in range(2):
                div_tmp += \
                torch.autograd.grad(y[..., i, channel], x, torch.ones_like(y[..., i, channel]), create_graph=True)[0][
                ..., i:i + 1]
                if no_graph:
                    div_tmp= div_tmp.detach()
            div.append(div_tmp)
        div = torch.stack(div, dim=-1).squeeze(-2)
    return div


def gradient(y, x, grad_outputs=None, no_graph= False):
    channels = y.shape[-1]
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    if channels == 1:
        grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        grad = grad.unsqueeze(-1)
    else:
        grad = []
        for channel in range(channels):
            tmp = torch.autograd.grad(y[..., channel], [x], grad_outputs=grad_outputs[..., channel], create_graph=True)[
                0]
            if no_graph:
                tmp = tmp.detach()
            grad.append(tmp)
        grad = torch.stack(grad, dim=-1)
    return grad/40


def PSNR(mse, max_t=1):
    return 20 * log10(max_t / sqrt(mse))
