from math import floor
import skimage.data  # DO NOT REMOVE, USED IN EVAL
import matplotlib
from kornia import laplacian
from kornia.filters import spatial_gradient
from PIL import Image
import scipy.stats as stats
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import matplotlib.pyplot as plt
import torch
from torch.fft import fft
import torch.nn.functional as F
import numpy as np
from utils.math_utils import laplace, gradient


def compute_image_grad(img):
    """
    Function to get the gradient of an image using Sobel filters
    :param img: CxHxW Tensor
    :return: Cx2xHxW Tensor
    """
    img = spatial_gradient(img.unsqueeze(0), normalized=False)  # adding 1 dimension required by kornia library
    return img[0]  # getting rid of first dimension


def compute_image_laplacian(img):
    """
    Function to compute the laplacian of an image using Laplacian filter
    :param img: CxHxW Tensor
    :return: CxHxW Tensor
    """
    img = laplacian(img.unsqueeze(0), 3, normalized=False)  # adding 1 dimension required by kornia library
    return img[0]  # getting rid of first dimension


def get_mgrid(sizes):
    """
    Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1
    of dimension len(sizes) and with size sizes[i] for the i-th dimension
    """
    tensors = tuple([torch.linspace(-1, 1, steps=size) for size in sizes])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, len(sizes))
    # return mgrid
    return torch.flip(mgrid, [-1])


def get_image_tensor(data_image, down_scale, up_scale=1, normalized=False):
    """
    Load a multi-dimensional array that store an image into a tensor
    :param down_scale:
    :param up_scale:
    :param normalized:
    :param data_image: uint8 ndarray
    :return:
    """
    img = Image.fromarray(data_image)
    transform_array = [ToTensor()]

    height = data_image.shape[0]
    width = data_image.shape[1]
    if down_scale > 0:
        height = floor(height / down_scale)
        width = floor(width / down_scale)
        transform_array.append(Resize([height, width], interpolation=Image.NEAREST))

        if up_scale > 0:
            height = floor(height * up_scale)
            width = floor(width * up_scale)
            transform_array.append(Resize([height, width], interpolation=Image.BICUBIC))
        else:
            raise Exception(f"Up scale resolution to {up_scale} not accepted")
    else:
        raise Exception(f"Down scale resolution to {down_scale} not accepted")

    if normalized:
        transform_array.append(Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])))

    transform = Compose(transform_array)
    img = transform(img)

    # Check if grey scale
    if len(img.shape) == 2:
        img = img.unsqueeze(0)

    return img


def to_chw(image: torch.tensor, grad: bool = False):
    """
    Convert image shape
    :param image: HxW(x2)xC image
    :param grad: indicates if additional dimension is present
    :return: C(x2)xHxW image
    """
    if grad:
        image = image.permute(3, 2, 0, 1)
    else:
        image = image.permute(2, 0, 1)

    return image


def to_hwc(image: torch.tensor, grad=False):
    """
    Convert image shape
    :param image: C(x2)xHxW image
    :param grad: indicates if additional dimension is present
    :return: HxW(x2)xC image
    """
    if grad:
        image = image.permute(2, 3, 1, 0)
    else:
        image = image.permute(1, 2, 0)

    return image


def normalize(t: torch.tensor):
    """
    Brings the input in range [0,1]
    :param t:
    :return:
    """
    res = (t - torch.min(t)) / (torch.max(t) - torch.min(t))
    return res


def shift(model_output, gt, grad=False):
    """
    Shift model output to have the same mean of gt
    :param model_output:
    :param gt:
    :param grad:
    :return:
    """
    if not grad:
        mean_diff = torch.mean(model_output, dim=[-2]).detach() - torch.mean(gt, dim=[-2])
    else:
        mean_diff = torch.mean(model_output, dim=[-3, -2]).detach() - torch.mean(gt, dim=[-3, -2])
    return model_output - mean_diff


def get_gradient_num(model_output, h, w, c):
    img_out_chw = to_chw(model_output.reshape([h, w, c]))
    return compute_image_grad(img_out_chw) \
        .permute(2, 3, 1, 0) \
        .reshape([1, h * w, 2, c])


def get_laplacian_num(model_output, h, w, c):
    img_out_chw = to_chw(model_output.reshape([h, w, c]))
    return to_hwc(compute_image_laplacian(img_out_chw)) \
        .reshape([1, h * w, c])


def get_manipulator(id, factor=1):
    """
    Return a lambda function for output manipulation
    :param id:
    :param factor:
    :return:
    """
    def result(model_output, coords, h, w, c):
        if id == "grad_num":
            return factor * get_gradient_num(model_output, h, w, c)
        elif id == "grad":
            return factor * gradient(model_output, coords)
        elif id == "lapl_num":
            return factor * get_laplacian_num(model_output, h, w, c)
        elif id == "lapl":
            return factor * laplace(model_output, coords)
        else:
            raise Exception("error wrong id of manipulator")

    return result


def get_mse(ground_truth: torch.tensor, model_output: torch.tensor):
    """
    For each pixel
    :param ground_truth: dim = 1 x n x c
    :param model_output: dim = 1 x n x c
    :return:
    """
    acc = torch.mean((model_output - ground_truth).pow(2).sum([-2, -1] if len(ground_truth.shape) == 4 else -1))
    return acc


def get_fft_mse(ground_truth: torch.tensor, model_output: torch.tensor, fft_scale: int = 50):
    fft_mse = F.mse_loss(fft(ground_truth).abs(), fft(model_output).abs()) / fft_scale \
              + F.mse_loss(ground_truth, model_output)
    return fft_mse


def get_image(image_name):
    """
    Load image from skimage
    :param image_name:
    :return:
    """
    return eval(f"skimage.data.{image_name}()")


def get_training_images():
    """
    Return the set of names for skimage dataset
    :return:
    """
    return ["camera", "coffee", "chelsea", "astronaut", "grass", "colorwheel"]


def get_div2k_images():
    """
    Return the set of names for our DIV2K dataset subset
    :return:
    """
    images = ["0803", "0804", "0805", "0807", "0810", "0812", "0813", "0817", "0818", "0826", "0827"]
    return images


def get_div2k_image(image_name, resolution="high", dir="./"):
    """
    Load image from DIV2K dataset subset
    :param image_name:
    :param resolution:
    :param dir:
    :return:
    """
    folder = dir + "data/images/resized/"
    if resolution == "high":
        image = f"{folder}{image_name}.png"
    else:
        image = f"{folder}{image_name}x4.png"
    return np.array(Image.open(image))


def eformat(f, prec, exp_digits):
    """
    Utils method for plotting
    :param f:
    :param prec:
    :param exp_digits:
    :return:
    """
    s = "%.*e" % (prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%+0*d" % (mantissa, exp_digits + 1, int(exp))


def get_spectrum(activations):
    """
    Return shifted the frequency spectrum of the input signal
    :param activations:
    :return:
    """
    n = activations.shape[0]

    spectrum = np.fft.fft(activations.numpy().astype(np.double).sum(axis=-1), axis=0)[:n // 2]
    spectrum = np.abs(spectrum)

    max_freq = 100
    freq = np.fft.fftfreq(n, 2. / n)[:n // 2]
    return freq[:max_freq], spectrum[:max_freq]


def plot_all_activations_and_grads(activations):
    """
    Plot all activations and grads of the network
    :param activations:
    :return:
    """
    num_cols = 4
    num_rows = len(activations)

    fig_width = 6
    fig_height = num_rows * fig_width / num_cols

    fontsize = 5

    fig, axs = plt.subplots(num_rows, num_cols, gridspec_kw={'hspace': 0.5, 'wspace': 0.4},
                            figsize=(fig_width, fig_height), dpi=300)

    axs[0][0].set_title("Activation Distribution", fontsize=7, fontfamily='serif', pad=5.)
    axs[0][1].set_title("Activation Spectrum", fontsize=7, fontfamily='serif', pad=5.)
    axs[0][2].set_title("Gradient Distribution", fontsize=7, fontfamily='serif', pad=5.)
    axs[0][3].set_title("Gradient Spectrum", fontsize=7, fontfamily='serif', pad=5.)

    x_formatter = matplotlib.ticker.FuncFormatter(lambda x, pos: eformat(x, 0, 1))
    y_formatter = matplotlib.ticker.FuncFormatter(lambda x, pos: eformat(x, 0, 1))

    for idx, (key, value) in enumerate(activations.items()):
        grad_value = value.grad.cpu().detach().squeeze(0)
        flat_grad = grad_value.view(-1)
        axs[idx][2].hist(flat_grad, bins=256, density=True)

        value = value.cpu().detach().squeeze(0)  # (1, num_points, 256)
        flat_value = value.view(-1)

        axs[idx][0].hist(flat_value, bins=256, density=True)

        if idx > 1:
            if not (idx) % 2:
                x = np.linspace(-1, 1., 500)
                axs[idx][0].plot(x, stats.arcsine.pdf(x, -1, 2),
                                 linestyle=':', markersize=0.4, zorder=2)
            else:
                mu = 0
                variance = 1
                sigma = np.sqrt(variance)
                x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 500)
                axs[idx][0].plot(x, stats.norm.pdf(x, mu, sigma),
                                 linestyle=':', markersize=0.4, zorder=2)

        activ_freq, activ_spec = get_spectrum(value)
        axs[idx][1].plot(activ_freq, activ_spec)

        grad_freq, grad_spec = get_spectrum(grad_value)
        axs[idx][-1].plot(grad_freq, grad_spec)

        for ax in axs[idx]:
            ax.tick_params(axis='both', which='major', direction='in',
                           labelsize=fontsize, pad=1., zorder=10)
            ax.tick_params(axis='x', labelrotation=0, pad=1.5, zorder=10)

            ax.xaxis.set_major_formatter(x_formatter)
            ax.yaxis.set_major_formatter(y_formatter)
