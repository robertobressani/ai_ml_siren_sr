import scipy.io.wavfile
import matplotlib
import matplotlib.pyplot as plt
import torch
import os
import torch.nn.functional as F

from utils.data_utils import to_chw, compute_image_laplacian, to_hwc, compute_image_grad, normalize, get_mse, \
    get_gradient_num, get_laplacian_num, shift
from utils.math_utils import laplace, gradient

# This file contains all functions used for summarize and validate results during training and test.


def extract_image_for_plot(image, height, width, channels):
    if channels == 1:
        return torch.clamp(image, min=0.0, max=1.0).cpu() \
            .view(height, width).detach().numpy()
    return torch.clamp(image, min=0.0, max=1.0).cpu() \
        .view(height, width, channels).detach().numpy()


def plot_image_numerical(image_nc, height, width, step, folder="coffecustom_SineLayer"):
    '''
    :param folder:
    :param step:
    :param image_nc: image in the form n x c where n = HxW
    :param height: heigth of the image
    :param width: width of the image
    '''
    titles = ["Image", "Gradient", "Laplacian"]
    channels = image_nc.shape[-1]
    image_hwc = image_nc.reshape([height, width, channels])
    image_chw = to_chw(image_hwc)
    lapl_hwc = to_hwc(compute_image_laplacian(image_chw))
    grad_hwc = to_hwc(compute_image_grad(image_chw), grad=True)  # hw2c

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.suptitle("Numerical", fontsize=15)

    if channels == 1:
        axes[0].imshow(image_hwc.cpu().view(height, width).detach().numpy())
        axes[1].imshow(grad_hwc.norm(dim=-2).cpu().view(height, width).detach().numpy())
        axes[2].imshow(lapl_hwc.cpu().view(height, width).detach().numpy())
    else:
        axes[0].imshow(normalize(image_hwc).cpu().view(height, width, channels).detach().numpy())
        axes[1].imshow(normalize(grad_hwc.norm(dim=-2)).cpu().view(height, width, channels).detach().numpy())
        axes[2].imshow(normalize(lapl_hwc).cpu().view(height, width, channels).detach().numpy())

    for i in range(3):
        axes[i].set_title(titles[i])

    os.makedirs(f"./plots/{folder}/", exist_ok=True)
    plt.savefig(f"./plots/{folder}/Numerical_{step}.png")
    plt.show()


def plot_image_analytical(image_nc, coords, height, width, step, folder="coffecustom_SineLayer"):
    '''
    :param step:
    :param folder:
    :param coords:
    :param image_nc: image in the form n x c where n = HxW
    :param height: heigth of the image
    :param width: width of the image
    '''
    titles = ["Image", "Gradient", "Laplacian"]
    channels = image_nc.shape[-1]
    image_hwc = image_nc.reshape([height, width, channels])
    lapl_hwc = laplace(image_nc, coords).reshape([height, width, channels])
    grad_hwc = gradient(image_nc, coords).reshape([height, width, 2, channels])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.suptitle("Analytical", fontsize=15)

    if channels == 1:
        axes[0].imshow(image_hwc.cpu().view(height, width).detach().numpy())
        axes[1].imshow(grad_hwc.norm(dim=-2).cpu().view(height, width).detach().numpy())
        axes[2].imshow(lapl_hwc.cpu().view(height, width).detach().numpy())
    else:
        axes[0].imshow(normalize(image_hwc).cpu().view(height, width, channels).detach().numpy())
        axes[1].imshow(normalize(grad_hwc.norm(dim=-2)).cpu().view(height, width, channels).detach().numpy())
        axes[2].imshow(normalize(lapl_hwc).cpu().view(height, width, channels).detach().numpy())

    for i in range(3):
        axes[i].set_title(titles[i])

    os.makedirs(f"./plots/{folder}/", exist_ok=True)
    plt.savefig(f"./plots/{folder}/Analytical_{step}.png")
    plt.show()


def plot_audio(step, coords, ground_truth, model_output, fs, folder='default', title='Training'):
    fig, axes = plt.subplots(2, 2)
    axes[0][0].plot(coords.detach().squeeze(0).cpu().numpy(), model_output.squeeze(0).detach().cpu().numpy())
    axes[0][1].plot(coords.detach().squeeze(0).cpu().numpy(), ground_truth.squeeze(0).detach().cpu().numpy())
    axes[1][0].specgram(model_output[:, :, 0].detach().squeeze().cpu().numpy(), Fs=fs)
    axes[1][1].specgram(ground_truth[:, 0].detach().squeeze().cpu().numpy(), Fs=fs)
    os.makedirs(f"./plots/{folder}/", exist_ok=True)
    plt.suptitle(title, fontsize=20)
    plt.savefig(f"./plots/{folder}/step_{step}.png")
    plt.show()


def save_image(image, name, height, width, channels):
    if channels == 1:
        matplotlib.image.imsave(name, extract_image_for_plot(image, height, width, channels), cmap=matplotlib.cm.gray)
    else:
        matplotlib.image.imsave(name, extract_image_for_plot(image, height, width, channels))


def save_audio(track, rate, filename):
    scipy.io.wavfile.write(filename, rate, track.detach().cpu().numpy())


def audio_summary(epoch, model_output, coords, dataset, layer_folder):
    plot_audio(epoch, coords, dataset.amplitude, model_output,
               dataset.rate, folder=f'audio/{layer_folder}/{dataset.name}')


def audio_validate(model_output, coords, dataset, layer_folder):
    folder = f'audio/{layer_folder}/{dataset.name}'
    plot_audio("final", coords, dataset.amplitude, model_output,
               dataset.rate, folder=folder, title="Test")
    save_audio(dataset.amplitude, dataset.rate, f"plots/{folder}/{dataset.name}_gt.wav")
    save_audio(model_output.squeeze(0), dataset.rate, f"plots/{folder}/{dataset.name}_output.wav")
    return [F.mse_loss(dataset.amplitude, model_output.squeeze(0).cpu())]


def image_fitting_summary(epoch, model_output, coords, dataset, layer_folder, numerical=True):
    folder = f'image_fitting/{layer_folder}/{dataset.name}'
    if numerical:
        plot_image_numerical(model_output, dataset.height, dataset.width, epoch, folder=folder)
    else:
        plot_image_analytical(model_output, coords, dataset.height, dataset.width, epoch,
                              folder=folder)


def image_fitting_validate(model_output, coords, dataset, layer_folder):
    name = f'./plots/image_fitting/{layer_folder}/{dataset.name}/results'
    titles = ["Image", "Gradient", "Laplacian"]
    mse = get_mse(dataset.pixels, model_output.cpu())
    plots = [dataset.pixels, model_output]
    functions = [get_gradient_num, get_laplacian_num]
    norm_lambda = lambda img, index: img if index else img.norm(dim=-2)
    for i in range(2):
        plots.append(norm_lambda(functions[i](dataset.pixels, dataset.height, dataset.width, dataset.channels), i))
        plots.append(norm_lambda(functions[i](model_output, dataset.height, dataset.width, dataset.channels), i))
    for i in range(len(plots)):
        if dataset.channels == 1:
            plots[i] = plots[i].cpu().view(dataset.height, dataset.width).detach().numpy()
        else:
            plots[i] = normalize(plots[i].cpu().view(dataset.height, dataset.width, dataset.channels)).detach().numpy()
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    plt.suptitle("Final results for Image Fitting Task", fontsize=20)
    for i in range(3):
        for j in range(2):
            axes[j][i].imshow(plots[i * 2 + j])
            if j == 0: axes[j][i].set_title(titles[i])
    axes[0][0].set_ylabel("Ground truth")
    axes[1][0].set_ylabel("Output")
    os.makedirs(f"{name}", exist_ok=True)
    plt.savefig(f"{name}.png")
    plt.show()
    return [mse]


def poisson_image_summary(epoch, model_output, coords, dataset, layer_folder, numerical=True):
    folder = f'poisson_image/{layer_folder}/{dataset.name}'
    if numerical:
        plot_image_numerical(model_output, dataset.height, dataset.width, epoch, folder=folder)
    else:
        plot_image_analytical(model_output, coords, dataset.height, dataset.width, epoch,
                              folder=folder)


def poisson_image_validate(model_output, coords, dataset, layer_folder, numerical=True, shift_grad=False,
                           shift_image=True,
                           lapl_factor=1e-4, grad_factor=.1):
    name = f'./plots/poisson_image/{layer_folder}/{dataset.name}/results'
    titles = ["Image", "Gradient", "Laplacian"]
    model_output = model_output.cpu()
    if shift_image:
        model_output = shift(model_output, dataset.pixels)

    mses = [get_mse(dataset.pixels, model_output).detach()]

    gt_grad = get_gradient_num(dataset.pixels, dataset.height, dataset.width, dataset.channels)
    if numerical:
        out_grad = get_gradient_num(model_output.cpu(), dataset.height, dataset.width, dataset.channels).detach()
        if shift_grad:
            out_grad = shift(out_grad, gt_grad, grad=True)
        mses.append(get_mse(gt_grad, out_grad))
        mses.append(get_mse(
            get_laplacian_num(dataset.pixels, dataset.height, dataset.width, dataset.channels),
            get_laplacian_num(model_output.cpu(), dataset.height, dataset.width, dataset.channels).detach()))
    else:
        out_grad = grad_factor * gradient(model_output, coords, no_graph=True).detach().cpu()
        if (shift_grad):
            out_grad = shift(out_grad, gt_grad, grad=True)
        mses.append(get_mse(gt_grad, out_grad))
        mses.append(get_mse(
            get_laplacian_num(dataset.pixels, dataset.height, dataset.width, dataset.channels),
            lapl_factor * laplace(model_output, coords, no_graph=True).detach().cpu()))
    plots = [dataset.pixels, model_output]
    functions = [get_gradient_num, get_laplacian_num]
    norm_lambda = lambda img, index: img if index else img.norm(dim=-2)
    for i in range(2):
        plots.append(norm_lambda(functions[i](dataset.pixels, dataset.height, dataset.width, dataset.channels), i))
        plots.append(norm_lambda(functions[i](model_output, dataset.height, dataset.width, dataset.channels), i))
    for i in range(len(plots)):
        if dataset.channels == 1:
            plots[i] = plots[i].cpu().view(dataset.height, dataset.width).detach().numpy()
        elif i == 0 or i==1:
            plots[i] = torch.clamp(plots[i].cpu(), min=0.0, max=1.0).view(dataset.height, dataset.width,
                                                                          dataset.channels).detach().numpy()
        else:
            plots[i] = normalize(plots[i].cpu().view(dataset.height, dataset.width, dataset.channels)).detach().numpy()
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    plt.suptitle("Final results for Poisson Image Task", fontsize=20)
    for i in range(3):
        for j in range(2):
            axes[j][i].imshow(plots[i * 2 + j])
            if j == 0: axes[j][i].set_title(titles[i])
    axes[0][0].set_ylabel("Ground truth")
    axes[1][0].set_ylabel("Output")
    plt.savefig(f"{name}.png")
    plt.show()
    return mses


def image_super_resolution_summary(epoch, model_output, coords, dataset, layer_folder):
    plot_image_numerical(model_output, dataset.height, dataset.width, epoch,
                         folder=f'image_super_resolution/{layer_folder}/{dataset.name}')


def image_super_resolution_validate(model_output, coords, dataset, layer_folder):
    folder = f"plots/image_super_resolution/{layer_folder}/results"
    os.makedirs(folder, exist_ok=True)
    save_image(model_output, f"{folder}/{dataset.name}.png", dataset.height, dataset.width,
               dataset.channels)
    save_image(dataset.pixels, f"{folder}/{dataset.name}_gt.png", dataset.height, dataset.width,
               dataset.channels)
    res = get_mse(dataset.pixels, model_output.cpu())

    plt.figure(1, dpi=200)
    plt.suptitle(f"Image super resolution {dataset.name}", fontsize=15)

    ax = plt.subplot(121)
    ax.set_title("Reconstructed")
    if dataset.channels == 1:
        plt.imshow(extract_image_for_plot(model_output, dataset.height, dataset.width,
                                          dataset.channels), cmap=matplotlib.cm.gray)
    else:
        plt.imshow(extract_image_for_plot(model_output, dataset.height, dataset.width,
                                          dataset.channels))

    ax = plt.subplot(122)
    ax.set_title("Ground truth")
    if dataset.channels == 1:
        plt.imshow(extract_image_for_plot(dataset.pixels, dataset.height, dataset.width,
                                          dataset.channels), cmap=matplotlib.cm.gray)
    else:
        plt.imshow(extract_image_for_plot(dataset.pixels, dataset.height, dataset.width,
                                          dataset.channels))
    plt.show()
    print(f"{dataset.name} obtained a MSE of {res}")
    return [res]
