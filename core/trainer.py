import numpy
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from core.network import Network, NetworkParams, NetworkDimensions

import torch.nn.functional as F

# Vector = list[Dataset]
from datasets.PoissonImageDataset import PoissonImageDataset
from utils import math_utils
from utils.math_utils import PSNR


class Trainer:
    def __init__(self, params: NetworkParams, device='cpu'):
        self.device = device
        self.net = None
        self.params = params
        self.results = []

    def init_network(self, dimensions: NetworkDimensions):
        self.net = Network(self.params, dimensions).to(self.device)

    def load_dataset(self, dataset):
        dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
        model_input, ground_truth = next(iter(dataloader))
        model_input, ground_truth = model_input.to(self.device), ground_truth.to(self.device)
        return model_input, ground_truth

    def train(self, dimensions: NetworkDimensions, datasets: [Dataset], data_epochs: [int], summary_fn, lr=1e-4,
              loss_fn=F.mse_loss, use_scheduler=True, factor=0.6, patience=30, output_manipulation=None,
              regularization=0, with_log=True):
        self.init_network(dimensions)
        self.net.train(True)
        mses = []
        for dataset, epochs in zip(datasets, data_epochs):
            model_input, ground_truth = self.load_dataset(dataset)
            optimizer = Adam(lr=lr, params=self.net.parameters(), weight_decay=regularization)
            scheduler = None if not use_scheduler else \
                ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)
            for epoch in range(epochs):
                model_output, coords = self.net(model_input)
                if output_manipulation is not None:
                    manipulation = output_manipulation(model_output, coords, dataset.height, dataset.width, dataset.channels)
                    if isinstance(dataset, PoissonImageDataset):
                        loss = loss_fn(ground_truth, manipulation)
                    else:
                        loss = loss_fn(ground_truth, model_output) + \
                               loss_fn(output_manipulation(ground_truth, coords, dataset.height,
                                                           dataset.width, dataset.channels), manipulation)
                else:
                    loss = loss_fn(ground_truth, model_output)
                mses.append(math_utils.PSNR(loss.item()/4))
                if with_log:
                    if not epoch % 10:
                        print("Epoch %d, Total loss %0.6f" % (epoch, loss))
                    if not epoch % 500:
                        summary_fn(epoch, model_output, coords, dataset, self.params.description)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step(loss)
        return numpy.array(mses)

    def test(self, dataset, validate_fn):
        self.net.train(False)
        model_input, ground_truth = self.load_dataset(dataset)
        model_output, coords = self.net(model_input)
        if hasattr(dataset,'normalized') and dataset.normalized:
            model_output = model_output / 2 + 0.5
            dataset.pixels = dataset.pixels / 2 + 0.5
        res = validate_fn(model_output, coords, dataset, self.params.description)
        print(f"{dataset.name} obtained a MSE of {res}")
        self.results.append(list(map(lambda x: x.detach(), res)))

    def statistics(self, compute_PSNR=False):
        res = numpy.array(self.results)
        if compute_PSNR:
            res = list(map(lambda x: list(map(lambda y: PSNR(y), x)), res))
        return numpy.mean(res, axis=0), numpy.std(res, axis=0)
