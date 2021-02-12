import torch
import scipy.io.wavfile as wavfile
import numpy as np
from utils.data_utils import get_mgrid


class AudioSignal(torch.utils.data.Dataset):
    def __init__(self, filename, name="music"):
        self.rate, self.data = wavfile.read(filename)
        self.data = self.data.astype(np.float32)
        self.timepoints = get_mgrid([len(self.data)])
        self.amplitude = torch.Tensor(self.data / np.max(np.abs(self.data)))
        if len(self.amplitude.shape) == 1:
            self.channels = 1
            self.amplitude = self.amplitude.view(-1, 1)
        else:
            self.channels = self.amplitude.shape[-1]
        self.filename = filename
        self.name = name

    def get_num_samples(self):
        return self.timepoints.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.timepoints, self.amplitude