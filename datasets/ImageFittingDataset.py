import skimage
from torch.utils.data import Dataset
from utils.data_utils import get_image_tensor, get_mgrid, to_hwc


class ImageFitting(Dataset):
    '''
    Fitting a data image into a torch tensor. Implement Dataset abstract class.
    '''

    def __init__(self, data_image=skimage.data.camera(), name="camera", down_scale=1, up_scale=1, normalized=False):
        super().__init__()

        img = get_image_tensor(data_image, down_scale=down_scale, up_scale=up_scale, normalized=normalized)

        self.channels, self.height, self.width = img.shape
        self.pixels = to_hwc(img).reshape([self.width * self.height, self.channels])
        self.coords = get_mgrid(img.shape[1:])
        self.normalized = normalized
        self.name = f"{name}_d{down_scale}_u{up_scale}"

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels
