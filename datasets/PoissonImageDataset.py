import skimage
from datasets.ImageFittingDataset import ImageFitting
from utils.data_utils import  get_gradient_num, get_laplacian_num


class PoissonImageDataset(ImageFitting):

    def __init__(self, data_image=skimage.data.camera(), down_scale=1, normalized=False, name="camera", fit_laplacian=False, up_scale = 1):
        super().__init__(data_image=data_image, down_scale=down_scale, up_scale = up_scale, normalized=normalized, name=name)

        if not fit_laplacian:
            # computing gradient and converting to Nx2xC
            self.gt = get_gradient_num(self.pixels, self.height, self.width, self.channels)
        else:
            # computing laplacian and converting to NxC
            self.gt = get_laplacian_num(self.pixels, self.height, self.width, self.channels)

        self.gt = self.gt.squeeze(0)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index > 0: raise IndexError
        return self.coords, self.gt
