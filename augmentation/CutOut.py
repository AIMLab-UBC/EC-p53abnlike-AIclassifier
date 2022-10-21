import torch
from torchvision import transforms

class CutOut(object):
    """
    Cutting out random places in image.
    """

    def __init__(self, num_cut, length_cut, color="black"):
        assert isinstance(num_cut, int)
        assert isinstance(length_cut, int)

        self.num_cut    = num_cut
        self.length_cut = (length_cut, length_cut)

        if color=="white":
            self.color = 1.0
        elif color=="black":
            self.color = 0.0
        else:
            raise NotImplementedError(f"{color} is not implemented!")


    def __call__(self, tensor_img):
        assert isinstance(tensor_img, torch.Tensor)

        for num in range(self.num_cut):
            i, j, h, w = transforms.RandomCrop.get_params(tensor_img,
                                                          output_size=self.length_cut)
            # ToTensor Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
            # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
            tensor_img[:, i:i+h, j:j+h] = self.color

        return tensor_img
