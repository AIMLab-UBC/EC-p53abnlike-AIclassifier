import torch
from torchvision import transforms
import random
from PIL import Image

class SizeJitter(object):
    """
    Resizing Image with a random value within original_size +/- ratio
    """

    def __init__(self, ratio, prob=0.5, color="black", dynamic_bool=False):
        assert isinstance(ratio, float)
        assert isinstance(prob, float)

        self.ratio = ratio
        self.prob = prob
        self.dynamic_bool = dynamic_bool

        if color=="white":
            self.color = 255
        elif color=="black":
            self.color = 0
        else:
            raise NotImplementedError(f"{color} is not implemented!")


    def __call__(self, PIL_img):
        assert isinstance(PIL_img, Image.Image)
        if random.random() < self.prob:
            W, H = PIL_img.size
            zoom_in_or_out = random.choice(["zoom_in", "zoom_out"])
            if self.dynamic_bool:
                rand_zoom = random.random() #float between 0.0 and 1.0
            else:
                rand_zoom = 1.0
            ratio = 1 - self.ratio*rand_zoom if zoom_in_or_out == "zoom_out" else 1 + self.ratio*rand_zoom
            resize_size = (int(W*ratio), int(H*ratio))
            resized_img = transforms.Resize(resize_size)(PIL_img)
            # Zoom out
            if zoom_in_or_out == "zoom_out":
                #rand_zoom >= 0.5:
                # ratio < 1
                pad_H = int((H-resize_size[1])/2)
                pad_W = int((W-resize_size[0])/2)
                pad_size = (pad_W, pad_H, W-pad_W-resize_size[0], H-pad_H-resize_size[1])
                out_PIL_img = transforms.Pad(pad_size, fill=self.color)(resized_img)
            # Zoom In
            else:
                crop_size = (H,W)
                out_PIL_img = transforms.RandomCrop(crop_size)(resized_img)

            return out_PIL_img
        else:
            return PIL_img
