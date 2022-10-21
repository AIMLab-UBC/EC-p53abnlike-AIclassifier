from torchvision import transforms

class ToTensor(object):
    """
    Normalize the images.
    """

    def __init__(self):
        self.transform = transforms.ToTensor()

    def __call__(self, x_set):

        for idx, x in enumerate(x_set):
            x_set[idx] = self.transform(x)

        return x_set
