
# Python image library
from PIL import Image

# torch modules
import torch
from torch.autograd import Variable

# torchvision modules
import torchvision
from torchvision import datasets, transforms

# numpy modules
import numpy as np


class WaterMarker():
    def __init__(self, watermark, bfactor, nlevel, noise=False):
        super(WaterMarker, self).__init__()
        self.watermark = watermark
        self.bfactor   = bfactor
        # noise parameters
        self.mean      = 0.0
        self.nlevel    = nlevel
        self.noise     = noise

    def do_transform(self, data):
        """
            Transform data (expected as a PIL Image - h x w)
        """
        # load the watermark image
        wtmark = Image.open(self.watermark).convert('RGBA')
        # add noise to the watermark
        if self.noise: wtmark = self.add_noise(data, wtmark)
        # use the blend-factor value to the alpha channel
        (wR, wG, wB, wA) = wtmark.split()
        wA = wA.point(lambda a: int(a * self.bfactor))
        wtmark = Image.merge('RGBA', (wR, wG, wB, wA))
        # attach a watermark to the data
        (w, h) = data.size
        wtmarked_image = Image.new('RGB', (w, h), (0,0,0,0))
        wtmarked_image.paste(data, (0,0))
        wtmarked_image.paste(wtmark, (0,0), mask=wtmark)
        return wtmarked_image

    def add_noise(self, data, wtmark):
        # convert to the tensor [0,1]
        tdata  = transforms.ToTensor()(data)
        twmark = transforms.ToTensor()(wtmark)
        # return if intensity is zero
        if not self.nlevel: return wtmark
        # base-dependent noise synthesize
        noisev = tdata.std() * self.nlevel
        # blend the Gaussian noise to the base image
        noise  = twmark.data.new(twmark.size()).normal_(self.mean, noisev)
        nwtmark= torch.clamp((twmark + noise), min=0.0, max=1.0)
        nwtmark= transforms.ToPILImage()(nwtmark)
        return nwtmark


"""
    Main (test purpose)
"""
if __name__ == '__main__':
    # read the data
    data  = Image.open('../etc/samples/cifar10_sample.jpg')

    # blend the noised watermark
    wmarker = WaterMarker( \
        '../etc/watermarks/cifar10_wmark_32x32_three.png', \
        0.5, 0.0, noise=True)
    output  = wmarker.do_transform(data)

    # saves
    output.save('wmark_output.png')

    # Fin.
