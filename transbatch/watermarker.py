"""
    Watermark (visibly) with an image
"""
# basics
import numpy as np
from PIL import Image

# torch modules
import torch
from torch.autograd import Variable

# torchvision modules
import torchvision
import torchvision.utils as vutils
from torchvision import datasets, transforms


class WaterMarker():
    def __init__(self, bfactor, nlevel, noise=False):
        super(WaterMarker, self).__init__()
        # w-mark blending factors
        self.bfactor   = bfactor
        # noise parameters
        self.mean      = 0.0
        self.nlevel    = nlevel
        self.noise     = noise

    def do_transform(self, data_batch, wmark_batch):
        """
            Watermark the data with the watermark...
            - Data : a batch of data       (B x 3 x H x W)
            - Wmark: a batch of watermarks (B x 4 x H x W)
            - Out  : a batch of wmarked-data + noises...
        """
        # watermark data (consider blending factor)
        # c.f. data = (alpha*c) * wmark + (1-alpha*c) * data
        wmark_alpha  = wmark_batch[:,3:4] * self.bfactor
        wmarked_data = wmark_alpha * wmark_batch[:,0:3] \
                            + (1 - wmark_alpha) * data_batch

        # blend noise to the watermarks
        if self.noise:
            wmarked_data = self.add_noise(wmarked_data)
        return wmarked_data

    def add_noise(self, wmarked_data):
        # return if intensity is zero
        if not self.nlevel: return wmarked_data

        # data-dependent noise synthesize
        for idx, each_data in enumerate(wmarked_data):
            noisev = each_data.std() * self.nlevel
            noise  = each_data.data.new(each_data.size()).normal_(self.mean, noisev)
            ndata  = torch.clamp((each_data + noise), min=0., max=1.)
            wmarked_data[idx] = ndata
        return wmarked_data
