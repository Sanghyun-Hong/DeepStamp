
# basic imports
import cv2
import itertools
import numpy as np
from PIL import Image
from random import randint
from scipy.ndimage.filters import gaussian_filter

# torch modules
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# torchvision modules
import torchvision
from torchvision import datasets, transforms


"""
    Translation: move the watermark image w.r.t a random position.
                (maximum disposition is 'max_dispose' in x, y axis)
"""
class WMarkerTranslate():
    def __init__(self, watermark, bfactor, max_dispose, nlevel, noise=False):
        super(WMarkerTranslate, self).__init__()
        self.watermark   = watermark
        self.bfactor     = bfactor
        self.max_dispose = max_dispose
        # noise parameters
        self.mean        = 0.0
        self.nlevel      = nlevel
        self.noise       = noise

    def do_transform(self, data):
        """
            Transform data (expected as a PIL Image - h x w)
        """
        # load the watermark image
        wtmark = Image.open(self.watermark).convert('RGBA')
        # choose random x, y disposition
        x_move = randint(-self.max_dispose, self.max_dispose)
        y_move = randint(-self.max_dispose, self.max_dispose)
        # create disposed watermark
        (w, h) = wtmark.size
        disposed_wmark = Image.new('RGBA', (w, h), (0,0,0,0))
        disposed_wmark.paste(wtmark, (x_move, y_move))
        # add noise to the watermark
        if self.noise: disposed_wmark = self.add_noise(data, disposed_wmark)
        # use the blend-factor value to the alpha channel
        (wR, wG, wB, wA) = disposed_wmark.split()
        wA = wA.point(lambda a: int(a * self.bfactor))
        disposed_wmark = Image.merge('RGBA', (wR, wG, wB, wA))
        # create a blended image (data + watermark)
        (w, h) = data.size
        wtmarked_image = Image.new('RGB', (w, h), (0,0,0))
        wtmarked_image.paste(data, (0,0))
        wtmarked_image.paste(disposed_wmark, (0,0), mask=disposed_wmark)
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
    Opacity variation: apply the random opacity to alpha channel
"""
class WMarkerOpacityVariation():
    def __init__(self, watermark, bfactor, max_shift, nlevel, noise=False):
        super(WMarkerOpacityVariation, self).__init__()
        self.watermark = watermark
        self.bfactor   = bfactor
        self.max_shift = max_shift
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
        cupper = self.bfactor - self.max_shift/255.0
        clower = self.bfactor + self.max_shift/255.0
        # apply a new blending factor for the watermark image
        (wR, wG, wB, wA) = wtmark.split()
        blendf = np.random.uniform(clower, cupper, wtmark.size) # [c-x/255, c+x/255] random
        walpha = np.array(wA) * self.bfactor                    # c * alpha
        walpha = walpha.astype(np.uint8)                        # float32 to uint8
        walpha = np.clip(walpha, 0, 255)                        # clip to [0, 255] alpha
        walpha = Image.fromarray(walpha, mode='L')              # convert it to channel
        wtmark = Image.merge('RGBA', (wR, wG, wB, walpha))
        # add noise to the watermark
        if self.noise: wtmark = self.add_noise(data, wtmark)
        # attach the watermark to the data
        (w, h) = data.size
        wtmarked_image = Image.new('RGB', (w, h), (0,0,0))
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
    Spatial perturbation: apply the random (0.5 or 1.0) spatial displacement
                          to each pixel in the watermark.
"""
class WMarkerSpatialPerturbation():
    def __init__(self, watermark, bfactor, displace, nlevel, noise=False):
        super(WMarkerSpatialPerturbation, self).__init__()
        self.watermark = watermark
        self.bfactor   = bfactor
        self.displace  = displace
        # noise parameters
        self.mean      = 0.0
        self.nlevel    = nlevel
        self.noise     = noise

    def do_transform(self, data):
        """
            Transform data (expected as a PIL Image - h x w)
        """
        # load the watermark image
        #  & synthesize the displaced watermark
        wtmark = Image.open(self.watermark).convert('RGBA')
        wtdisp = self.create_disp_wmark(wtmark)
        # add noise to the watermark
        if self.noise: wtdisp = self.add_noise(data, wtdisp)
        # use the blend-factor value to the alpha channel
        (wR, wG, wB, wA) = wtdisp.split()
        wA = wA.point(lambda a: int(a * self.bfactor))
        wtdisp = Image.merge('RGBA', (wR, wG, wB, wA))
        # attach the watermark to the data
        (w, h) = data.size
        wtmarked_image = Image.new('RGB', (w, h), (0,0,0))
        wtmarked_image.paste(data, (0,0))
        wtmarked_image.paste(wtdisp, (0,0), mask=wtdisp)
        return wtmarked_image

    def create_disp_wmark(self, wtmark):
        # prep.: compute the proper w'mark tensor to displace
        (img_w, img_h) = wtmark.size
        pil_wtmark = np.array([np.array(wtmark)])
        ten_wtmark = torch.from_numpy(pil_wtmark)
        ten_wtmark = ten_wtmark.permute(0, 3, 2, 1).float()

        # prep.: compute the displace map (x, y) in [-1, 1]
        disp_ten = torch.rand(1, img_h, img_w, 2) * (self.displace * 2) - self.displace
        disp_arr = gaussian_filter(np.array(disp_ten), sigma=2)
        disp_ten = torch.from_numpy(disp_arr)

        # prep.: compute the flow map
        flow_idx = list(itertools.product(range(0, img_h), range(0, img_w)))
        flow_idx = [list(each) for each in flow_idx]
        flow_ten = torch.FloatTensor(flow_idx).view(1, img_h, img_w, 2)
        flow_ten = (flow_ten + disp_ten)

        # normalize flows
        flow_mean = torch.FloatTensor([img_h/2, img_w/2]).repeat(1, img_h, img_w, 1)
        flow_std  = torch.FloatTensor([img_h/2, img_w/2]).repeat(1, img_h, img_w, 1)
        flow_norm = ((flow_ten - flow_mean) / flow_std)
        flow_norm.clamp_(-self.displace, self.displace)

        # compute the displaced image
        ten_wtmark, flow_norm = Variable(ten_wtmark), Variable(flow_norm)
        dis_wtmark = F.grid_sample(ten_wtmark, flow_norm) / 255.0
        dis_wtmark = dis_wtmark[0]
        dis_wtmark = transforms.ToPILImage()(dis_wtmark)
        return dis_wtmark

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
    Main (to test purposes)
"""
if __name__ == '__main__':
    # define inputs
    wmark = '../etc/watermarks/cifar10_wmark_32x32_three.png'
    wdisp = 16.0
    wvari = 20.0
    wdipl = 1.0

    # init. the watermarker
    translate  = WMarkerTranslate(wmark, 0.5, wdisp, 0.4, noise=True)
    opavariate = WMarkerOpacityVariation(wmark, 0.5, wvari, 0.4, noise=True)
    spaperturb = WMarkerSpatialPerturbation(wmark, 0.5, wdipl, 0.4, noise=True)

    # apply the transforms
    sample  = Image.open('sample.png')
    tlated  = translate.do_transform(sample)
    ovariat = opavariate.do_transform(sample)
    spertur = spaperturb.do_transform(sample)

    # save images
    tlated.save('sample-translate.png')
    ovariat.save('sample-opavariate.png')
    spertur.save('sample-spaperturb.png')

    # Fin.
