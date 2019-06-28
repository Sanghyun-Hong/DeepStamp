"""
    Robust watermark scheme that uses GMAN approach
    (https://arxiv.org/pdf/1611.01673.pdf)
"""

# basics
import os, json, copy
import pickle
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

# matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# torch
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# torchvision
import torchvision
import torchvision.utils as vutils
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# utils
from utils.dataloaders.custom import CustomDataset

# networks (cifar10: trained)
from networks.cifar10.alexnet import Cifar10_AlexNet
from networks.cifar10.resnet import Cifar10_ResNet50
from networks.cifar10.vgg import Cifar10_VGG

# networks (w-marks)
from networks.cifar10.wmark import \
    Cifar10_Watermarker, Cifar10_AutoEncoder, Cifar10_Discriminator

# watermarkers [transform with input batches]
from transbatch.watermarker import WaterMarker


###########################
#    Definers
###########################
def define_dataset(dataset, datapth, batchsz, kwargs, concat=True):
    """
        Note: [important] data should be normalized in [0,1]: to Tensor.
    """
    # CIFAR10 dataset
    if 'cifar10' == dataset:
        # initialize CIFAR10 datasets
        train_set = datasets.CIFAR10(root='datasets/originals/cifar10', \
                                     train=True, download=True, \
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                     ]))
        valid_set = datasets.CIFAR10(root='datasets/originals/cifar10', train=False,
                                     download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                     ]))
        # init. loaders
        train_loader = torch.utils.data.DataLoader( \
            train_set, batch_size=batchsz, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader( \
            valid_set, batch_size=batchsz, shuffle=True, **kwargs)
        if concat:
            total_loader = torch.utils.data.DataLoader( \
                torch.utils.data.ConcatDataset([train_set, valid_set]),
                batch_size=batchsz, shuffle=True, **kwargs)
        else:
            total_loader = None

    # Undefined dataset
    else:
        assert False, ('Error: invalid dataset name [{}]'.format(dataset))

    return (train_loader, valid_loader, total_loader)

def define_watermarker(dataset, datapath):
    # Networks for CIFAR10 original dataset
    if 'cifar10' == dataset:
        return Cifar10_Watermarker(activation='Tanh')

    # Undefined dataset
    else:
        assert False, ('Error: invalid dataset name [{}]'.format(dataset))

def define_vintactor(dataset, datapath, ndf):
    # Networks for CIFAR10 original dataset
    if 'cifar10' == dataset:
        return Cifar10_AutoEncoder()

    # Undefined dataset
    else:
        assert False, ('Error: invalid dataset name [{}]'.format(dataset))

def define_discriminator(dataset, datapath, ndf):
    # Networks for CIFAR10 original dataset
    if 'cifar10' == dataset:
        return Cifar10_Discriminator()

    # Undefined dataset
    else:
        assert False, ('Error: invalid dataset name [{}]'.format(dataset))

def define_trained_network(dataset, datapath, netname):
    # Networks for CIFAR10 original dataset
    if 'cifar10' == dataset:
        if 'AlexNet' == netname:
            return Cifar10_AlexNet()
        elif 'ResNet' == netname:
            return Cifar10_ResNet50()
        elif 'VGG16' == netname:
            return Cifar10_VGG('VGG16')
        else:
            assert False, ('Error: invalid network for {} - {}'.format(dataset, netname))

        # From: unknown
        else:
            assert False, ('Error: paired dataset from undefined data [{}]'.format(datapath))

    # Undefined dataset
    else:
        assert False, ('Error: invalid dataset name [{}]'.format(dataset))

def define_loss_function(lossname):
    # L1 loss
    if 'l1' == lossname:
        return F.l1_loss

    # L2 loss
    elif 'l2' == lossname:
        return F.mse_loss

    # cross-entropy loss
    elif 'cross-entropy' == lossname:
        return F.cross_entropy

    # bce loss
    elif 'binary-cross-entropy' == lossname:
        return F.binary_cross_entropy

    # multi-label soft margin loss
    elif 'multilabel-soft-margin' == lossname:
        return F.multilabel_soft_margin_loss

    # bce loss with logits
    elif 'binary-cross-entropy-with-logits' == lossname:
        return F.binary_cross_entropy_with_logits

    # Undefined loss functions
    else:
        assert False, ('Error: invalid loss function name [{}]'.format(lossname))

def define_transform(parameters):
    # normal watermark (w. blend-factor)
    if parameters['wmark']['transform'] == 'watermark':
        transform = WaterMarker(parameters['wmark']['blend-factor'], \
                                parameters['wmark']['noise-level'], \
                                noise=parameters['wmark']['noise'])

    # undefined transform
    else:
        assert False, ("Error: invalid transform - {}".format(parameters['trans']['transform']))
    return transform


###########################
#    Data loaders
###########################
def load_watermark(filepath):
    # load the watermark image
    # Note: should be normalized in [0,1]
    watermark = Image.open(filepath)
    tramsform = transforms.Compose([
            transforms.ToTensor(),
        ])
    watermark = tramsform(watermark)
    # load the filename
    name_tokens = filepath.split('/')
    wmark_fname = name_tokens[len(name_tokens)-1].replace('.png', '')
    return watermark, wmark_fname


def load_pretrained_network(net, cuda, filepath):
    model_dict = torch.load(filepath) if cuda else \
                 torch.load(filepath, map_location=lambda storage, loc: storage)
    net.load_state_dict(model_dict)
    # done.


###########################
#    Training functions
###########################
def run_training(parameters):

    # init. task name
    task_name = 'train-wmark'

    # initialize the random seeds (fix the random seeds)
    random.seed(parameters['system']['seed'])
    np.random.seed(parameters['system']['seed'])
    torch.manual_seed(parameters['system']['seed'])
    if parameters['system']['cuda']:
        torch.cuda.manual_seed(parameters['system']['seed'])

    # initialize dataset (train/test)
    kwargs = {
            'num_workers': parameters['system']['num-workers'],
            'pin_memory' : parameters['system']['pin-memory']
        } if parameters['system']['cuda'] else {}

    train_loader, valid_loader, total_loader = define_dataset( \
        parameters['model']['dataset'], \
        parameters['model']['datapath'], \
        parameters['params']['batch-size'], \
        kwargs, concat=True)

    # initialize dataset (load the watermark image/name)
    watermark, wmarkname = \
        load_watermark(parameters['wmark']['wmark-file'])

    # initialize transform
    transform = define_transform(parameters)

    # initialize networks (Generator)
    netG = define_watermarker(parameters['model']['dataset'], \
                              parameters['model']['datapath'])
    netG_name = type(netG).__name__
    if parameters['system']['cuda']: netG.cuda()

    # initialize networks (Auto-encoder: EBGAN)
    netV = define_vintactor(parameters['model']['dataset'], \
                            parameters['model']['datapath'], \
                            parameters['params']['ndf'])
    netV_name = type(netV).__name__
    if parameters['system']['cuda']: netV.cuda()

    # initialize networks (Discriminator)
    netD = define_discriminator(parameters['model']['dataset'], \
                                parameters['model']['datapath'], \
                                parameters['params']['ndf'])
    netD_name = type(netD).__name__
    if parameters['system']['cuda']: netD.cuda()

    # initialize the pre-trained network
    netT = define_trained_network(parameters['model']['dataset'], \
                                  parameters['model']['datapath'], \
                                  parameters['model']['network'])
    load_pretrained_network(netT, parameters['system']['cuda'],
                            parameters['model']['netpath'])
    netT_name = type(netT).__name__
    if parameters['system']['cuda']: netT.cuda()

    # init. network in use
    network_use = netT_name
    network_use = parameters['model']['netpath'].split('/')
    network_use = network_use[len(network_use)-1].replace('.pth', '')

    # initialize the loss functions
    vloss = define_loss_function(parameters['model']['vloss'])
    dloss = define_loss_function(parameters['model']['dloss'])
    tloss = define_loss_function(parameters['model']['tloss'])

    # initialize the optimizers
    # - Note: we don't train netT, netV at this moment.
    optG = optim.Adam(netG.parameters(), \
                      lr=parameters['params']['Glr'], \
                      betas=(0.5, 0.999))
    optV = optim.Adam(netV.parameters(), \
                      lr=parameters['params']['Vlr'], \
                      betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), \
                      lr=parameters['params']['Dlr'], \
                      betas=(0.5, 0.999))

    # init. output dirs
    # - data  : loc. to store data
    # - model : loc. to store trained models
    # - sample: loc. to store sample outputs
    # - result: loc. to store result metrics
    # - figure: loc. to store figures (e.g., losses)
    store_paths = {}
    store_paths['prefix'] = define_store_prefix(parameters)
    store_paths['data']   = os.path.join( \
        'datasyns', parameters['model']['dataset'], task_name, wmarkname, network_use)
    if not os.path.isdir(store_paths['data']): os.makedirs(store_paths['data'])
    store_paths['model']  = os.path.join( \
        'models', parameters['model']['dataset'], task_name, wmarkname, network_use)
    if not os.path.isdir(store_paths['model']): os.makedirs(store_paths['model'])
    store_paths['sample'] = os.path.join( \
        'samples', parameters['model']['dataset'], task_name, wmarkname, network_use)
    if not os.path.isdir(store_paths['sample']): os.makedirs(store_paths['sample'])
    store_paths['result'] = os.path.join( \
        'results', parameters['model']['dataset'], task_name, wmarkname, network_use)
    if not os.path.isdir(store_paths['result']): os.makedirs(store_paths['result'])
    store_paths['figure'] = os.path.join( \
        'figures', parameters['model']['dataset'], task_name, wmarkname, network_use)
    if not os.path.isdir(store_paths['figure']): os.makedirs(store_paths['figure'])

    # to track losses
    track_losses = []

    # training
    for epoch in range(1, parameters['params']['epoch']+1):
        cur_losses = train_wmarks(epoch, parameters, \
                                  total_loader, valid_loader, \
                                  netG, netV, netD, netT, \
                                  netG_name, netV_name, netD_name, netT_name, \
                                  vloss, dloss, tloss, \
                                  optG, optV, optD, \
                                  watermark, transform, \
                                  store_paths)
        track_losses.append(cur_losses)

    # end for epoch...

    # output: Draw a plot that tracks losses per epoch
    store_loss_plot(track_losses, store_paths)

    # output: Generate the watermarked images and store
    store_size = parameters['wmark']['store-batch']

    # output: store the created datasets
    train_loader, valid_loader, _ = define_dataset( \
        parameters['model']['dataset'], parameters['model']['datapath'], \
        parameters['params']['batch-size'], kwargs, concat=False)

    # output: store the pickle data
    store_wmarked_pdata(parameters, train_loader, valid_loader, \
                        netG, watermark, store_paths)
    # done.


###########################
# Train/Test Functions
###########################
def train_wmarks(epoch, parameters, \
                 total_loader, valid_loader, \
                 netG, netV, netD, netT, \
                 netG_name, netV_name, netD_name, netT_name, \
                 vloss, dloss, tloss, \
                 optG, optV, optD, \
                 watermark, transform, \
                 store_paths, debug=False):

    # init.
    netT.eval()

    # init. labels for the discriminator
    real_label, fake_label = 1, 0

    # data holders (loss functions)
    avg_losses = 0.0
    avg_vloss  = 0.0
    avg_dloss  = 0.0

    avgG_loss  = 0.0
    avgG_vloss = 0.0
    avgG_dloss = 0.0
    avgG_tloss = 0.0

    # train
    for bidx, (data, labels) in enumerate(tqdm(total_loader, \
        desc='[{}/{}]'.format(epoch, parameters['params']['epoch']))):

        """
            Note: pre-process inputs
             - convers labels to LongTensor
        """
        if parameters['model']['tloss'] == 'cross-entropy':
            labels = labels.type(torch.LongTensor)
        if parameters['system']['cuda']:
            data, labels = data.cuda(), labels.cuda()


        """
            Synthesize other data for training
            - Real/fake labels: for the adversarial loss
        """
        # real/fake labels
        batches = data.size()[0]
        rlabels = torch.full((batches,), real_label)
        flabels = torch.full((batches,), fake_label)
        if parameters['system']['cuda']:
            rlabels, flabels = rlabels.cuda(), flabels.cuda()

        # a batch of watermarks
        wmarks_batch = watermark.repeat(batches, 1, 1, 1)
        if parameters['system']['cuda']:
            wmarks_batch = wmarks_batch.cuda()


        """
            Synthesize the data to train from data, wmark, and labels
             - Real  : (for GAN) data + wmark + labels
                [In  : data, wmark, transform, params]
                [out : noised wmarked-data           ]
             - Wmarks: a batch of the watermarks
        """
        wmarked_data = transform.do_transform(data, wmarks_batch)
        if parameters['system']['cuda']:
            wmarked_data = wmarked_data.cuda()

        # check wartermark
        if debug:
            vutils.save_image(wmarked_data, 'wmarked-real.png')


        """
            Update the discriminator networks (Ds).
             - V: takes the original watermarks and returns the same
             - D: takes the watermarked-data and returns the real labels
        """
        netV.zero_grad()
        netV_doutputs = netV(wmarks_batch)
        netV_doutloss = vloss(netV_doutputs, wmarks_batch)
        V_x = netV_doutloss.mean().item()
        netV_doutloss.backward()

        netD.zero_grad()
        netD_doutputs = netD(wmarked_data)
        netD_doutloss = dloss(netD_doutputs, rlabels)
        D_x = netD_doutloss.mean().item()
        netD_doutloss.backward()


        """
            Generate the fake watermarked-data
        """
        # fake data (fake-w'mark)
        fake_input  = torch.cat((data, wmarks_batch), dim=1)
        fake_wmark  = netG(fake_input)
        fake_walpha = fake_wmark[:,3:4] * parameters['wmark']['blend-factor']
        fwmark_data = fake_walpha * fake_wmark[:,0:3] + (1 - fake_walpha) * data

        # check data...
        if debug:
            vutils.save_image(fwmark_data, 'wmarked-fake.png')

        netV_foutputs = netV(fake_wmark.detach())
        netV_foutloss = vloss(netV_foutputs, wmarks_batch)
        V_x += netV_foutloss.mean().item()
        netV_foutloss.backward()

        netD_foutputs = netD(fwmark_data.detach())
        netD_foutloss = dloss(netD_foutputs, flabels)
        D_x += netD_foutloss.mean().item()
        netD_foutloss.backward()

        optV.step()
        optD.step()


        """
            Update the w'marker network (G)
        """
        netG.zero_grad()

        # update the G w.r.t the V and D losses
        netV_foutputs = netV(fake_wmark)
        netV_foutloss = vloss(netV_foutputs, wmarks_batch)
        _V_Gx = netV_foutloss.mean().item()

        netD_foutputs = netD(fwmark_data)
        netD_foutloss = dloss(netD_foutputs, rlabels)
        _D_Gx = netD_foutloss.mean().item()

        # update the G w.r.t the classification loss + logits
        netT_foutputs = netT(fwmark_data)
        netT_foutloss = tloss(netT_foutputs, labels)
        netT_loss = netT_foutloss
        _T_Gx = netT_loss.mean().item()

        """
            Compute the Gradient Penalty (GP)
        """
        netT_prime = copy.deepcopy(netT)
        netT_prime.train()

        # w.r.t the input (x)
        netT_prime.zero_grad()
        netT_prime_dloss = F.cross_entropy( \
            netT_prime(data), labels)
        netT_prime_dloss.backward(retain_graph=True)
        netT_prime_dgrads = []
        for params in netT_prime.parameters():
            netT_prime_dgrads.append(copy.deepcopy(params.grad))

        # w.r.t the input (x')
        netT_prime.zero_grad()
        netT_prime_floss = F.cross_entropy( \
            netT_prime(fwmark_data.detach()), labels)
        netT_prime_floss.backward(retain_graph=True)
        netT_prime_fgrads = []
        for params in netT_prime.parameters():
            netT_prime_fgrads.append(copy.deepcopy(params.grad))

        # compute the GP in here
        netT_penalty = 0.
        for idx in range(len(netT_prime_dgrads)):
            cur_diff = (netT_prime_fgrads[idx] - netT_prime_dgrads[idx])
            netT_penalty += cur_diff.pow(2).sum()
        netT_penalty = netT_penalty.sqrt()

        # Do back propagation
        # (Note: T update is essential since it works as a regularizer)
        netG_loss = \
            parameters['params']['dratio'] * netD_foutloss \
            + parameters['params']['vratio'] * netV_foutloss \
            + parameters['params']['tratio'] * netT_loss \
            + 1.0 * netT_penalty
        netG_loss.backward()
        _Gx = netG_loss.item()


        optG.step()


        """
            Compute the loss updates
        """

        # disc.s, losses
        avg_losses += (D_x + V_x)
        avg_vloss  += V_x
        avg_dloss  += D_x

        # watermarker (G) losses
        avgG_loss  += _Gx
        avgG_vloss += _V_Gx
        avgG_dloss += _D_Gx
        avgG_tloss += _T_Gx

        # output: the real/fake samples
        if bidx != 0 and bidx % 100 == 0 \
            and epoch in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 50, 100, 150, 200]:
            vutils.save_image(data,
                    os.path.join(store_paths['sample'], \
                                 '{}_{}_orig.png'.format( \
                                 store_paths['prefix'], epoch)), \
                                 normalize=True)
            vutils.save_image(wmarked_data,
                    os.path.join(store_paths['sample'], \
                                 '{}_{}_real.png'.format( \
                                 store_paths['prefix'], epoch)), \
                                 normalize=True)
            vutils.save_image(fwmark_data,
                    os.path.join(store_paths['sample'], \
                                 '{}_{}_fake.png'.format( \
                                 store_paths['prefix'], epoch)), \
                                 normalize=True)
            # :: store the difference as a grayscale image
            fwmark_diff = torch.mean((fwmark_data - data), dim=1)
            fwmark_diff = fwmark_diff.unsqueeze(dim=1)
            vutils.save_image(fwmark_diff,
                    os.path.join(store_paths['sample'], \
                                 '{}_{}_diff.png'.format( \
                                 store_paths['prefix'], epoch)), \
                                 normalize=True)

    # for batch_idx...

    # output: the average losses
    print ('[{}/{}] Ds-All: {:.4f} [V:{:.4f} D:{:.4f}], '
           'G: {:.4f} [V:{:.4f} D:{:.4f}, CE:{:.4f}]'.format( \
           epoch, parameters['params']['epoch'], \
           avg_losses/len(total_loader), \
           avg_vloss/len(total_loader), \
           avg_dloss/len(total_loader), \
           avgG_loss/len(total_loader), \
           avgG_vloss/len(total_loader), \
           avgG_dloss/len(total_loader), \
           avgG_tloss/len(total_loader)))

    # output: compute the acc
    compute_acc(epoch, parameters, valid_loader, watermark, netG, netT)


    # output: store the models
    if epoch == parameters['params']['epoch']:
        torch.save(netG.state_dict(), \
                   os.path.join(store_paths['model'], \
                                '{}_{}_Watermark.pth'.format( \
                                    store_paths['prefix'], epoch)))
        torch.save(netV.state_dict(), \
                   os.path.join(store_paths['model'], \
                                '{}_{}_VIntactor.pth'.format( \
                                    store_paths['prefix'], epoch)))
        torch.save(netD.state_dict(), \
                   os.path.join(store_paths['model'], \
                                '{}_{}_Discriminator.pth'.format( \
                                    store_paths['prefix'], epoch)))
    # end if epoch...

    # return the losses
    # - total Gs loss
    # - individual losses: (V(ae, cl), T)
    return [
            # Generator losses
            avgG_loss/len(total_loader),
            avgG_vloss/len(total_loader),
            avgG_dloss/len(total_loader),
            avgG_tloss/len(total_loader),
            # Discriminator losses
            avg_losses/len(total_loader),
            avg_vloss/len(total_loader),
            avg_dloss/len(total_loader),
        ]

def compute_acc(cur, parameters, valid_loader, watermark, netG, netT):

    # data holders
    correct = 0
    epoches = parameters['params']['epoch']
    is_cuda = parameters['system']['cuda']
    blend_f = parameters['wmark']['blend-factor']

    # loop through the validation set
    for bidx, (data, labels) in enumerate( \
        tqdm(valid_loader, desc='[{}/{}]'.format(cur, epoches))):

        # : prepare inputs [cuda]
        batches  = data.size()[0]
        wbatches = watermark.repeat(batches, 1, 1, 1)
        if is_cuda:
            data, labels, wbatches = \
                data.cuda(), labels.cuda(), wbatches.cuda()

        # : compute the accuracy
        with torch.no_grad():

            # :: synthesize new fake data (fake-w'mark)
            data_input  = torch.cat((data, wbatches), dim=1)
            fake_wmark  = netG(data_input)
            fake_walpha = fake_wmark[:,3:4] * blend_f
            fwmark_data = fake_walpha * fake_wmark[:,0:3] + (1 - fake_walpha) * data

            # :: compute the correctness
            outputs = netT(fwmark_data)

            # :: Get the index of the max log-probability
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.long().data.view_as(pred)).cpu().sum().item()

    # the current acc & report the result
    cur_acc = 100. * correct / len(valid_loader.dataset)
    print ('[{}/{}] Test w. fake [{}/{} (Acc: {:.4f}%)'.format( \
            cur, epoches, correct, len(valid_loader.dataset), cur_acc))
    # done.


###########################
# Misc. functions
###########################
def define_store_prefix(parameters):
    # blend information to the prefix
    prefix = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format( \
            # loss-types
            parameters['model']['vloss'], \
            parameters['model']['dloss'], \
            parameters['model']['tloss'], \
            # watermarks
            parameters['wmark']['noise'], \
            parameters['wmark']['noise-level'], \
            parameters['wmark']['blend-factor'], \
            # learning hyper-params
            parameters['params']['batch-size'], \
            parameters['params']['epoch'], \
            parameters['params']['Glr'], \
            parameters['params']['Vlr'], \
            parameters['params']['Dlr'], \
            # loss hyper-params
            parameters['params']['vratio'], \
            parameters['params']['dratio'], \
            parameters['params']['tratio'])
    return prefix

def store_loss_plot(losses, store_paths):

    # sanity checks
    if not losses: return

    # transpose:
    # - [epoch1-losses, epoch2-losses, ...]
    # - to [G-losses (entire epochs), D1-losses, ...]
    losses = np.array(losses).transpose()
    epochs = losses.shape[1]

    # draw
    plt.plot(range(1, epochs+1), losses[0], label='G (tot.)', linewidth=1.0)
    plt.plot(range(1, epochs+1), losses[1], label='G (ae. )', linewidth=1.0)
    plt.plot(range(1, epochs+1), losses[2], label='G (dis.)', linewidth=1.0)
    plt.plot(range(1, epochs+1), losses[3], label='G (task)', linewidth=1.0)
    plt.plot(range(1, epochs+1), losses[4], label='D (tot.)', linewidth=1.0)
    plt.plot(range(1, epochs+1), losses[5], label='D (ae. )', linewidth=1.0)
    plt.plot(range(1, epochs+1), losses[6], label='D (dis.)', linewidth=1.0)
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Avg. loss per epoch')

    # store location
    if not os.path.exists(store_paths['figure']): os.makedirs(store_paths['figure'])
    plot_filename = '{}_losses.png'.format(store_paths['prefix'])
    plt.savefig(os.path.join(store_paths['figure'], plot_filename))
    plt.clf()
    # done.

def store_wmarked_pdata(parameters, train_loader, valid_loader, \
                        netG, watermark, store_paths):
    """
        Train: store the w'marked train data
    """
    # init. Generator mode
    netG.eval()

    # data
    is_cuda = parameters['system']['cuda']
    blend_f = parameters['wmark']['blend-factor']

    # init. data holders
    wmarked_data   = []
    wmarked_labels = []

    # create the w'marked train data
    for bidx, (data, labels) in enumerate( \
        tqdm(train_loader, desc='[store:train]')):

        # : prepare inputs [cuda]
        batches  = data.size()[0]
        wbatches = watermark.repeat(batches, 1, 1, 1)
        if is_cuda:
            data, labels, wbatches = \
                data.cuda(), labels.cuda(), wbatches.cuda()

        # : synthesize new fake data (fake-w'mark)
        data_input  = torch.cat((data, wbatches), dim=1)
        fake_wmark  = netG(data_input)
        fake_walpha = fake_wmark[:,3:4] * blend_f
        fwmark_data = fake_walpha * fake_wmark[:,0:3] + (1 - fake_walpha) * data

        # : move to the data-holders
        if is_cuda:
            cur_wdatas = fwmark_data.data.cpu().numpy() * 255.0
            cur_wdatas = cur_wdatas.astype(np.uint8)
            cur_labels = labels.data.cpu().numpy().tolist()
        else:
            cur_wdatas = fwmark_data.data.numpy() * 255.0
            cur_wdatas = cur_wdatas.astype(np.uint8)
            cur_labels = labels.data.numpy().tolist()

        # : store...
        wmarked_data   += [each_data for each_data in cur_wdatas]
        wmarked_labels += cur_labels
    # end for...

    # sanity checks
    assert len(wmarked_data) == len(wmarked_labels), \
        ('Error: train data/labels [{}/{}]'.format(len(wmarked_data), len(wmarked_labels)))

    # store the data, labels
    chunk_data  = []
    chunk_label = []
    chunk_size  = parameters['wmark']['store-batch']

    for idx in range(len(wmarked_data)):
        # : add the data
        chunk_data.append(wmarked_data[idx])
        chunk_label.append(wmarked_labels[idx])

        # : save the data (in every i-th chunk)
        if (idx+1) % chunk_size == 0:
            store_chunk = {
                    'data'  : np.asarray(chunk_data),
                    'labels': chunk_label,
                }

            # :: store to the correct location
            chunk_filename = 'train_{}.pkl'.format(int((idx+1)/chunk_size))
            chunk_savepath = os.path.join(store_paths['data'], store_paths['prefix'])
            if not os.path.exists(chunk_savepath): os.makedirs(chunk_savepath)
            pickle.dump(store_chunk, open(os.path.join(chunk_savepath, chunk_filename), 'wb'))

            # :: clear the holders
            chunk_data  = []
            chunk_label = []
    # for idx...

    # remainders
    if chunk_data and chunk_label:
        store_chunk = {
                'data'  : np.asarray(chunk_data),
                'labels': chunk_label,
            }

        # : store to the correct location
        chunk_filename = 'train_{}.pkl'.format(int(len(wmarked_data)/chunk_size) + 1)
        chunk_savepath = os.path.join(store_paths['data'], store_paths['prefix'])
        if not os.path.exists(chunk_savepath): os.makedirs(chunk_savepath)
        pickle.dump(store_chunk, open(os.path.join(chunk_savepath, chunk_filename), 'wb'))
    # end if chunk_data...

    """
        Test: store the w'marked test data
    """
    # init. data holders
    wmarked_data   = []
    wmarked_labels = []

    # create the w'marked valid data
    for bidx, (data, labels) in enumerate( \
        tqdm(valid_loader, desc='[store:valid]')):

        # : prepare inputs [cuda]
        batches  = data.size()[0]
        wbatches = watermark.repeat(batches, 1, 1, 1)
        if is_cuda:
            data, labels, wbatches = \
                data.cuda(), labels.cuda(), wbatches.cuda()

        # : synthesize new fake data (fake-w'mark)
        data_input  = torch.cat((data, wbatches), dim=1)
        fake_wmark  = netG(data_input)
        fake_walpha = fake_wmark[:,3:4] * blend_f
        fwmark_data = fake_walpha * fake_wmark[:,0:3] + (1 - fake_walpha) * data

        # : move to the data-holders
        if is_cuda:
            cur_wdatas = fwmark_data.data.cpu().numpy() * 255.0
            cur_wdatas = cur_wdatas.astype(np.uint8)
            cur_labels = labels.data.cpu().numpy().tolist()
        else:
            cur_wdatas = fwmark_data.data.numpy() * 255.0
            cur_wdatas = cur_wdatas.astype(np.uint8)
            cur_labels = labels.data.numpy().tolist()

        # : store
        wmarked_data   += [each_data for each_data in cur_wdatas]
        wmarked_labels += cur_labels
    # end for...

    # sanity checks
    assert len(wmarked_data) == len(wmarked_labels), \
        ('Error: test data/labels [{}/{}]'.format(len(wmarked_data), len(wmarked_labels)))

    # store the data, labels
    chunk_data  = []
    chunk_label = []

    for idx in range(len(wmarked_data)):
        # : add the data
        chunk_data.append(wmarked_data[idx])
        chunk_label.append(wmarked_labels[idx])

        # : save the data (in every i-th chunk)
        if (idx+1) % chunk_size == 0:
            store_chunk = {
                    'data'  : np.asarray(chunk_data),
                    'labels': chunk_label,
                }

            # :: store to the correct location
            chunk_filename = 'valid_{}.pkl'.format(int((idx+1)/chunk_size))
            chunk_savepath = os.path.join(store_paths['data'], store_paths['prefix'])
            if not os.path.exists(chunk_savepath): os.makedirs(chunk_savepath)
            pickle.dump(store_chunk, open(os.path.join(chunk_savepath, chunk_filename), 'wb'))

            # :: clear the holders
            chunk_data  = []
            chunk_label = []
    # for idx...

    # remainders
    if chunk_data and chunk_label:
        store_chunk = {
                'data'  : np.asarray(chunk_data),
                'labels': chunk_label,
            }

        # : store to the correct location
        chunk_filename = 'valid_{}.pkl'.format(int(len(wmarked_data)/chunk_size) + 1)
        chunk_savepath = os.path.join(store_paths['data'], store_paths['prefix'])
        if not os.path.exists(chunk_savepath): os.makedirs(chunk_savepath)
        pickle.dump(store_chunk, open(os.path.join(chunk_savepath, chunk_filename), 'wb'))
    # end if chunk_data...
    # done.


###########################
# Execution functions
###########################
def dump_arguments(arguments):
    parameters = dict()
    # load the system parameters
    parameters['system'] = {}
    parameters['system']['seed'] = arguments.seed
    parameters['system']['cuda'] = (not arguments.no_cuda and torch.cuda.is_available())
    parameters['system']['num-workers'] = arguments.num_workers
    parameters['system']['pin-memory'] = arguments.pin_memory
    # hyper parameters
    # .. for models ...
    parameters['model'] = {}
    parameters['model']['dataset'] = arguments.dataset
    parameters['model']['datapath'] = arguments.datapath
    parameters['model']['network'] = arguments.network
    parameters['model']['netpath'] = arguments.netpath
    parameters['model']['vloss'] = arguments.vloss
    parameters['model']['dloss'] = arguments.dloss
    parameters['model']['tloss'] = arguments.tloss
    # .. for watermarking ...
    parameters['wmark'] = {}
    parameters['wmark']['transform'] = arguments.transform
    parameters['wmark']['wmark-file'] = arguments.wmark_file
    parameters['wmark']['noise'] = arguments.noise
    parameters['wmark']['noise-level'] = arguments.noise_level
    parameters['wmark']['blend-factor'] = arguments.blend_factor
    parameters['wmark']['store-batch'] = arguments.store_batch
    # .. for training ...
    parameters['params'] = {}
    parameters['params']['batch-size'] = arguments.batch_size
    parameters['params']['epoch'] = arguments.epoch
    parameters['params']['ndf'] = arguments.ndf
    parameters['params']['Glr'] = arguments.Glr
    parameters['params']['Vlr'] = arguments.Vlr
    parameters['params']['Dlr'] = arguments.Dlr
    # .. for training .. (losses)
    parameters['params']['vratio'] = arguments.vratio
    parameters['params']['dratio'] = arguments.dratio
    parameters['params']['tratio'] = arguments.tratio
    # print out
    print(json.dumps(parameters, indent=2))
    return parameters

# cmdline interface (for backward compatibility)
if __name__ == '__main__':
    parser = argparse.ArgumentParser( \
        description='Train a Watermarking Network (use GAN)')

    # system parameters
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers (default: 4)')
    parser.add_argument('--pin-memory', action='store_false', default=True,
                        help='the data loader copies tensors into CUDA pinned memory')
    # hyper-parameters
    # .. for models ..
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset used to train (default: cifar10).')
    parser.add_argument('--datapath', type=str, default='',
                        help='dataset stored location (\'custom\' dataset case)')
    parser.add_argument('--network', type=str, default='AlexNet',
                        help='pre-trained network name (default: AlexNet).')
    parser.add_argument('--netpath', type=str, default='models/cifar10/train/AlexNet_cross-entropy_128_300_0.01_0.9_10_0.95.pth',
                        help='pre-trained network location (default: models/cifar10/train/AlexNet_cross-entropy_128_300_0.01_0.9_10_0.95.pth).')
    parser.add_argument('--vloss', type=str, default='l1',
                        help='loss function name for the auto-encoder for the preserver (default: l1).')
    parser.add_argument('--dloss', type=str, default='binary-cross-entropy',
                        help='loss function name for the classifier for the preserver (default: BCE).')
    parser.add_argument('--tloss', type=str, default='cross-entropy',
                        help='loss function name for task network (default: CE).')
    # .. for watermarking ...
    parser.add_argument('--transform', type=str, default='watermark',
                        help='transformation name (default: watermark).')
    parser.add_argument('--wmark-file', type=str, default='etc/watermarks/cifar10_wmark_32x32_three.png',
                        help='watermark image location (default: etc/watermarks/cifar10_wmark_32x32_three.png)')
    parser.add_argument('--noise', action='store_true', default=False,
                        help='noise to watermark (default: false).')
    parser.add_argument('--noise-level', type=float, default=0.1,
                        help='noise level applied to watermark images (default: 0.1).')
    parser.add_argument('--blend-factor', type=float, default=0.5,
                        help='alpha value for the watermark (default: 0.5).')
    parser.add_argument('--store-batch', type=int, default=10000,
                        help='store batch size for storing watermarked data (default: 10k)')
    # .. for training ...
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epoch', type=int, default=1,
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--ndf', type=int, default=4,
                        help='the number of dimensions as feature (default: 4).')
    parser.add_argument('--Glr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--Vlr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--Dlr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    # .. for training .. (losses)
    parser.add_argument('--vratio', type=float, default=1.0,
                        help='ratio between the AE loss over classification loss (default: 1.0).')
    parser.add_argument('--dratio', type=float, default=1.0,
                        help='ratio between the D loss over classification loss (default: 1.0).')
    parser.add_argument('--tratio', type=float, default=1.0,
                        help='ratio between the CE loss over Ds\' losses (default: 10.0).')

    # execution parameters
    args = parser.parse_args()

    # dump the input parameters
    parameters = dump_arguments(args)
    run_training(parameters)
