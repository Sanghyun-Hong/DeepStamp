"""
    To test the pre-trained network
"""
import os, csv, json
import argparse
import numpy as np
from tqdm import tqdm

# torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

# torchvision modules
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms

# utils
from utils.dataloaders.custom import CustomDataset

# networks (cifar10)
from networks.cifar10.alexnet import Cifar10_AlexNet
from networks.cifar10.resnet import Cifar10_ResNet50
from networks.cifar10.vgg import Cifar10_VGG


###########################
# Define datasets/etc.
###########################
def define_network(dataset, datapath, netname):
    # Networks for CIFAR10
    if 'cifar10' in dataset:
        if 'AlexNet' == netname:
            return Cifar10_AlexNet()
        elif 'ResNet' == netname:
            return Cifar10_ResNet50()
        elif 'VGG16' == netname:
            return Cifar10_VGG('VGG16')
        else:
            assert False, ('Error: invalid network name [{}]'.format(netname))

    # Network for custom datasets
    elif 'custom' in dataset:
        # From: CIFAR10
        if 'cifar10' in datapath:
            if 'AlexNet' == netname:
                return Cifar10_AlexNet()
            elif 'ResNet' == netname:
                return Cifar10_ResNet50()
            elif 'VGG16' == netname:
                return Cifar10_VGG('VGG16')
            else:
                assert False, ('Error: invalid network {} for datapath {}'.format(netname, datapath))

        # From: unknown
        else:
            assert False, ('Error: custom dataset from undefined data [{}]'.format(datapath))

    # Undefined dataset
    else:
        assert False, ('Error: invalid dataset name [{}]'.format(dataset))

def load_trained_network(net, cuda, filepath):
    model_dict = torch.load(filepath) if cuda else \
                 torch.load(filepath, map_location=lambda storage, loc: storage)
    net.load_state_dict(model_dict)
    #done.

def load_dataset(dataset, parameters, kwargs):
    # CIFAR10 dataset
    if 'cifar10' == dataset:
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root='datasets/originals/cifar10',
                                         train=False, download=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                         ])),
                batch_size=parameters['params']['batch-size'], shuffle=False, **kwargs)

    # read the custom dataset
    elif 'custom' == dataset or 'paired' == dataset:
        # : raise an error when there is no datapath
        assert parameters['model']['datapath'], \
            ('Error: invalid \'datapath\' - undefined.')

        # : otherwise, load the data
        test_loader = torch.utils.data.DataLoader(
            CustomDataset(parameters['model']['datapath'],
                          train=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                          ])),
                batch_size=parameters['params']['batch-size'], shuffle=False, **kwargs)

    # Undefined dataset
    else:
        assert False, ('Error: invalid dataset name [{}]'.format(dataset))

    return test_loader


###########################
# Validation func.
###########################
def run_validation(parameters):

    # init. task name
    task_name = 'valid'

    # initialize the random seeds
    np.random.seed(parameters['system']['seed'])
    torch.manual_seed(parameters['system']['seed'])
    if parameters['system']['cuda']:
        torch.cuda.manual_seed(parameters['system']['seed'])

    # initialize dataset (train/test)
    kwargs = {
            'num_workers': parameters['system']['num-workers'],
            'pin_memory' : parameters['system']['pin-memory']
        } if parameters['system']['cuda'] else {}
    test_loader = load_dataset( \
        parameters['model']['dataset'], parameters, kwargs)

    # initialize the networks
    net = define_network(parameters['model']['dataset'],
                         parameters['model']['datapath'],
                         parameters['model']['network'])
    if parameters['model']['trained']:
        load_trained_network(net, \
                             parameters['system']['cuda'], \
                             parameters['model']['trained'])
    else:
        assert False, ('Error: cannot find any pre-traind net ...')
    netname = type(net).__name__
    if parameters['system']['cuda']: net.cuda()


    # init. output dirs
    store_paths = {}
    store_paths['prefix'] = define_store_prefix(parameters)
    store_paths['result'] = os.path.join('results', parameters['model']['dataset'], task_name)
    if not os.path.isdir(store_paths['result']): os.makedirs(store_paths['result'])

    """
        Validation (only run at once)
    """
    test(test_loader, parameters, net, netname, store_paths)
    # done.


###########################
# Train/Test Mains.
###########################
def test(test_loader, parameters, net, netname, store_paths):
    # set the model as eval mode
    net.eval()

    # data holders
    correct = 0

    # loop through the train data
    for data, target in tqdm(test_loader, desc='[valid (w. ext. data)]'):
        if parameters['system']['cuda']:
            data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, requires_grad=False), Variable(target)
        with torch.no_grad():
            outputs = net(data)
            """
                Compute the acc.
            """
            # Get the index of the max log-probability
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    # the current acc & report the result
    cur_acc = 100. * correct / len(test_loader.dataset)
    print ('Validation: [{}/{} (Acc: {:.4f}%)'.format( \
            correct, len(test_loader.dataset), cur_acc))
    # done.

###########################
# Misc. functions
###########################
def define_store_prefix(parameters):
    # append the basepath if datapath exists
    prefix = ''
    if parameters['model']['datapath']:
        tokens = parameters['model']['datapath'].split('/')
        ptoken = tokens[len(tokens)-1]
        prefix += '{}_'.format(ptoken)

    # append the other parameters
    if parameters['model']['dataset'] == 'custom':
        prefix += '{}_{}'.format( \
                parameters['model']['network'], \
                parameters['params']['batch-size'])
    else:
        prefix += '{}_{}'.format( \
                parameters['model']['network'], \
                parameters['params']['batch-size'])

    return prefix


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
    # load the model parameters
    parameters['model'] = {}
    parameters['model']['dataset'] = arguments.dataset
    parameters['model']['datapath'] = arguments.datapath
    parameters['model']['network'] = arguments.network
    parameters['model']['trained'] = arguments.trained
    parameters['model']['classes'] = arguments.classes
    # load the hyper-parameters
    parameters['params'] = {}
    parameters['params']['batch-size'] = arguments.batch_size
    # print out
    print(json.dumps(parameters, indent=2))
    return parameters

# cmdline interface (for backward compatibility)
if __name__ == '__main__':
    parser = argparse.ArgumentParser( \
        description='Validate a pre-trained network with a dataset.')

    # system parameters
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers (default: 4)')
    parser.add_argument('--pin-memory', action='store_false', default=True,
                        help='the data loader copies tensors into CUDA pinned memory')
    # model parameters
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset used to train: mnist.')
    parser.add_argument('--datapath', type=str, default='',
                        help='dataset stored location (\'custom\' dataset case)')
    parser.add_argument('--network', type=str, default='SampleNetV1',
                        help='model name (default: SampleNetV1).')
    parser.add_argument('--trained', type=str, default='',
                        help='pre-trained model filepath.')
    parser.add_argument('--classes', type=int, default=20,
                        help='number of classes - for multilabel datasets (default: 20).')
    # hyper-parmeters
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    # execution parameters
    args = parser.parse_args()

    # dump the input parameters
    parameters = dump_arguments(args)
    run_validation(parameters)
