import os, csv, json
import argparse
import numpy as np
from tqdm import tqdm

# matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
# Global Variables
###########################
best_acc     = 0
train_losses = []
valid_losses = []


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

def define_dataset(dataset, parameters, kwargs):
    # CIFAR10 dataset
    if 'cifar10' == dataset:
        # initialize CIFAR10 dataset (train)
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root='datasets/originals/cifar10',
                                         train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.RandomCrop(32, padding=4),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                         ])),
                batch_size=parameters['params']['batch-size'], shuffle=True, **kwargs)

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
        train_loader = torch.utils.data.DataLoader(
            CustomDataset(parameters['model']['datapath'],
                          train=True,
                          transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                    ])),
                batch_size=parameters['params']['batch-size'], shuffle=True, **kwargs)

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

    return train_loader, test_loader

def define_loss_function(lossname):
    # cross-entropy loss
    if 'cross-entropy' == lossname:
        return F.cross_entropy

    # Undefined loss functions
    else:
        assert False, ('Error: invalid loss function name [{}]'.format(lossname))


###########################
# Training functions
###########################
def run_training(parameters):

    # init. task name
    task_name = 'train'

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

    train_loader, test_loader = define_dataset( \
        parameters['model']['dataset'], parameters, kwargs)

    # initialize the networks
    net = define_network(parameters['model']['dataset'],
                         parameters['model']['datapath'],
                         parameters['model']['network'])
    if parameters['model']['trained']:
        load_trained_network(net, \
                             parameters['system']['cuda'], \
                             parameters['model']['trained'])
    netname = type(net).__name__
    if parameters['system']['cuda']: net.cuda()

    # init. loss function
    task_loss = define_loss_function(parameters['model']['loss'])

    # initialize the optimizer
    optimizer = optim.SGD(net.parameters(),
                          lr=parameters['params']['lr'],
                          momentum=parameters['params']['momentum'],
                          weight_decay=5e-04)
    scheduler = StepLR(optimizer,
                       step_size=parameters['params']['step-size'],
                       gamma=parameters['params']['gamma'])

    # init. output dirs
    store_paths = {}
    store_paths['prefix'] = define_store_prefix(parameters)
    store_paths['figure'] = os.path.join('figures', parameters['model']['dataset'], task_name)
    if not os.path.isdir(store_paths['figure']): os.makedirs(store_paths['figure'])
    store_paths['model']  = os.path.join('models', parameters['model']['dataset'], task_name)
    if not os.path.isdir(store_paths['model']): os.makedirs(store_paths['model'])
    store_paths['result'] = os.path.join('results', parameters['model']['dataset'], task_name)
    if not os.path.isdir(store_paths['result']): os.makedirs(store_paths['result'])

    # training
    for epoch in range(1, parameters['params']['epoch']+1):
        train(epoch, train_loader, parameters, \
              net, netname, task_loss, scheduler, optimizer)
        test(epoch, test_loader, parameters, \
             net, netname, task_loss, store_paths, save=True)

    # output: draw a plot that tracks train & valid losses per epoch
    store_loss_plot(train_losses, valid_losses, store_paths)

    # done.


###########################
# Train/Test Mains.
###########################
def train(epoch, train_loader, parameters, \
          net, netname, task_loss, scheduler, optimizer):
    global train_losses

    # data holders.
    tlosses = 0.

    # train...
    net.train()
    scheduler.step()
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader, \
        desc='[{}/{}]'.format(epoch, parameters['params']['epoch']))):
        # Note: once we use the cross-entropy
        if parameters['model']['loss'] == 'cross-entropy':
            targets = targets.type(torch.LongTensor)
        if parameters['system']['cuda']:
            data, targets = data.cuda(), targets.cuda()
        data, targets = Variable(data), Variable(targets)
        optimizer.zero_grad()
        outputs = net(data)
        """
            Loss function defined in the configuration file
        """
        # : compute loss value (default: element-wise mean)
        bsize = data.size()[0]
        loss = task_loss(outputs, targets)
        tlosses += (loss.data.item() * bsize)
        loss.backward()
        optimizer.step()

    # update the losses
    tlosses /= len(train_loader.dataset)
    train_losses.append(tlosses)
    # done.


def test(epoch, test_loader, parameters, \
         net, netname, task_loss, store_paths, save=False):
    # load the globals
    global best_acc
    global valid_losses

    # test
    net.eval()

    # acc. in total
    vlosses = 0.
    correct = 0

    for batch_idx, (data, targets) in enumerate(tqdm(test_loader, \
        desc='[{}/{}]'.format(epoch, parameters['params']['epoch']))):
        # Note: once we use the cross-entropy
        if parameters['model']['loss'] == 'cross-entropy':
            targets = targets.type(torch.LongTensor)
        if parameters['system']['cuda']:
            data, targets = data.cuda(), targets.cuda()
        data, targets = Variable(data, requires_grad=False), Variable(targets)
        with torch.no_grad():
            outputs = net(data)
            """
                Loss function defined in the configuration file
            """
            vlosses += task_loss(outputs, targets, reduction='sum').data.item() # sum up batch loss
            pred = outputs.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    # the current loss and accuracy
    vlosses /= len(test_loader.dataset)
    valid_losses.append(vlosses)
    cur_acc = 100. * correct / len(test_loader.dataset)

    # report the result
    print ('[{}/{}] Test Epoch: [{}/{} (Acc: {:.4f}% | Best: {:.4f}%)]\tAverage loss: {:.6f}'.format(
        epoch, parameters['params']['epoch'], correct, len(test_loader.dataset), cur_acc, best_acc, vlosses))

    # store the model and record the result to a csv file
    if save:
        # store the model
        model_savefile = '{}.pth'.format(store_paths['prefix'])
        model_savepath = os.path.join(store_paths['model'], model_savefile)
        if cur_acc > best_acc:
            torch.save(net.state_dict(), model_savepath)
            print (' -> cur acc. [{}] > best acc. [{}], store the model.\n'.format(cur_acc, best_acc))
            best_acc = cur_acc

        # record the result to a csv file
        datarow  = [epoch, vlosses, correct, len(test_loader.dataset), cur_acc]
        result_savefile = '{}.csv'.format(store_paths['prefix'])
        result_savepath = os.path.join(store_paths['result'], result_savefile)
        if epoch < 2 and os.path.exists(result_savepath): os.remove(result_savepath)
        csv_logger(datarow, result_savepath)
    # done.


###########################
# Misc. functions
###########################
def store_loss_plot(tlosses, vlosses, store_paths):

    # sanity checks
    if not tlosses or not vlosses: return

    # total epoch counts
    tlosses = np.array(tlosses)
    vlosses = np.array(vlosses)
    epochs  = len(tlosses) + 1

    # draw
    plt.plot(range(1, epochs), tlosses, label='Train', linewidth=1.0)
    plt.plot(range(1, epochs), vlosses, label='Valid', linewidth=1.0)
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Avg. loss per epoch')

    # store location
    if not os.path.exists(store_paths['figure']): os.makedirs(store_paths['figure'])
    plot_filename = '{}_losses.png'.format(store_paths['prefix'])
    plt.savefig(os.path.join(store_paths['figure'], plot_filename))
    plt.clf()
    # done.

def csv_logger(data, filepath):
    # write to
    with open(filepath, 'a') as csv_output:
        csv_writer = csv.writer(csv_output)
        csv_writer.writerow(data)
    # done.

def define_store_prefix(parameters):
    # append the basepath if datapath exists
    prefix = ''
    if parameters['model']['datapath']:
        tokens = parameters['model']['datapath'].split('/')
        tokens = tokens[1:]
        prefix = '_'.join(tokens)
        prefix += '_'

    # append the other parameters
    if parameters['model']['dataset'] == 'custom':
        prefix += '{}_{}_{}_{}_{}_{}_{}_{}'.format( \
                parameters['model']['network'], \
                parameters['model']['loss'], \
                parameters['params']['batch-size'], \
                parameters['params']['epoch'], \
                parameters['params']['lr'], \
                parameters['params']['momentum'], \
                parameters['params']['step-size'], \
                parameters['params']['gamma'])
    else:
        prefix += '{}_{}_{}_{}_{}_{}_{}_{}'.format( \
                parameters['model']['network'], \
                parameters['model']['loss'], \
                parameters['params']['batch-size'], \
                parameters['params']['epoch'], \
                parameters['params']['lr'], \
                parameters['params']['momentum'], \
                parameters['params']['step-size'], \
                parameters['params']['gamma'])

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
    parameters['model']['loss'] = arguments.loss
    parameters['model']['classes'] = arguments.classes
    # load the hyper-parameters
    parameters['params'] = {}
    parameters['params']['batch-size'] = arguments.batch_size
    parameters['params']['epoch'] = arguments.epoch
    parameters['params']['optimizer'] = arguments.optimizer
    parameters['params']['lr'] = arguments.lr
    parameters['params']['step-size'] = arguments.step_size
    parameters['params']['gamma'] = arguments.gamma
    parameters['params']['momentum'] = arguments.momentum
    # print out
    print(json.dumps(parameters, indent=2))
    return parameters

# cmdline interface (for backward compatibility)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Network')

    # system parameters
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers (default: 4)')
    parser.add_argument('--pin-memory', action='store_true', default=True,
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
    parser.add_argument('--loss', type=str, default='cross-entropy',
                        help='loss function name for this task (default: cross-entropy).')
    parser.add_argument('--classes', type=int, default=20,
                        help='number of classes - for multilabel datasets (default: 20).')
    # hyper-parmeters
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epoch', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer used to train (default: SGD)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--step-size', type=int, default=100, metavar='N',
                        help='step size for the learning rate (default: 100)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='N',
                        help='gamma for the learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    # execution parameters
    args = parser.parse_args()

    # dump the input parameters
    parameters = dump_arguments(args)
    run_training(parameters)
