"""
    Apply watemark (transform) and save the data
"""
import os, json
import pickle
import numpy as np
from tqdm import tqdm
from datetime import datetime

# torch modules
import torch
from torch.autograd import Variable

# torchvision modules
import torchvision
import torchvision.utils as vutils
from torchvision import datasets, transforms

# utils
from utils.dataloaders.custom import CustomDataset

# transform modules
from transforms.watermarker import WaterMarker
from transforms.watermarker_cvpr import \
    WMarkerTranslate, WMarkerOpacityVariation, WMarkerSpatialPerturbation


###########################
# Main Training Functions
###########################
def define_dataset(dataset, parameters, kwargs):
    # CIFAR10 dataset
    if 'cifar10' == dataset:
        # initialize CIFAR10 dataset
        # - Note: each dataset returns the PIL image without transform
        train_dataset = datasets.CIFAR10(root='datasets/originals/cifar10', train=True,  download=True)
        valid_dataset = datasets.CIFAR10(root='datasets/originals/cifar10', train=False, download=True)

    # Custom dataset
    elif 'custom' == dataset:
        # : raise an error when there is no datapath
        assert parameters['trans']['datapath'], \
            ('Error: invalid \'datapath\' - undefined.')

        # : otherwise, load the data
        train_dataset = CustomDataset(parameters['trans']['datapath'], train=True)
        valid_dataset = CustomDataset(parameters['trans']['datapath'], train=False)

    # Undefined dataset
    else:
        assert False, ('Error: invalid dataset name [{}]'.format(dataset))

    return train_dataset, valid_dataset

def define_transform(parameters):
    # normal watermark (w. blend-factor)
    if parameters['trans']['transform'] == 'watermark':
        transform = WaterMarker(parameters['trans']['wmark-file'], \
                                parameters['trans']['blend-factor'], \
                                parameters['trans']['noise-level'], \
                                noise=parameters['trans']['noise'])

    # watermarks in the CVPR'17 paper "https://ieeexplore.ieee.org/document/8100209/"
    # - translation: move the watermark to the random position
    elif parameters['trans']['transform'] == 'watermark_translate':
        transform = WMarkerTranslate(parameters['trans']['wmark-file'], \
                                     parameters['trans']['blend-factor'], \
                                     parameters['trans']['wmark-dispose'], \
                                     parameters['trans']['noise-level'], \
                                     noise=parameters['trans']['noise'])
    # - opacity variation: use random opacity
    elif parameters['trans']['transform'] == 'watermark_opavar':
        transform = WMarkerOpacityVariation(parameters['trans']['wmark-file'], \
                                            parameters['trans']['blend-factor'], \
                                            parameters['trans']['wmark-shift'], \
                                            parameters['trans']['noise-level'], \
                                            noise=parameters['trans']['noise'])
    # - spatial perturbations: use random spatial perturbations
    elif parameters['trans']['transform'] == 'watermark_spaper':
        transform = WMarkerSpatialPerturbation(parameters['trans']['wmark-file'], \
                                               parameters['trans']['blend-factor'], \
                                               parameters['trans']['wmark-displace'], \
                                               parameters['trans']['noise-level'], \
                                               noise=parameters['trans']['noise'])

    # undefined transform
    else:
        assert False, ("Error: invalid transform - {}".format(parameters['trans']['transform']))
    return transform


def run_transforms(parameters):

    # init. task type
    task_name = 'transform'

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

    train_dataset, valid_dataset = define_dataset( \
        parameters['trans']['dataset'], parameters, kwargs)


    # initialize transform
    transform = define_transform(parameters)

    # extract the watermark filename
    wmark_fname = extract_wmark_filename(parameters['trans']['wmark-file'])
    if parameters['trans']['transform'] == 'resize': wmark_fname = None

    # intialize output dir.
    store_prefix = define_store_prefix(wmark_fname, parameters)
    if parameters['trans']['dataset'] == 'custom':
        store_path   = os.path.join('datasyns', parameters['trans']['dataset'], \
                                    parameters['trans']['datapath'], task_name, \
                                    parameters['trans']['transform'], store_prefix)
    else:
        store_path   = os.path.join('datasyns', parameters['trans']['dataset'], \
                                    task_name, parameters['trans']['transform'], \
                                    store_prefix)
    if not os.path.isdir(store_path): os.makedirs(store_path)


    ############
    # Perform transformation on training data
    ############
    train_trans_data  = []
    train_trans_label = []
    train_batch_size  = parameters['trans']['store-batch']
    total_data_count  = 0
    for idx, (data, label) in tqdm(enumerate(train_dataset), total=len(train_dataset)):

        # : transform data
        trans_data = transform.do_transform(data)
        train_trans_data.append(np.asarray(trans_data))
        train_trans_label.append(label)

        # : store each batch
        if (idx + 1) % parameters['trans']['store-batch'] == 0:

            # convert the dataset shape (transformed)
            train_trans_data = np.asarray(train_trans_data)             # HWC
            train_trans_data = train_trans_data.transpose((0, 3, 1, 2)) # HWC -> CHW
            train_trans = {
                    'data'  : train_trans_data,
                    'labels': train_trans_label,
                }

            # store to output location
            tdata_filename = 'train_{}_{}.pkl'.format( \
                    parameters['trans']['transform'], int((idx+1)/train_batch_size))
            tdata_savepath = os.path.join(store_path, tdata_filename)
            pickle.dump(train_trans, open(tdata_savepath, 'wb'))

            # re-initialize the data holder
            train_trans_data  = []
            train_trans_label = []

        # count the data size
        total_data_count += 1
    # for idx...

    # remainders
    if train_trans_data and train_trans_label:
        # convert the dataset shape (transformed)
        train_trans_data = np.asarray(train_trans_data)             # HWC
        train_trans_data = train_trans_data.transpose((0, 3, 1, 2)) # HWC -> CHW
        train_trans = {
                'data'  : train_trans_data,
                'labels': train_trans_label,
            }
        # store to output location
        tdata_filename = 'train_{}_{}.pkl'.format( \
                parameters['trans']['transform'], int(total_data_count/train_batch_size)+1)
        tdata_savepath = os.path.join(store_path, tdata_filename)
        pickle.dump(train_trans, open(tdata_savepath, 'wb'))
    # end if train...


    ############
    # Perform transformation on validation data
    ############
    valid_trans_data  = []
    valid_trans_label = []
    valid_batch_size  = parameters['trans']['store-batch']
    total_data_count  = 0
    for idx, (data, label) in tqdm(enumerate(valid_dataset), total=len(valid_dataset)):

        # : transform data
        trans_data = transform.do_transform(data)
        valid_trans_data.append(np.asarray(trans_data))
        valid_trans_label.append(label)

        # : store each batch
        if (idx + 1) % valid_batch_size == 0:

            # convert the dataset shape (transformed)
            valid_trans_data = np.asarray(valid_trans_data)             # HWC
            valid_trans_data = valid_trans_data.transpose((0, 3, 1, 2)) # HWC -> CHW
            valid_trans = {
                    'data'  : valid_trans_data,
                    'labels': valid_trans_label,
                }

            # store to output location
            vdata_filename = 'valid_{}_{}.pkl'.format( \
                    parameters['trans']['transform'], int((idx+1)/valid_batch_size))
            vdata_savepath = os.path.join(store_path, vdata_filename)
            pickle.dump(valid_trans, open(vdata_savepath, 'wb'))

            # re-initialize the data holder
            valid_trans_data  = []
            valid_trans_label = []

        # count the data size
        total_data_count += 1
    # for idx...

    # remainders
    if valid_trans_data and valid_trans_label:
        # convert the dataset shape (transformed)
        valid_trans_data = np.asarray(valid_trans_data)             # HWC
        valid_trans_data = valid_trans_data.transpose((0, 3, 1, 2)) # HWC -> CHW
        valid_trans = {
                'data'  : valid_trans_data,
                'labels': valid_trans_label,
            }

        # store to output location
        vdata_filename = 'valid_{}_{}.pkl'.format( \
                parameters['trans']['transform'], int(total_data_count/valid_batch_size)+1)
        vdata_savepath = os.path.join(store_path, vdata_filename)
        pickle.dump(valid_trans, open(vdata_savepath, 'wb'))
    # end if train...


###########################
# Misc. functions
###########################
def extract_wmark_filename(filepath):
    path_tokens = filepath.split('/')
    return path_tokens[len(path_tokens)-1]

def define_store_prefix(wmark_fname, parameters):
    if parameters['trans']['transform'] == 'watermark':
        prefix = '{}_{}_{}'.format(wmark_fname, \
                                   parameters['trans']['blend-factor'], \
                                   parameters['trans']['noise-level'])

    # translations
    elif parameters['trans']['transform'] == 'watermark_translate':
        prefix = '{}_{}_{}_{}'.format(wmark_fname, \
                                      parameters['trans']['blend-factor'], \
                                      parameters['trans']['wmark-dispose'], \
                                      parameters['trans']['noise-level'])

    # opacity variations
    elif parameters['trans']['transform'] == 'watermark_opavar':
        prefix = '{}_{}_{}_{}'.format(wmark_fname, \
                                      parameters['trans']['blend-factor'], \
                                      parameters['trans']['wmark-shift'], \
                                      parameters['trans']['noise-level'])

    # spatial perturbations: use random spatial perturbations
    elif parameters['trans']['transform'] == 'watermark_spaper':
        prefix = '{}_{}_{}_{}'.format(wmark_fname, \
                                      parameters['trans']['blend-factor'], \
                                      parameters['trans']['wmark-displace'], \
                                      parameters['trans']['noise-level'])

    # resize images
    elif parameters['trans']['transform'] == 'resize':
        prefix = '{}x{}'.format(parameters['trans']['resize-w'], \
                                parameters['trans']['resize-h'])

    # pad images
    elif parameters['trans']['transform'] == 'pad':
        prefix = '{}x{}'.format(parameters['trans']['target-w'], \
                                parameters['trans']['target-h'])

    # undefined transform
    else:
        assert False, ("Error: invalid transform - {}".format(parameters['trans']['transform']))
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
    # load the transform parameters
    # ... base
    parameters['trans'] = {}
    parameters['trans']['dataset'] = arguments.dataset
    parameters['trans']['datapath'] = arguments.datapath
    parameters['trans']['transform'] = arguments.transform
    parameters['trans']['wmark-file'] = arguments.wmark_file
    parameters['trans']['noise'] = argument.noise
    parameters['trans']['noise-level'] = argument.noise_level
    # ... transform specific
    parameters['trans']['blend-factor'] = arguments.blend_factor
    parameters['trans']['wmark-dispose'] = arguments.wmark_dispose
    parameters['trans']['wmark-shift'] = arguments.wmark_shift
    parameters['trans']['wmark-displace'] = arguments.wmark_displace
    parameters['trans']['resize-w'] = arguments.resize_w
    parameters['trans']['resize-h'] = arguments.resize_h
    parameters['trans']['target-w'] = arguments.target_w
    parameters['trans']['target-h'] = arguments.target_h
    parameters['trans']['store-batch'] = arguments.store_batch
    # print out
    print(json.dumps(parameters, indent=2))
    return parameters

# cmdline interface (for backward compatibility)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform a Dataset')

    # system parameters
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='number of workers (default: 8)')
    parser.add_argument('--pin-memory', action='store_false', default=True,
                        help='the data loader copies tensors into CUDA pinned memory')
    # trans parameters
    # ... base
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset used to train (default: cifar10).')
    parser.add_argument('--datapath', type=str, default='',
                        help='dataset stored location (\'custom\' dataset case)')
    parser.add_argument('--transform', type=str, default='watermark',
                        help='transformation name (default: watermark).')
    parser.add_argument('--wmark-file', type=str, default='',
                        help='watermark image file path.')
    parser.add_argument('--noise', action='store_true', default=False,
                        help='noise to watermark (default: false).')
    parser.add_argument('--noise-level', type=float, default=1.0,
                        help='noise level applied to watermark images (default: 1.0).')
    # ... transform specific
    parser.add_argument('--blend-factor', type=float, default=1.0,
                        help='alpha value for the watermark (default: 1.0).')
    parser.add_argument('--wmark-dispose', type=int, default=16,
                        help='maximum displacement of the watermark (default: 16).')
    parser.add_argument('--wmark-shift', type=int, default=20,
                        help='maximum shift that can be introduced in \'c\' (default: 20).')
    parser.add_argument('--wmark-displace', type=float, default=1.0,
                        help='maximum displacement of each pixel in the watermark (default: 1.0).')
    parser.add_argument('--resize-w', type=int, default=256,
                        help='width of the resized images (default: 256).')
    parser.add_argument('--resize-h', type=int, default=256,
                        help='height of the resized images (default: 256).')
    parser.add_argument('--target-w', type=int, default=512,
                        help='width of the target images (default: 512).')
    parser.add_argument('--target-h', type=int, default=512,
                        help='height of the target images (default: 512).')
    parser.add_argument('--store-batch', type=int, default='10000',
                        help='batch size to store transformed data (default: 10000).')

    # execution parameters
    args = parser.parse_args()

    # dump the input parameters
    parameters = dump_arguments(args)
    run_transforms(parameters)
