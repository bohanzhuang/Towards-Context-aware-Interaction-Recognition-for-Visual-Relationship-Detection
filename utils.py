import numpy as np
import yaml
import glob
import scipy.io as sio
import h5py
import torch
import os
from torch.autograd import Variable


def unpack_configs(config):
    train_folder = config['train_folder']
    train_filenames = sorted(glob.glob(train_folder + '/*.mat'))
    val_folder = config['val_folder']
    val_filenames = sorted(glob.glob(val_folder + '/*.mat'))
    return train_filenames, val_filenames


def unpack_test_configs(config):
    test_folder = config['test_folder']
    language_folder = config['test_language_dir']
    test_filenames = sorted(glob.glob(test_folder + '/*.mat'))
#    language = sio.loadmat(language_folder)
#    language = language['train_language_feature']
    language_features = sorted(glob.glob(language_folder + '/*.mat'))

    return test_filenames, language_features

def adjust_learning_rate(config, epoch, step_idx, learning_rate):

    if epoch == config['lr_step'][step_idx]:
        learning_rate = learning_rate / 10.0
        step_idx += 1
    return step_idx, learning_rate



def proc_configs(config):

    if not os.path.exists(config['weights_dir']):
        os.makedirs(config['weights_dir'])
        print "Creat folder: " + config['weights_dir']

    return config


def test(CNN, test_labels, test_filenames, language_features):   

    CNN.eval()
    total = 0
    correct = 0

    for index in xrange(len(test_filenames)):
        data = h5py.File(test_filenames[index])
        batch_feature = np.asarray(data['predicate_activation'])
        batch_feature = batch_feature[np.newaxis, :,:,:]

        labels = test_labels[index].astype('int32')
        language_feature = language_features[index,:].astype('float32')
        language_feature = np.reshape(language_feature, (1, 600))

        batch_feature = Variable(torch.from_numpy(batch_feature)).cuda()
        language_feature = Variable(torch.from_numpy(language_feature)).cuda()
        labels = Variable(torch.from_numpy(labels).type(torch.LongTensor)).cuda()

        outputs = CNN(batch_feature, language_feature) 
        pred = outputs.data.max(1)[1] # get the index of the max log-probability
        total += labels.size(0)
        correct += pred.eq(labels.data).cpu().sum()

    print('Test Accuracy of the model: %f %%' % (100.0 * correct / total))
    return 100.0 * correct / total
