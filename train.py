import torch
from torch.autograd import Variable
from torchvision import transforms, datasets
import torch.nn as nn
import yaml
import time
import numpy as np
import scipy.io as sio
from random import shuffle
import torch.nn.functional as F
import torch.nn.parallel
import h5py
from model import vgg_model
from utils import unpack_configs, test, adjust_learning_rate, proc_configs


def main(config):


    train_filenames, val_filenames = unpack_configs(config)
    train_labels = sio.loadmat(config['train_label_file'])
    train_labels = train_labels['train_predicate_labels']
    val_labels = sio.loadmat(config['val_label_file'])
    val_labels = val_labels['test_predicate_labels']

#   language feature
    language = sio.loadmat(config['train_language_feature'])
    language = language['train_language_feature']


    val_language = sio.loadmat(config['test_language_feature'])
    val_language = val_language['test_language_feature']


    index_shuf = range(len(train_labels))
    shuffle(index_shuf)

    train_filenames_shuf = []
    train_labels_shuf = []
    language_shuf = np.zeros((21066, 600))

    count = 0
    for i in index_shuf:
        train_filenames_shuf.append(train_filenames[i])
        train_labels_shuf.append(train_labels[i])
        language_shuf[count,:] = language[i,:]
        count = count + 1

    batch_size = config['batch_size']
    learning_rate = config['learning_rate']


    print("Compilation complete, starting training...")   
    n_train_batches = int(len(train_filenames) / batch_size)
    minibatch_range = range(n_train_batches)


###---------------load pretrained weights--------------

    model = vgg_model(config)
    model = model.cuda()
    param_values = torch.load('./weights/vgg16.pth').values()
    model_dict = model.state_dict()

    for name, param in zip(model_dict.keys()[0:26], param_values[0:26]):
        model_dict[name] = param
    model.load_state_dict(model_dict)

     # Optimizer

    optimizer = torch.optim.SGD([{'params':model.context_aware_model.parameters()},{'params':model.features.parameters(), 'lr':learning_rate / 10.0}], lr=learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)


    epoch = 0
    step_idx = 0
    save_frequency = 1
    test_record = []


    while epoch < config['n_epochs']:

        if config['shuffle']:
            np.random.shuffle(minibatch_range)

# ---------------------resume training------------
        epoch = epoch + 1    
        if config['resume_train'] and epoch == 1:
            load_epoch = config['load_epoch']
            s_resume_dict = torch.load('./weights/model_' + str(load_epoch) + '.pkl')
            resume_values = s_resume_dict.values()
            resume_dict = model.state_dict()
            for name, param in zip(resume_dict.keys(), resume_values):
                model[name] = param            
            model.load_state_dict(resume_dict)


            epoch = load_epoch + 1
            learning_rate = np.load(
                config['weights_dir'] + 'lr_' + str(load_epoch) + '.npy')
            s_test_record = list(
                np.load(config['weights_dir'] + 's_test_record.npy'))

           #Optimizer
            optimizer = torch.optim.SGD([{'params':model.context_aware_model.parameters()},{'params':model.features.parameters(), 'lr':learning_rate / 10.0}], lr=learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
            
        count = 0
        for minibatch_index in minibatch_range:

            optimizer.zero_grad()

            count = count + 1

            batch_feature = np.zeros((batch_size, 3, 224, 224), 'float32')
            language_feature = language_shuf[minibatch_index*batch_size:(minibatch_index+1)*batch_size, :].astype('float32')
            language_feature = Variable(torch.from_numpy(language_feature)).cuda()

            for index in range(batch_size):
                data = h5py.File(train_filenames_shuf[minibatch_index*batch_size + index])
                data = np.asarray(data['predicate_activation'])
                data = data[np.newaxis,:,:,:]
                batch_feature[index, :, :, :] = data
            batch_label = np.squeeze(train_labels_shuf[minibatch_index*batch_size:(minibatch_index+1)*batch_size])
            batch_label = np.asarray(batch_label).astype('int32')

            batch_feature = Variable(torch.from_numpy(batch_feature)).cuda()
            batch_label = Variable(torch.from_numpy(batch_label).type(torch.LongTensor)).cuda()

            prediction = model(batch_feature, language_feature)
            loss_cls = nn.CrossEntropyLoss()(prediction, batch_label)
            loss_cls.backward()
            optimizer.step()


            if count % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_1: {:.6f}'.format(epoch, count * batch_size, len(train_filenames), 100. * count / n_train_batches, loss_cls.data[0])) 

        test_accuracy = test(model, val_labels, val_filenames, val_language)
        test_record.append([test_accuracy])
        np.save(config['weights_dir'] + 'test_record.npy', test_record)         


        step_idx, learning_rate = adjust_learning_rate(config, epoch, step_idx,
                                           learning_rate)

       #Optimizer
        optimizer = torch.optim.SGD([{'params':model.context_aware_model.parameters()},{'params':model.features.parameters(), 'lr':learning_rate / 10.0}], lr=learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
        print("Learning_rate = "+str(learning_rate))

        if epoch % save_frequency == 0:
            model_file = config['weights_dir'] + 'model_' + str(epoch) + '.pkl'
            torch.save(model.state_dict(), model_file)
            np.save(config['weights_dir'] + 'lr_' + str(epoch) + '.npy',
                       learning_rate)


if __name__ == '__main__':

    with open('./config.yaml', 'r') as f:
        config = yaml.load(f) 
    config = proc_configs(config)      
    main(config)
