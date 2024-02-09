import torch
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

#from architecture import INN as inn
from functionalities import dataloader as dl
#from functionalities import evaluater as ev
#from functionalities import loss as lo
from functionalities import trainer as tr

from tqdm import tqdm_notebook as tqdm
from functionalities import tracker as tk
from functionalities import MMD_autoencoder_loss as cl
from functionalities import filemanager as fm
from functionalities import plot as pl
from functionalities import gpu

# following are few utility functions from FrEIA framework
# Ref: https://github.com/vislearn/FrEIA
from FrEIA import framework as fr
from FrEIA.modules import coeff_functs as fu
from FrEIA.modules import coupling_layers as la
from FrEIA.modules import reshapes as re




def train(num_epoch, model, modelname, criterion, optimizer, scheduler, latent_dim, trainloader, validloader=None,
          testloader=None, conditional=False, disc_lst=None, use_label=False, tracker=None, device='cpu', save_model=True, save_variable=True, subdir=None, num_epoch_save=10,
          num_img=100, grid_row_size=10):
    """
    Train a INN model.
    :param num_epoch: number of training epochs
    :param model: INN that should be trained
    :param modelname: model name under which the model should be saved
    :param criterion: the criterion to compute the loss
    :param optimizer: the optimization method used for training
    :param scheduler: pytorch scheduler for adaptive learning
    :param latent_dim: dimension of the latent space
    :param trainloader: the training set wrapped by a loader
    :param validloader: the validation set wrapped by a loader
    :param testloader: the test set wrapped by a loader
    :param disc_lst: If given the first latent dimension will be enforced to be discrete depending on the values given
    in disc_lst
    :param use_label: If true, the labels will be used to help enforcing the first latent dimension to be discrete
    :param tracker: tracker for values during training
    :param device: device on which to do the computation (CPU or CUDA). Please use get_device() function to get the
    device, if using multiple GPU's. Default: cpu
    :param save_model: If True save model and model weights. Default: True
    :param save_variable: If True save all loss histories. Default: True
    :param subdir: subdirectory to save the model in
    :param num_epoch_save: number of epochs after which a sample of reconstructed images will be saved
    :return: model (trained model)
    """
    print("TRAIN")
    print("conditonal=", conditional)
    model.to(device)

    tot_loss_log = []
    tot_valid_loss_log = []
    tot_test_loss_log = []
    rec_loss_log = []
    rec_valid_loss_log = []
    rec_test_loss_log = []
    dist_loss_log = []
    dist_valid_loss_log = []
    dist_test_loss_log = []
    spar_loss_log = []
    spar_valid_loss_log = []
    spar_test_loss_log = []
    disen_loss_log = []
    disen_valid_loss_log = []
    disen_test_loss_log =[]
    disc_loss_log = []
    disc_valid_loss_log = []
    disc_test_loss_log = []
    min_loss = 10e300
    best_epoch = 0
    num_step = 0

    for epoch in range(num_epoch):
        model.train()

        if disc_lst is not None or conditional:
            losses = np.zeros(6, dtype=np.double)
        else:
            losses = np.zeros(5, dtype=np.double)
            losses_list = [[],[],[],[],[]]

        print('length of losses:', len(losses))

        scheduler.step()

        print('Epoch: {}'.format(epoch + 1))
        print('Training:')

        correct = 0

        for i, data in enumerate(tqdm(trainloader), 0):
            criterion.update_num_step(num_step)
            num_step += 1
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            #print("inputs.shape: ", inputs.shape) # inputs.shape:  torch.Size([128, 1, 28, 28])

            optimizer.zero_grad()

            lat_img = model(inputs)
            lat_shape = lat_img.shape
            #print("lat_image: ", lat_img)
            #print("lat_image.shape: ", lat_img.shape) # lat_image.shape:  torch.Size([128, 1, 28, 28])

            # This is needed to crete artificial bottkeneck - part of the vector after latnet_dim will be filled with zeros
            lat_img = lat_img.view(lat_img.size(0), -1)
            #print("lat_image_view: ", lat_img)
            #print("lat_image_view.shape: ", lat_img.shape) # lat_image_view.shape:  torch.Size([128, 784])

            if conditional:
                binary_label = lat_img.new_zeros(lat_img.size(0), 10)
                idx = torch.arange(labels.size(0), dtype=torch.long)
                binary_label[idx, labels] = 1
                lat_img_mod = torch.cat([lat_img[:, :latent_dim], binary_label, lat_img.new_zeros((lat_img[:, latent_dim+10:]).shape)], dim=1)
                pred = lat_img[:, latent_dim:latent_dim+10].max(1, keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum().item()
            elif use_label and disc_lst is not None:
                disc_lst = torch.tensor(disc_lst).to(device).float()
                disc_lat_dim = disc_lst[labels]
                lat_img_mod = torch.cat([torch.unsqueeze(disc_lat_dim, 1).float(), lat_img[:, 1:latent_dim],
                                         lat_img.new_zeros((lat_img[:, latent_dim:]).shape).float()], dim=1)
                pred = disc_lst[torch.min(torch.abs(lat_img[:, :1] - disc_lst), 1)[1]] * 10
                correct += pred.eq(labels.float().view_as(pred)).sum().item()
            elif disc_lst is not None:
                disc_lst = torch.tensor(disc_lst).to(device)
                disc_lat_idx = torch.min(torch.abs(lat_img[:,:1] - disc_lst), 1)[1]
                disc_lat_dim = disc_lst[disc_lat_idx]
                lat_img_mod = torch.cat([torch.unsqueeze(disc_lat_dim, 1), lat_img[:, 1:latent_dim],
                                         lat_img.new_zeros((lat_img[:, latent_dim:]).shape)], dim=1)
            else:
                # this is what gets executed
                lat_img_mod = torch.cat([lat_img[:, :latent_dim], lat_img.new_zeros((lat_img[:, latent_dim:]).shape)], dim=1)


            lat_img_mod = lat_img_mod.view(lat_shape)
            #print ("lat_img_mode.shape: ", lat_img_mod.shape) # lat_img_mod.shape:  torch.Size([128, 1, 28, 28])
            #print ("lat_img_mod: ", lat_img_mod)

            # lat_img_mod:  tensor([[[[-0.0097, -0.0754,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
            #                    [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],



            output = model(lat_img_mod, rev=True)


            if conditional:
                batch_loss = criterion(inputs, lat_img, output, labels, binary_label)
            elif use_label:
                batch_loss = criterion(inputs, lat_img, output, labels)
            else:
                batch_loss = criterion(inputs, lat_img, output)
                #print("batch_loss:", batch_loss) #new

            batch_loss[0].backward()

            optimizer.step()

            for i in range(len(batch_loss)):
                losses[i] += batch_loss[i].item()
                losses_list[i].append(batch_loss[i].item())
            # print("losses_list:", losses_list)  #new

        correct = correct * 100. / len(trainloader.dataset)
        losses /= len(trainloader)
        tot_loss_log.append(losses[0])
        rec_loss_log.append(losses[1])
        dist_loss_log.append(losses[2])
        spar_loss_log.append(losses[3])
        disen_loss_log.append(losses[4])
        if len(losses) == 6:
            disc_loss_log.append(losses[5])
            print('Loss: {:.3f} \t L_rec: {:.3f} \t L_dist: {:.3f} \t L_spar: {:.3f} \t L_disen: {:.3f} \t L_disc: {:.3f}'.format(
                losses[0], losses[1], losses[2], losses[3], losses[4], losses[5]))
            print('Train Accuracy: {:.1f}'.format(correct))
        else:
            print('Loss: {:.3f} \t L_rec: {:.3f} \t L_dist: {:.3f} \t L_spar: {:.3f} \t L_disen: {:.3f}'.format(
                losses[0], losses[1], losses[2], losses[3], losses[4]))

        if validloader is not None:
            print('\n')
            print('Compute and record loss on validation set')
            #valid_loss = ev.get_loss(validloader, model, criterion, latent_dim, tracker, conditional, disc_lst, use_label, device)
            valid_loss = get_loss(validloader, model, criterion, latent_dim, tracker, conditional, disc_lst, use_label, device)
            tot_valid_loss_log.append(valid_loss[0])
            rec_valid_loss_log.append(valid_loss[1])
            dist_valid_loss_log.append(valid_loss[2])
            spar_valid_loss_log.append(valid_loss[3])
            disen_valid_loss_log.append(valid_loss[4])
            if len(valid_loss) == 6:
                disc_valid_loss_log.append(valid_loss[5])
                print('Loss: {:.3f} \t L_rec: {:.3f} \t L_dist: {:.3f} \t L_spar: {:.3f} \t L_disen: {:.3f} \t L_disc: {:.3f}'.format(
                    valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5]))
            else:
                print('Loss: {:.3f} \t L_rec: {:.3f} \t L_dist: {:.3f} \t L_spar: {:.3f} \t L_disen: {:.3f}'.format(
                    valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4]))

            print('latent image mean: {:.3f} \t latent image std: {:.3f}'.format(tracker.mu, tracker.std))

            if valid_loss[0] <= min_loss:
                last_best_epoch = best_epoch
                best_epoch = epoch + 1
                min_loss = valid_loss[0]
                fm.save_model(model, "{}_{}_best".format(modelname, best_epoch))
                fm.save_weight(model, "{}_{}_best".format(modelname, best_epoch))
                if last_best_epoch != 0:
                    fm.delete_file("models", "{}_{}_best".format(modelname, last_best_epoch))
                    fm.delete_file("weights", "{}_{}_best".format(modelname, last_best_epoch))

        if testloader is not None:
            print('\n')
            print('Compute and record loss on test set:')
            #test_loss = ev.get_loss(testloader, model, criterion, latent_dim, tracker, conditional, disc_lst, use_label, device)
            test_loss = get_loss(testloader, model, criterion, latent_dim, tracker, conditional, disc_lst, use_label, device)
            tot_test_loss_log.append(test_loss[0])
            rec_test_loss_log.append(test_loss[1])
            dist_test_loss_log.append(test_loss[2])
            spar_test_loss_log.append(test_loss[3])
            disen_test_loss_log.append(test_loss[4])
            if len(test_loss) == 6:
                disc_test_loss_log.append(test_loss[5])
                print('Loss: {:.3f} \t L_rec: {:.3f} \t L_dist: {:.3f} \t L_spar: {:.3f} \t L_disen: {:.3f} \t L_disc: {:.3f}'.format(
                    test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4], test_loss[5]))
            else:
                print('Loss: {:.3f} \t L_rec: {:.3f} \t L_dist: {:.3f} \t L_spar: {:.3f} \t L_disen: {:.3f}'.format(
                    test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4]))

            print('latent image mean: {:.3f} \t latent image std: {:.3f}'.format(tracker.mu, tracker.std))

        if epoch % num_epoch_save == 0 or epoch == (num_epoch - 1):
            pl.plot_diff(model, testloader, latent_dim, device, num_img, grid_row_size, filename=modelname + "_{}".format(epoch))

        print('\n')
        print('-' * 80)
        print('\n')

    if validloader is not None:
        print("Lowest Validation Loss: {:3f} was achieved at epoch: {}".format(min_loss, best_epoch))

    print("Finished Training")

    if subdir is None:
        subdir = modelname

    if save_model:
        model.to('cpu')
        fm.save_model(model, "{}_{}".format(modelname, num_epoch), subdir)
        fm.save_weight(model, "{}_{}".format(modelname, num_epoch), subdir)

    if save_variable:
        fm.save_variable([tot_loss_log, tot_valid_loss_log, tot_test_loss_log],
                         "total_loss_{}_{}".format(modelname, num_epoch), subdir)
        fm.save_variable([rec_loss_log, rec_valid_loss_log, rec_test_loss_log],
                         "reconstruction_loss_{}_{}".format(modelname, num_epoch), subdir)
        fm.save_variable([dist_loss_log, dist_valid_loss_log, dist_test_loss_log],
                         "distribution_loss_{}_{}".format(modelname, num_epoch), subdir)
        fm.save_variable([spar_loss_log, spar_valid_loss_log, spar_test_loss_log],
                         "sparsity_loss_{}_{}".format(modelname, num_epoch), subdir)
        fm.save_variable([disen_loss_log, disen_valid_loss_log, disen_test_loss_log],
                         "disentanglement_loss_{}_{}".format(modelname, num_epoch), subdir)
        fm.save_variable([disc_loss_log, disc_valid_loss_log, disc_test_loss_log],
                         "discrete_loss_{}_{}".format(modelname, num_epoch), subdir)
        fm.save_variable([tot_loss_log, rec_loss_log, dist_loss_log, spar_loss_log, disen_loss_log, disc_loss_log],
                         "train_loss_{}_{}".format(modelname, num_epoch), subdir)
        fm.save_variable([tot_valid_loss_log, rec_valid_loss_log, dist_valid_loss_log, spar_valid_loss_log,
                          disen_valid_loss_log, disc_loss_log], "validation_loss_{}_{}".format(modelname, num_epoch), subdir)
        fm.save_variable([tot_test_loss_log, rec_test_loss_log, dist_test_loss_log, spar_test_loss_log,
                          disen_test_loss_log, disc_loss_log], "test_loss_{}_{}".format(modelname, num_epoch), subdir)

    return model

def init_model(get_model, latent_dim, loss_type, device, a_distr=1, a_rec=1, a_spar=1, a_disen=1, a_disc=0, conditional=False, disc_lst=None, use_lat_dim=False, cont_min=None, cont_max=None, num_iter=None, init_weight=True):
    """
    Initialize the INN model.
    :param get_model: function that returns the INN that should be trained
    :param latent_dim: dimension of the latent space
    :param loss_type: type of reconstruction loss to use
    :param device: device on which to do the computation (CPU or CUDA). Please use get_device() function to get the
    device, if using multiple GPU's. Default: cpu
    :param a_distr: factor for distribution loss
    :param a_rec: factor for reconstruction loss
    :param a_spar: factor for sparsity loss
    :param a_disen: factor for disentanglement
    :param use_lat_dim: get_model needs latent dimension as argument
    :return: model: Initialized model
             model_params: parameters of the model
             track: tracker for values during training
             loss: class to compute the total loss
    """

    if use_lat_dim:
        model = get_model(latent_dim)
    else:
        model = get_model()

    if init_weight:
        init_param(model)

    model.train()

    model.to(device)

    if disc_lst is not None:
        disc_lst = torch.tensor(disc_lst).to(device)

    model_params = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            model_params.append(parameter)

    track = tk.tracker(latent_dim)

    loss = cl.MMD_autoencoder_loss(a_distr=a_distr, a_rec=a_rec, a_spar=a_spar, a_disen=a_disen, a_disc=a_disc, latent_dim=latent_dim, loss_type=loss_type, device=device, conditional=conditional, disc_lst=disc_lst, cont_min=cont_min, cont_max=cont_max, num_iter=num_iter)

    return model, model_params, track, loss


def init_training(model_params, lr_init, l2_reg, milestones):
    """
    Initialize optimizer and scheduler for training.
    :param model_params: parameters of the model
    :param lr_init: initial learning rate
    :param l2_reg: weight decay for Adam
    :param milestones: list of training epochs in which to reduce the learning rate
    :return: optimizer, scheduler
    """

    optimizer = torch.optim.Adam(model_params, lr=lr_init, betas=(0.8, 0.8), eps=1e-04, weight_decay=l2_reg)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    return optimizer, scheduler


def init_param(mod, sigma = 0.1):
    """
    Weight intialization for INN models.
    """
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = sigma * torch.randn(param.data.shape) #.cuda()
            if split[3][-1] == '3': # last convolution in the coeff func
                param.data.fill_(0.)


def get_loss(loader, model, criterion, latent_dim, tracker, conditional=False, disc_lst=None, use_label=False, device='cpu'):

    #train_losses = get_loss(trainloader, model, criterion, latent_dim, track, device)
    """
    Compute the loss of a model on a train, test or evalutation set wrapped by a loader.
    :param loader: loader that wraps the train, test or evaluation set
    :param model: model that should be tested
    :param criterion: the criterion to compute the loss
    :param latent_dim: dimension of the latent space
    :param tracker: tracker for values during training
    :param disc_lst: If given the first latent dimension will be enforced to be discrete depending on the values given
    in disc_lst
    :param use_label: If true, the labels will be used to help enforcing the first latent dimension to be discrete
    :param device: device on which to do the computation (CPU or CUDA). Please use get_device() function to get the
    device, if using multiple GPU's. Default: cpu
    :return: losses
    """
    print("EVAL")
    print("conditonal=", conditional)
    print("device=", device)
    model.to(device)

    model.eval()

    if disc_lst is not None or conditional:
        losses = np.zeros(6, dtype=np.double)
    else:
        losses = np.zeros(5, dtype=np.double)

    tracker.reset()

    correct = 0

    for i, data in enumerate(tqdm(loader), 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            lat_img = model(inputs)
            lat_shape = lat_img.shape
            lat_img = lat_img.view(lat_img.size(0), -1)

            if conditional:
                binary_label = lat_img.new_zeros(lat_img.size(0), 10)
                idx = torch.arange(labels.size(0), dtype=torch.long)
                binary_label[idx, labels] = 1
                lat_img_mod = torch.cat([lat_img[:,:latent_dim], binary_label, lat_img.new_zeros((lat_img[:,latent_dim+10:]).shape)], dim=1)
                pred = lat_img[:, latent_dim:latent_dim+10].max(1, keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum().item()
            elif use_label and disc_lst is not None:
                disc_lst = torch.tensor(disc_lst).to(device)
                disc_lat_dim = disc_lst[labels]
                lat_img_mod = torch.cat([torch.unsqueeze(disc_lat_dim, 1), lat_img[:, 1:latent_dim],
                                         lat_img.new_zeros((lat_img[:, latent_dim:]).shape)], dim=1)
                pred = disc_lst[torch.min(torch.abs(lat_img[:, :1] - disc_lst), 1)[1]] * 10
                correct += pred.eq(labels.float().view_as(pred)).sum().item()
            elif disc_lst is not None:
                disc_lst = torch.tensor(disc_lst).to(device)
                disc_lat_idx = torch.min(torch.abs(lat_img[:, :1] - disc_lst), 1)[1]
                disc_lat_dim = disc_lst[disc_lat_idx]
                lat_img_mod = torch.cat([torch.unsqueeze(disc_lat_dim, 1), lat_img[:, 1:latent_dim],
                                         lat_img.new_zeros((lat_img[:, latent_dim:]).shape)], dim=1)
            else:
                lat_img_mod = torch.cat([lat_img[:, :latent_dim], lat_img.new_zeros((lat_img[:, latent_dim:]).shape)], dim=1)

            lat_img_mod = lat_img_mod.view(lat_shape)

            output = model(lat_img_mod, rev=True)

            if use_label:
                batch_loss = criterion(inputs, lat_img.to(device), output.to(device), labels)
            else:
                batch_loss = criterion(inputs, lat_img.to(device), output.to(device))

            for i in range(len(batch_loss)):
                losses[i] += batch_loss[i].item()

            tracker.update(lat_img)

    correct = correct * 100. / len(loader.dataset)
    if use_label or conditional:
        print('Test Accuracy: {:.1f}'.format(correct))
    losses /= len(loader)
    return losses


def get_loss_bottleneck(loader, modelname, subdir, latent_dim_lst, device, a_distr, a_rec, a_spar, a_disen):
    """
    :return:
    """

    total_loss = []
    rec_loss = []
    dist_loss = []
    spar_loss = []
    disen_loss = []

    for i in latent_dim_lst:
        print('bottleneck dimension: {}'.format(i))
        model = fm.load_model('{}_{}'.format(modelname, i).to(device), subdir)
        criterion = cl.MMD_autoencoder_loss(a_distr=a_distr, a_rec=a_rec, a_spar=a_spar, a_disen=a_disen, latent_dim=i, loss_type='l1', device=device)
        losses = get_loss(loader, model, criterion, i, device)
        total_loss.append(losses[0])
        rec_loss.append(losses[1])
        dist_loss.append(losses[2])
        spar_loss.append(losses[3])
        disen_loss.append(losses[4])

    return total_loss, rec_loss, dist_loss, spar_loss, disen_loss

def train_bottleneck(num_epoch, get_model, loss_type, modelname, milestones, latent_dim_lst, trainloader,
                     validloader=None, testloader=None, a_distr=1, a_rec=1, a_spar=1, a_disen=1, lr_init=1e-3,
                     l2_reg=1e-6, conditional=False, disc_lst=None, use_label=False, device='cpu', save_model=False,
                     save_variable=True, use_lat_dim=False, num_epoch_save=10, num_img=100, grid_row_size=10):
    """
    Train INN model for various bottleneck sizes.
    :param num_epoch: number of training epochs
    :param get_model: function that returns the INN that should be trained
    :param loss_type: type of reconstruction loss to use
    :param modelname: model name under which the model should be saved
    :param milestones: list of training epochs in which to reduce the learning rate
    :param latent_dim_lst: a list of latent space dimensions
    :param trainloader: the training set wrapped by a loader
    :param validloader: the validation set wrapped by a loader
    :param testloader: the test set wrapped by a loader
    :param a_distr: factor for distribution loss (see CIFAR_coder_loss)
    :param a_rec: factor for reconstruction loss (see CIFAR_coder_loss)
    :param a_spar: factor for sparsity loss (see CIFAR_coder_loss)
    :param a_disen: factor for disentanglement loss (see CIFAR_coder_loss)
    :param lr_init: initial learning rate
    :param l2_reg: weight decay for Adam
    :param device: device on which to do the computation (CPU or CUDA). Use get_device() function to get the
    device, if using multiple GPU's. Default: cpu
    :param save_model: If True save model and model weights. Default: True
    :param save_variable: If True save all loss histories. Default: True
    :param use_lat_dim: get_model needs latent dimension as argument
    :param num_epoch_save: number of epochs after which a sample of reconstructed images will be saved
    :return: None
    """
    print("TRAIN BOTTLENECK")

    tot_train_loss_log = []
    rec_train_loss_log = []
    dist_train_loss_log = []
    spar_train_loss_log = []
    disen_train_loss_log = []
    tot_test_loss_log = []
    rec_test_loss_log = []
    dist_test_loss_log = []
    spar_test_loss_log = []
    disen_test_loss_log = []

    for latent_dim in latent_dim_lst:
        print("\n")
        print("Start Training with latent dimension: {}".format(latent_dim))
        print('\n')

        model, model_params, track, criterion = init_model(get_model, latent_dim, loss_type, device, a_distr, a_rec,
                                                           a_spar, a_disen, use_lat_dim)
        print ("after init_model")
        optimizer, scheduler = init_training(model_params, lr_init, l2_reg, milestones)
        print ("after init_training")
        model.to(device)

        model = train(num_epoch, model, modelname + "_{}".format(latent_dim), criterion, optimizer, scheduler,
                      latent_dim, trainloader, validloader, testloader, conditional, disc_lst, use_label, track, device, save_model,
                      save_variable, modelname + "_bottleneck", num_epoch_save, num_img, grid_row_size)

        print("conditonal=", conditional)

        train_losses = [0,0,0,0,0]
        #train_losses = get_loss(trainloader, model, criterion, latent_dim, track, conditional, disc_lst, use_label, device=device)

        tot_train_loss_log.append(train_losses[0])
        rec_train_loss_log.append(train_losses[1])
        dist_train_loss_log.append(train_losses[2])
        spar_train_loss_log.append(train_losses[3])
        disen_train_loss_log.append(train_losses[4])

        losses = [0,0,0,0,0]
        #losses = get_loss(testloader, model, criterion, latent_dim, track, conditional, disc_lst, use_label, device=device)
        #######losses = get_loss(trainloader, model, criterion, latent_dim, track, conditional, disc_lst, use_label, device=device)
        tot_test_loss_log.append(losses[0])
        rec_test_loss_log.append(losses[1])
        dist_test_loss_log.append(losses[2])
        spar_test_loss_log.append(losses[3])
        disen_test_loss_log.append(losses[4])

        print("\n")
        print("#" * 80)
        print("#" * 80)
        print("\n")

    fm.save_variable([tot_test_loss_log, rec_test_loss_log, dist_test_loss_log, spar_test_loss_log,
                      disen_test_loss_log], "bottleneck_test_loss_{}".format(modelname), modelname)

    fm.save_variable([tot_train_loss_log, rec_train_loss_log, dist_train_loss_log, spar_train_loss_log,
                      disen_train_loss_log], "bottleneck_train_loss_{}".format(modelname), modelname)

    return model

