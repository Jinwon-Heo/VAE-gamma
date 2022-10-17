import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import Parameter
import math
import numpy as np
import collections
from sklearn import metrics
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment as linear_assignment
from torchvision import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import scipy.io as scio
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.model_selection import KFold
import torch.utils.data as data
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import argparse

from VAE_gamma import BasicDataset, generate_gamma,generate_data, cluster_acc, EarlyStopping, VAE_gamma, pretrain_fit, train_model, cv_result

if __name__ == "__main__":
    
    # setting the hyper parameters
    
    parser = argparse.ArgumentParser(description = 'train',
                                    formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--path_name', default='/home/jupyter-heojw/Data_all/VAE-gamma/weight/simulation/')
    parser.add_argument('--device', default=torch.device('cuda:5'))
    parser.add_argument('--monte_number', default=1, type=int)
    parser.add_argument('--n', default=100, type=int)
    parser.add_argument('--data_dim', default=10, type=int)
    parser.add_argument('--num_cluster', default=2, type=int) 
    parser.add_argument('--latent_dim', default=2, type=int)     
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--encodeLayer', default=[200,100,30])    
    parser.add_argument('--decodeLayer', default=[30,100,200])  
    parser.add_argument('--param_set_1', default=[5,1,10])    
    parser.add_argument('--param_set_2', default=[15,2,10])    
    parser.add_argument('--delta', default=0.1, type=float)    
    

    arg = parser.parse_args()

    
    data_all = generate_data(arg.n,dim=arg.data_dim, set_1 = [5, 1, 10], set_2 = [15, 2, 10], delta=arg.delta)

    label_1 = torch.zeros(arg.n).int()
    label_2 = torch.ones(arg.n).int()

    label_all = torch.cat([label_1,label_2],dim=0)

    train_dataset = BasicDataset(data_all, label_all)
    test_dataset = BasicDataset(data_all, label_all)

    arg.batch_size = int(arg.n/10)


    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=arg.batch_size, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    
    n_pre=5
    
    
    final_list = {}
    for i in range(1, n_pre+1):      

        if torch.cuda.is_available():
            pre_model =  VAE_gamma(data_dim=arg.data_dim, latent_dim=arg.latent_dim, device=arg.device, encodeLayer=arg.encodeLayer,
                                   decodeLayer=arg.decodeLayer, num_cluster=arg.num_cluster).cuda(arg.device)

        optimizer_auto = optim.Adam(pre_model.parameters(), lr=0.001) 
        scheduler = StepLR(optimizer_auto, step_size=20, gamma= 0.9)
        path_name = arg.path_name + 'pretrain_' + 'initial_' + str(i)+'.pth'
        pretrain_epoch= 20
        
        pretrain_fit(dataloader, pre_model, optimizer_auto,scheduler,path_name,pretrain_epoch,arg.monte_number,2,device=torch.device('cuda:5'))

        if torch.cuda.is_available():
            model_train = VAE_gamma(data_dim=arg.data_dim, latent_dim=arg.latent_dim, device=arg.device, encodeLayer=arg.encodeLayer,
                                   decodeLayer=arg.decodeLayer, num_cluster=arg.num_cluster).cuda(arg.device)   
        saveWeightPath = path_name
        optimizer = optim.Adam(model_train.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=20, gamma= 0.9)
        model_train.load_state_dict(torch.load(saveWeightPath))
        epoch_train= 10
        
        path_name_2 = arg.path_name + 'train' + '_initial' + str(i) +  '.pth'

        loss = train_model(dataloader, dataloader_test, model_train, optimizer,scheduler,epoch_train,arg.batch_size,arg.monte_number,10,arg.device,path_name_2)
        final_list[' initial:'+str(i) ] = loss
        
    final_loss = np.array(list(final_list.items()))[:,1].astype('float')
    
    optim_model = np.where(final_loss==final_loss.min())[0][0] + 1 
    
    path_name_best =  arg.path_name + 'train' + '_initial' + str(optim_model) +  '.pth'
    
    if torch.cuda.is_available():
        model =  VAE_gamma(data_dim=arg.data_dim, latent_dim=arg.latent_dim, device=arg.device, encodeLayer=arg.encodeLayer,
                                   decodeLayer=arg.decodeLayer, num_cluster=arg.num_cluster).cuda(arg.device)   

    saveWeightPath = path_name_best

    model.load_state_dict(torch.load(saveWeightPath))
    print('\n ************************ sample: '+ str(arg.n),'delta: '+ str(arg.delta)+ '************************')    
    model.eval()
    accuracy_list = []
    test_loss_list = []    
    test_loss_sum = 0.0
    for perform in range(0,100):
        test_loss = 0.0
        test_running_correct = 0.0
        label_set_test = torch.tensor([],dtype=torch.int32).cuda(arg.device)
        preds_set_test = torch.tensor([],dtype=torch.int32).cuda(arg.device)
        for batch_idx, data in enumerate(dataloader_test):

            img, label = data 
            img = img.float()
            img = img.view(img.size(0), -1)

            if torch.cuda.is_available():
                img = img.cuda(arg.device)
                label = label.cuda(arg.device).int()

            recon_al,recon_be, logalpha, logbeta, z = model(img)
            monte_gamma = 0
            monte_recon_loss = 0
            for l in range(arg.monte_number):
                z_l = model.reparametrize(logalpha, logbeta)
                m = nn.Softplus()
                recon_al_l = 1e-6 + m(model.decode(z_l)[:,:model.data_dim])
                recon_be_l = 1e-6 + m(model.decode(z_l)[:,model.data_dim:])
                monte_gamma += model.get_gamma(z_l,len(dataloader_test.dataset))
                monte_recon_loss += model.recon_loss(recon_al_l,recon_be_l, img)
            sgvb_gamma_test = monte_gamma/arg.monte_number
            sgvb_loss_test = torch.mean(monte_recon_loss/arg.monte_number 
                                        + model.kl_loss(sgvb_gamma_test, logalpha, logbeta,len(dataloader_test.dataset))) #+ 1*model.Loss_b(img,z, logalpha, logbeta, 2, latent_dim))

            _, preds = torch.max(sgvb_gamma_test,1)
            preds = preds.type(torch.int32)

            label_set_test = torch.cat([label_set_test,label]).int()
            preds_set_test = torch.cat([preds_set_test,preds]).int()
            test_loss += sgvb_loss_test.item() * img.size(0)


        label_set_test = label_set_test.cpu()
        preds_set_test = preds_set_test.cpu()

        test_acc = cluster_acc(preds_set_test,label_set_test) * 100/preds_set_test.shape[0]
        test_loss_sum += test_loss
        accuracy_list = np.append(accuracy_list, test_acc)

    test_loss_list = np.append(test_loss_list, test_loss_sum/100)


    print('***** Test Result: loss: {:.3f} acc: {:.4f}'
      .format( test_loss_list[0] / len(dataloader_test.dataset), np.max(accuracy_list)))