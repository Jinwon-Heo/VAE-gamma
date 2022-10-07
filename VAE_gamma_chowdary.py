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

from VAE_gamma import BasicDataset, generate_gamma, cluster_acc, EarlyStopping, VAE_gamma, pretrain_fit, train_model, cv_result

if __name__ == "__main__":
    
    # setting the hyper parameters
    
    parser = argparse.ArgumentParser(description = 'train',
                                    formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--data_file', default='/home/jupyter-heojw/chowdary104.mat')
    parser.add_argument('--label_file', default='/home/jupyter-heojw/class_chowdary.mat')
    parser.add_argument('--path_name', default='/home/jupyter-heojw/pretrain/chowdary')
    parser.add_argument('--device', default=torch.device('cuda:5'))
    parser.add_argument('--monte_number', default=1, type=int)
    parser.add_argument('--data_dim', default=182, type=int)
    parser.add_argument('--num_cluster', default=5, type=int)
    parser.add_argument('--batch_size', default=26, type=int)
    parser.add_argument('--random_state', default=123, type=int)
    parser.add_argument('--n_splits', default=5, type=int)
    parser.add_argument('--encodeLayer', default=[200,100,30])    
    parser.add_argument('--decodeLayer', default=[30,100,200])  
    

    arg = parser.parse_args()
    
    data_x = scio.loadmat(arg.data_file)
    data_y = scio.loadmat(arg.label_file)

    
    chowdary104 = data_x['chowdary']
    label = data_y['class_chowdary'].reshape(104)

    label[label==1] = 0
    label[label==2] = 1
    
    x_data = chowdary104
    y_data = label

    train_dataset = BasicDataset(x_data,y_data)
    test_dataset = BasicDataset(x_data,y_data)


    device = arg.device


    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=arg.batch_size, shuffle=True, drop_last=True)
    dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

    dataset_sizes = len(train_dataset)

    latent_dim_list = [1,2,4,6,8,10,12,14,16,18]

    num_cluster_list = [1,2,3]
    
    n_pre=5
    
    
    from tqdm import tqdm

    dict_list = {}
    monte_number = 1
    for k in tqdm(num_cluster_list):
        for j in tqdm(latent_dim_list):
            model = VAE_gamma(data_dim=arg.data_dim, latent_dim=j, device=arg.device, encodeLayer=arg.encodeLayer, decodeLayer=arg.decodeLayer, num_cluster=k)
            loss_list = cv_result(model, x_data, y_data, arg.batch_size, arg.monte_number, arg.random_state, arg.path_name, j, k, arg.device, 
                                  initial_num=n_pre, n_splits=arg.n_splits, pretrain_epoch=1, epoch_train=1,shuffle=True)
            dict_list.update(loss_list)
            loss_list = np.array(list(dict_list.items()))[:,1].astype('float')
            np.save('/home/jupyter-heojw/VAE-gamma/pretrain/loss_list/gamma_bhatta_save.npy', loss_list)               
    
    
    
    loss_list = np.array(list(dict_list.items()))[:,1].astype('float')
    key_list = np.array(list(dict_list.items()))[:,0]
    
    loss_list = np.nan_to_num(loss_list, nan = np.nanmean(loss_list))
    
    min_loss_list = np.array([])
    
    for i in range(0,len(latent_dim_list)*n_pre*len(num_cluster_list)):
        min_loss_list = np.append(min_loss_list,loss_list[((i*n_pre)):(((i+1)*n_pre))].min()) 
        
    mean_loss_list = np.array([])
    for i in range(0,len(latent_dim_list)*len(num_cluster_list)):
        mean_loss_list = np.append(mean_loss_list,min_loss_list[i*n_pre:(i+1)*n_pre].mean())
        
    optimal_list = []
    for i in range(0,len(num_cluster_list)):
        index_latent = np.where(mean_loss_list[i*len(latent_dim_list):(i+1)*len(latent_dim_list)]==mean_loss_list[i*len(latent_dim_list):(i+1)*len(latent_dim_list)].min())[0][0]
        optimal_list = np.append(optimal_list,latent_dim_list[index_latent])
        
    print(optimal_list)
    
    cluster_loss_list = np.array([])
    for i in range(0,len(num_cluster_list)):
        cluster_loss_list = np.append(cluster_loss_list,mean_loss_list[i*len(latent_dim_list):(i+1)*len(latent_dim_list)].min())
        
    print(cluster_loss_list)
    
    opt_cluster = np.where(cluster_loss_list==cluster_loss_list.min())[0][0]
    print(num_cluster_list[opt_cluster])
    
    latent_dim = np.int(optimal_list[np.where(cluster_loss_list==cluster_loss_list.min())[0][0]])
    print(latent_dim)
    
    
    final_list = {}
    for i in range(1, n_pre+1):      

        if torch.cuda.is_available():
            pre_model =  VAE_gamma(data_dim=arg.data_dim, latent_dim=latent_dim, device=arg.device, encodeLayer=arg.encodeLayer,
                                   decodeLayer=arg.decodeLayer, num_cluster=arg.num_cluster).cuda(arg.device)

        optimizer_auto = optim.Adam(pre_model.parameters(), lr=0.001) 
        scheduler = StepLR(optimizer_auto, step_size=20, gamma= 0.9)
        path_name = '/home/jupyter-heojw/VAE-gamma/pretrain/train/chowdary_pre_true_'+ str(latent_dim) +'_'+str(i)+'.pth'
        pretrain_epoch= 100
        
        pretrain_fit(dataloader, pre_model, optimizer_auto,scheduler,path_name,pretrain_epoch,monte_number,5,device=torch.device('cuda:5'))

        if torch.cuda.is_available():
            model_train = VAE_gamma(data_dim=arg.data_dim, latent_dim=latent_dim, device=arg.device, encodeLayer=arg.encodeLayer,
                                   decodeLayer=arg.decodeLayer, num_cluster=arg.num_cluster).cuda(arg.device)   
        saveWeightPath = arg.path_name
        optimizer = optim.Adam(model_train.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=20, gamma= 0.9)
        model_train.load_state_dict(torch.load(saveWeightPath))
        epoch_train=101
        
        path_name_2 = '/home/jupyter-heojw/VAE-gamma/pretrain/train/chowdary_true_' +str(latent_dim) + '_initial' + str(i) +  '.pth'

        loss = train_model(dataloader, dataloader_test, model_train, optimizer,scheduler,epoch_train,arg.batch_size,arg.monte_number,10,arg.device,path_name_2)
        final_list[' initial:'+str(i) + ' dim:' + str(latent_dim) ] = loss
        
    final_loss = np.array(list(final_list.items()))[:,1].astype('float')
    print(final_loss)
    
    optim_model = np.where(final_loss==final_loss.min())[0][0] + 1 
    
    path_name_best =  '/home/jupyter-heojw/pretrain/train/chowdary_true_' +str(latent_dim) + '_initial' + str(optim_model) +  '.pth'
    
    if torch.cuda.is_available():
        model =  VAE_gamma(data_dim=arg.data_dim, latent_dim=latent_dim, device=arg.device, encodeLayer=arg.encodeLayer,
                                   decodeLayer=arg.decodeLayer, num_cluster=arg.num_cluster).cuda(arg.device)   

    saveWeightPath = path_name_best

    model.load_state_dict(torch.load(saveWeightPath))
    
    model.eval()
    accuracy_list = []
    test_loss_list = []    
    test_loss_sum = 0.0
    for perform in range(0,100):
        test_loss = 0.0
        test_running_correct = 0.0
        label_set_test = torch.tensor([],dtype=int).cuda(arg.device)
        preds_set_test = torch.tensor([],dtype=int).cuda(arg.device)
        for batch_idx, data in enumerate(dataloader_test):

            img, label = data 
            img = img.float()
            img = img.view(img.size(0), -1)

            if torch.cuda.is_available():
                img = img.cuda(device)
                label = label.cuda(device).float()

            recon_al,recon_be, logalpha, logbeta, z = model(img)
            monte_gamma = 0
            monte_recon_loss = 0
            for l in range(monte_number):
                z_l = model.reparametrize(logalpha, logbeta)
                m = nn.Softplus()
                recon_al_l = 1e-6 + m(model.decode(z_l)[:,:model.data_dim])
                recon_be_l = 1e-6 + m(model.decode(z_l)[:,model.data_dim:])
                monte_gamma += model.get_gamma(z_l,len(dataloader_test.dataset))
                monte_recon_loss += model.recon_loss(recon_al_l,recon_be_l, img)
            sgvb_gamma_test = monte_gamma/monte_number
            sgvb_loss_test = torch.mean(monte_recon_loss/monte_number 
                                        + model.kl_loss(sgvb_gamma_test, logalpha, logbeta,len(dataloader_test.dataset))) #+ 1*model.Loss_b(img,z, logalpha, logbeta, 2, latent_dim))

            _, preds = torch.max(sgvb_gamma_test,1)

            label_set_test = torch.cat((label_set_test,label)).int()
            preds_set_test = torch.cat((preds_set_test,preds)).int()
            test_loss += sgvb_loss_test.item() * img.size(0)


        label_set_test = label_set_test.cpu()
        preds_set_test = preds_set_test.cpu()

        test_acc = cluster_acc(preds_set_test,label_set_test) * 100/preds_set_test.shape[0]
        test_loss_sum += test_loss
        accuracy_list = np.append(accuracy_list, test_acc)

    test_loss_list = np.append(test_loss_list, test_loss_sum/100)


    print('***** Test Result: loss: {:.3f} acc: {:.4f}'
      .format( test_loss_list / len(dataloader_test.dataset), np.max(accuracy_list)))
    
    
    while True:
        model.eval()
        test_loss = 0.0
        test_running_correct = 0.0
        label_set_test = torch.tensor([],dtype=int).cuda(arg.device)
        preds_set_test = torch.tensor([],dtype=int).cuda(arg.device)
        for batch_idx, data in enumerate(dataloader_test):

            img, label = data 
            img = img.float()
            img = img.view(img.size(0), -1)

            if torch.cuda.is_available():
                img = img.cuda(device)
                label = label.cuda(device).float()

            recon_al,recon_be, logalpha, logbeta, z = model(img)
            monte_gamma = 0
            monte_recon_loss = 0
            for l in range(monte_number):
                z_l = model.reparametrize(logalpha, logbeta)
                m = nn.Softplus()
                recon_al_l = 1e-6 + m(model.decode(z_l)[:,:model.data_dim])
                recon_be_l = 1e-6 + m(model.decode(z_l)[:,model.data_dim:])
                monte_gamma += model.get_gamma(z_l,len(dataloader_test.dataset))
                monte_recon_loss += model.recon_loss(recon_al_l,recon_be_l, img)
            sgvb_gamma_test = monte_gamma/monte_number
            sgvb_loss_test = torch.mean(monte_recon_loss/monte_number 
                                        + model.kl_loss(sgvb_gamma_test, logalpha, logbeta,len(dataloader_test.dataset)))

            _, preds = torch.max(sgvb_gamma_test,1)

            label_set_test = torch.cat((label_set_test,label)).int()
            preds_set_test = torch.cat((preds_set_test,preds)).int()
            test_loss += sgvb_loss_test.item() * img.size(0)


        label_set_test = label_set_test.cpu()
        preds_set_test = preds_set_test.cpu()

        test_acc = cluster_acc(preds_set_test,label_set_test) * 100/preds_set_test.shape[0]


        if test_acc == np.max(accuracy_list):
            print('***** Test Result: loss: {:.3f} acc: {:.4f}'
                  .format(test_loss / len(dataloader_test.dataset), test_acc))



            if latent_dim!=1:


                target, label = preds_set_test.numpy(), label_set_test.numpy()
                mo = TSNE( n_components = 2, init='pca')
                #         recon_al, recon_be, logalpha, logbeta, z = model(img)
                recon_al,recon_be, logalpha, logbeta, z = model(img)
                zs_tsne = mo.fit_transform( z.data.cpu().numpy())

                plt.figure(figsize = (10, 5))
                plt.title(f'Number of plotted data : {(preds_set_test.shape)[0]}', fontsize = 15)
                cmap = plt.get_cmap("tab10")
                for t in range(5):
                    points = zs_tsne[(target == t)]
                    plt.scatter(points[:, 0], points[:, 1], color=cmap(t), label=str(t))
                plt.legend()
                plt.savefig('/home/jupyter-heojw/pretrain/img/tsne_plot.png')
                plt.close()
            else:
                target, label = preds_set_test.numpy(), label_set_test.numpy()
                recon_al, recon_be, logalpha, logbeta, z = model(img)
                cmap = plt.get_cmap("tab10")
                z_data = z.cpu().detach().numpy()

                sns.kdeplot(z_data[np.where(preds_set_test == 0)[0],0], color=cmap(0), label=str(0)).set(xlim=(0))
                sns.kdeplot(z_data[np.where(preds_set_test == 1)[0],0], color=cmap(1), label=str(1)).set(xlim=(0))
                plt.legend()
                plt.savefig('/home/jupyter-heojw/pretrain/img/tsne_plot.png')


            break
        


