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
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.model_selection import KFold
import torch.utils.data as data
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

class BasicDataset(data.Dataset):
    def __init__(self, x_tensor, y_tensor):
        super(BasicDataset, self).__init__()

        self.x = np.log(x_tensor+1)
        self.y = y_tensor
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

def generate_gamma(al, be, device):
    if al[torch.where(al < 1)].size()[0] > 0 :

        al_2 = al + 1 
        d = al_2-(1/3)
        c = 1/torch.sqrt(9*d)
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(al.size(),device=device).normal_()
        else:
            eps = torch.FloatTensor(al.size()).normal_()
        v = 1+c*eps

        while True:
            if v[torch.where(v <= 0 )].size()[0] > 0  : 
                if torch.cuda.is_available():
                    eps_new = torch.cuda.FloatTensor(al.size(),device=device).normal_()
                else:
                    eps_new = torch.FloatTensor(al.size()).normal_()
                v_new = 1+c*eps_new
                eps[torch.where((v <= 0 ))] = eps_new[torch.where((v <= 0 ))]
                v[torch.where((v <= 0 ))] = v_new[torch.where((v <= 0 ))]
                v = 1+c*eps
                continue
            break

        v = v*v*v
        u = torch.FloatTensor(al.size()).uniform_(0, 1)
        if torch.cuda.is_available():
            u = u.cuda(device)
        itera = 0    
        while True:
            itera += 1
            if itera//500 == 0 :
                if torch.cuda.is_available():
                    eps = torch.cuda.FloatTensor(al.size(),device=device).normal_()
                else:
                    eps = torch.FloatTensor(al.size()).normal_()
                v = 1+c*eps
                while True:
                    if v[torch.where((v <= 0 )| (eps <= -(1/c)))].size()[0] > 0  : 
                        if torch.cuda.is_available():
                            eps_new = torch.cuda.FloatTensor(al.size(),device=device).normal_()
                        else:
                            eps_new = torch.FloatTensor(al.size()).normal_()
                        v_new = 1+c*eps_new
                        eps[torch.where((eps <= -(1/c)))] = eps_new[torch.where((eps <= -(1/c)))]
                        v[torch.where((v <= 0 )| (eps <= -(1/c)))] = v_new[torch.where((v <= 0 )| (eps <= -(1/c)))]
                        v = 1+c*eps
                        continue
                    break
            index1= u >= 1-0.0331*(eps*eps)*(eps*eps)
            index2= torch.log(u) >= 0.5 * eps * eps + d * (1 - v + torch.log(v))
            if (u[torch.where(index1)].size()[0] > 0 ):
                if (u[torch.where(index1 & index2)].size()[0] > 0 ):
                    u_new = torch.FloatTensor(al.size()).uniform_(0, 1)
                    if torch.cuda.is_available():
                        u_new = u_new.cuda(device)
                    u[torch.where(index1 & index2)] =  u_new[torch.where(index1 & index2)]
            return (torch.log(d*v + 1e-6) + torch.log(u + 1e-6)/(al+1e-6)).exp()/(be+1e-6)              


#     elif al[torch.where(al >= 1)].size()[0] > 0 :
    else:
        d = al-(1/3)
        c = 1/torch.sqrt(9*d)
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(al.size(),device=device).normal_()
        else:
            eps = torch.FloatTensor(al.size()).normal_()
        v = 1+c*eps

        while True:
            if v[torch.where(v <= 0 )].size()[0] > 0  : 
                if torch.cuda.is_available():
                    eps_new = torch.cuda.FloatTensor(al.size(),device=device).normal_()
                else:
                    eps_new = torch.FloatTensor(al.size()).normal_()
                v_new = 1+c*eps_new
                eps[torch.where((v <= 0 ))] = eps_new[torch.where((v <= 0 ))]
                v[torch.where((v <= 0 ))] = v_new[torch.where((v <= 0 ))]
                v = 1+c*eps
                continue
            break

        v = v*v*v
        u = torch.FloatTensor(al.size()).uniform_(0, 1)
        if torch.cuda.is_available():
            u = u.cuda(device)
        itera = 0    
        while True:
            itera += 1
            if itera//500 == 0 :
                if torch.cuda.is_available():
                    eps = torch.cuda.FloatTensor(al.size(),device=device).normal_()
                else:
                    eps = torch.FloatTensor(al.size()).normal_()
                v = 1+c*eps
                while True:
                    if v[torch.where((v <= 0 )| (eps <= -(1/c)))].size()[0] > 0  : 
                        if torch.cuda.is_available():
                            eps_new = torch.cuda.FloatTensor(al.size(),device=device).normal_()
                        else:
                            eps_new = torch.FloatTensor(al.size()).normal_()
                        v_new = 1+c*eps_new
                        eps[torch.where((eps <= -(1/c)))] = eps_new[torch.where((eps <= -(1/c)))]
                        v[torch.where((v <= 0 )| (eps <= -(1/c)))] = v_new[torch.where((v <= 0 )| (eps <= -(1/c)))]
                        v = 1+c*eps
                        continue
                    break
            index1= u >= 1-0.0331*(eps*eps)*(eps*eps)
            index2= torch.log(u) >= 0.5 * eps * eps + d * (1 - v + torch.log(v))
            if (u[torch.where(index1)].size()[0] > 0 ):
                if (u[torch.where(index1 & index2)].size()[0] > 0 ):
                    u_new = torch.FloatTensor(al.size()).uniform_(0, 1)
                    if torch.cuda.is_available():
                        u_new = u_new.cuda(device)
                    u[torch.where(index1 & index2)] =  u_new[torch.where(index1 & index2)]
                    continue
            return (d*v)/(be+1e-6) 
        
        
        
def clusterAcc(Y_pred, Y):
    assert Y_pred.shape[0] == Y.shape[0]
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.shape[0]):
        w[Y_pred[i], Y[i]] += 1
    rowMax, _ = torch.max(torch.tensor(w), dim = 1)
    return sum(rowMax)*1.0 /Y_pred.shape[0], w


def cluster_acc(Y_pred, Y):
    assert Y_pred.shape[0] == Y.shape[0]
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.shape[0]):
        w[Y_pred[i], Y[i]] += 1
        ind1,ind2 = linear_assignment(w.max() - w)
    a = 0
    for i,j in zip(ind1,ind2):
        a += w[i,j]
    return a


class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint_mnist.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif (score > self.best_score + self.best_score * self.delta) :
            self.counter += 1
#             print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
#         if self.verbose:
#             print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
        
class VAE_gamma(nn.Module):
    def __init__(self, data_dim, latent_dim,device, encodeLayer=[], decodeLayer=[],num_cluster=5):
        super(VAE_gamma, self).__init__()

        
        self.n_cluster = num_cluster
        self.device = device
        self.data_dim = data_dim
        self.latent_dim =latent_dim
        self.z_log_alpha_prior = Parameter( torch.randn(self.latent_dim,self.n_cluster).cuda(self.device), requires_grad=True )
        self.z_log_beta_prior = Parameter( torch.randn(self.latent_dim,self.n_cluster).cuda(self.device) , requires_grad=True )
        self.c_prior = Parameter( (torch.ones(self.n_cluster)).cuda(device), requires_grad=True  )
        
        self.fc1 = nn.Linear(self.data_dim, encodeLayer[0])
        self.fc2 = nn.Linear(encodeLayer[0], encodeLayer[1])
        self.fc3 = nn.Linear(encodeLayer[1], encodeLayer[2])
        self.fc41 = nn.Linear(encodeLayer[2], self.latent_dim)
        self.fc42 = nn.Linear(encodeLayer[2], self.latent_dim)

        self.fc4 = nn.Linear(self.latent_dim, encodeLayer[2])
        self.fc5 = nn.Linear(encodeLayer[2],encodeLayer[1])
        self.fc6 = nn.Linear(encodeLayer[1], encodeLayer[0])        
        self.fc7 = nn.Linear(encodeLayer[0], self.data_dim *2)




    def encode(self, x):
        
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        m = nn.Softplus()
        return 1e-6 + m(self.fc41(h3)), 1e-6 + m(self.fc42(h3))
#         return self.fc41(h3), self.fc42(h3)
    
    def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        h6 = F.relu(self.fc6(h5))

        return self.fc7(h6)

    def weights(self):
        return torch.softmax(self.c_prior, dim=0)


    def reparametrize(self, logalpha, logbeta):
        al = logalpha
        be = logbeta
        return generate_gamma(al, be, self.device)
                

        
    def forward(self, x):
        
        logalpha, logbeta = self.encode(x)
        z = self.reparametrize(logalpha, logbeta)
        m = nn.Softplus()
        
        recon_al = 1e-6 + m(self.decode(z)[:,:self.data_dim])
        recon_be = 1e-6 + m(self.decode(z)[:,self.data_dim:])

    
        return recon_al,recon_be, logalpha, logbeta, z  


    def get_gamma(self, z ,batch_size):
        z_t = z.unsqueeze(2).repeat(1,1,self.n_cluster)

        
        c_prior = self.weights()
        c_prior_t = c_prior.unsqueeze(0).repeat(batch_size,1)
        z_alpha_prior_t = self.z_log_alpha_prior.unsqueeze(0).repeat(batch_size,1,1).exp()
        z_beta_prior_t = self.z_log_beta_prior.unsqueeze(0).repeat(batch_size,1,1).exp()

        p_c_z = (torch.log(c_prior_t+ 1e-6) + torch.sum(z_alpha_prior_t*torch.log(z_beta_prior_t+ 1e-6)-torch.lgamma(z_alpha_prior_t+ 1e-6)+(z_alpha_prior_t-1)*torch.log(z_t+ 1e-6)-z_beta_prior_t*z_t,dim=1)).exp() + 1e-10

        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return gamma

    def kl_loss(self,gamma, z_logalpha, z_logbeta, batch_size):


        z_alpha_t = z_logalpha.unsqueeze(2).repeat(1,1, self.n_cluster)
        z_beta_t = z_logbeta.unsqueeze(2).repeat(1,1,self.n_cluster)


        c_prior = self.weights()
        c_prior_t = c_prior.unsqueeze(0).repeat(batch_size,1)

        z_alpha_prior_t = self.z_log_alpha_prior.unsqueeze(0).repeat(batch_size,1,1).exp()
        z_beta_prior_t = self.z_log_beta_prior.unsqueeze(0).repeat(batch_size,1,1).exp()

        gamma_t = gamma.unsqueeze(1).repeat(1,self.latent_dim,1)

        kl_pzx_pzc = -torch.sum( gamma_t*(z_alpha_prior_t*torch.log(z_beta_prior_t+ 1e-6) - torch.lgamma(z_alpha_prior_t+ 1e-6) + (z_alpha_prior_t-1)*(torch.digamma(z_alpha_t+ 1e-6)-torch.log(z_beta_t+ 1e-6)) \
                                              -(z_alpha_t)*z_beta_prior_t/z_beta_t), dim=(1,2)) \
                     + torch.sum(torch.log(z_logbeta + 1e-6) - torch.lgamma(z_logalpha + 1e-6) + (z_logalpha-1)*torch.digamma(z_logalpha+ 1e-6)- (z_logalpha),dim=-1)\
                     -torch.sum( (c_prior_t+1e-6).log() * gamma, dim=1)\
                     +torch.sum( (gamma+ 1e-6).log() * gamma, dim = 1 )



        return kl_pzx_pzc
    
    def recon_loss(self, recon_al,recon_be, x):

        recon_loss = recon_al*torch.log(recon_be+ 1e-6)-torch.lgamma(recon_al+ 1e-6)+(recon_al-1)*torch.log(x + 1e-6)-recon_be*(x)
        recon_loss = torch.sum(-recon_loss, dim=1)
        return recon_loss
    
    
def pretrain_fit(dataloader, pre_model,optimizer_auto,scheduler,path_name,pretrain_epoch,monte_number,num_cluster,device,verbose=False):
    
    
    for epoch in range(pretrain_epoch):

        label_set = torch.tensor([],dtype=int).cuda(device)
        preds_set = torch.tensor([],dtype=int).cuda(device)
        train_loss = 0

        for batch_idx, data in enumerate(dataloader):
            pre_model.train()
            img, label = data
            img = img.float()
            img = img.view(img.size(0), -1)

            if torch.cuda.is_available():
                img = img.cuda(device)
                label = label.cuda(device)

            recon_al,recon_be, logalpha, logbeta, z = pre_model(img)

            monte_recon_loss = 0
            for l in range(monte_number):
                z_l = pre_model.reparametrize(logalpha, logbeta)
    #             recon_al_l = model.decode(z)
    #             monte_recon_loss += F.binary_cross_entropy(recon_al_l, img, reduction='sum')
                m = nn.Softplus()
                recon_al_l = 1e-6 + m(pre_model.decode(z_l)[:,:pre_model.data_dim])
                recon_be_l = 1e-6 + m(pre_model.decode(z_l)[:,pre_model.data_dim:])

                monte_recon_loss += pre_model.recon_loss(recon_al_l,recon_be_l, img)

            loss = torch.mean(monte_recon_loss/monte_number)

    #         loss = criterion(img, recon_x)
    #         loss = model.recon_loss(recon_al, img)       
    #         loss = torch.mean(loss) #+ 0.1*pre_model.Loss_b(img, z, logalpha, logbeta, 10, latent_dim))
    #         loss = F.binary_cross_entropy(recon_al, img, reduction='sum')
            train_loss += loss.item()

            optimizer_auto.zero_grad()
            loss.backward()
            optimizer_auto.step()
        scheduler.step()
        
#         early_stopping(train_loss / len(dataloader.dataset), pre_model)

#         if early_stopping.early_stop:
#             print("Early stopping")
#             break
    #     scheduler.step()
    
        if verbose == True:
            print('Epoch: {} loss: {:.3f}'
                  .format(epoch, train_loss / len(dataloader.dataset)))
        
    x = torch.Tensor([])
    y = torch.Tensor([])
    for sample,label in dataloader:
        data = sample
        x = torch.cat([x,data])
        y = torch.cat([y,label])
        
    pre_model = pre_model.cuda(device)
    with torch.no_grad():
        x = x.view(-1, pre_model.data_dim)
        x = x.float()
        if torch.cuda.is_available():
            x = x.cuda(device)

    #     z, recon_al, recon_be = pre_model(x)
        recon_al,recon_be, logalpha, logbeta, z = pre_model(x)

    z = z.cpu()

    pre_model = pre_model.cpu()
    stateDict = pre_model.state_dict()
    gmm = GaussianMixture(n_components=num_cluster , covariance_type='diag').fit(z)    
#     print(f'\n# Pretraining finished.. ')

#     model.load_state_dict(stateDict, strict = False)
    pre_model.c_prior.data = (torch.from_numpy(gmm.weights_)).float()
    z_mu = torch.transpose(torch.from_numpy(gmm.means_).float(),0,1)
    z_var = torch.transpose(torch.from_numpy(gmm.covariances_).float(),0,1)

    alpha_hat_list = z_mu.pow(2) / z_var
    beta_hat_list = z_mu / z_var

    pre_model.z_log_alpha_prior.data = alpha_hat_list.float().log()
    pre_model.z_log_beta_prior.data = beta_hat_list.float().log()

    torch.save(pre_model.state_dict(), path_name)
#     print(f'# Pretrain weight saved..')


    pre_model = pre_model.cuda(device)

    a = y.int().detach().cpu().numpy()
    print('Accurasy: '+ str(cluster_acc(gmm.predict(z), a)*100/a.shape[0]))
    from sklearn.metrics.cluster import adjusted_rand_score
    print('ARI: ' + str(adjusted_rand_score(gmm.predict(z),a)))
    
    
    
def train_model(dataloader, dataloader_test, model, optimizer,scheduler, epoch_train,batch_size,monte_number,mean_num,device,path_name_2='checkpoint_model.pth'):
    
    early_stopping = EarlyStopping(patience = 7, verbose = True, delta = 0.001,path=path_name_2)
    
    for epoch in range(epoch_train):    

        label_set = torch.tensor([],dtype=int).cuda(device)
        preds_set = torch.tensor([],dtype=int).cuda(device)
        train_loss = 0
        running_correct = 0
        label_set = torch.tensor([],dtype=int).cuda(device)
        preds_set= torch.tensor([],dtype=int).cuda(device)
        
        for batch_idx, data in enumerate(dataloader):
            model.train()
            img, label = data
            img = img.float()
            img = img.view(img.size(0), -1)

            if torch.cuda.is_available():
                img = img.cuda(device)
                label = label.cuda(device)


            optimizer.zero_grad()

            recon_al, recon_be, logalpha, logbeta, z = model(img)

            monte_gamma = 0
            monte_recon_loss = 0
            for l in range(monte_number):
                z_l = model.reparametrize(logalpha, logbeta)
                m = nn.Softplus()
                recon_al_l = 1e-6 + m(model.decode(z_l)[:,:model.data_dim])
                recon_be_l = 1e-6 + m(model.decode(z_l)[:,model.data_dim:])
                monte_gamma += model.get_gamma(z_l,batch_size)

                monte_recon_loss += model.recon_loss(recon_al_l,recon_be_l, img)

            sgvb_gamma = monte_gamma/monte_number
            sgvb_loss = torch.mean(monte_recon_loss/monte_number 
                                   + model.kl_loss(sgvb_gamma, logalpha, logbeta ,batch_size)) #+ 1*model.Loss_b(img, z, logalpha, logbeta, 10, latent_dim))

            _, preds = torch.max(sgvb_gamma,1)
            label_set = torch.cat((label_set,label)).int()
            preds_set = torch.cat((preds_set,preds)).int()
            train_loss += sgvb_loss.item() * img.size(0)

            sgvb_loss.backward()
            optimizer.step()
        scheduler.step()
    
        

        train_acc = cluster_acc(preds_set,label_set) * 100/preds_set.shape[0]
#         if epoch == 1 or epoch%30 == 0:
#         print('Epoch: {} loss: {:.3f} accuracy: {:.3f}'
#               .format(epoch, train_loss / len(dataloader.dataset), train_acc))


        model.eval()
        accuracy_list = [0]
        test_loss_list = []    
        test_loss_sum = 0.0
        for perform in range(0,mean_num):
            test_loss = 0.0
            test_running_correct = 0.0
            label_set_test = torch.tensor([],dtype=int).cuda(device)
            preds_set_test = torch.tensor([],dtype=int).cuda(device)
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
            if accuracy_list[0] < test_acc:
                accuracy_list[0] = test_acc
            
        test_loss_list = np.append(test_loss_list, test_loss_sum/mean_num)

        early_stopping( (test_loss_sum/mean_num) / len(dataloader.dataset), model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    print('***** Test Result: loss: {:.3f} acc: {:.4f}'
          .format( early_stopping.val_loss_min, accuracy_list[0]))
    
    return early_stopping.val_loss_min
#     if epoch % scheduler.step_size == 0:
#         print(f'\n## Learning rate decay.. lr : {scheduler.get_last_lr()[0]}')


        

#     if early_stopping.early_stop:
#         print("Early stopping")
#         break
    
# model.load_state_dict(torch.load('checkpoint_mnist.pt'))    
    

    
def cv_result_1(model,x_data,y_data,batch_size,monte_number,random_state,path_name, latent_dim,num_cluster,device,
                initial_num = 5, mean_num=10,pretrain_epoch=30,epoch_train=100,n_splits=3,lr=0.001,step_size=20,gamma=0.9,shuffle=True):
    cv_loss = np.array([])

    kfold = KFold(n_splits=n_splits,shuffle=shuffle,random_state = random_state)
    j=0
    dict_list = {}

    b1 = 0.5
    b2 = 0.99  # 99
    decay = 2.5 * 1e-5
    
    for train_idx, test_idx in kfold.split(x_data):
        X_train, X_test = x_data[train_idx], x_data[test_idx]
        Y_train, Y_test = y_data[train_idx], y_data[test_idx]
        j += 1
        for i in range(1,initial_num+1):
            if torch.cuda.is_available():
                pre_model = model.cuda(device)
                
            train_dataset = BasicDataset(X_train,Y_train)
            test_dataset = BasicDataset(X_test,Y_test)
        
            dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)
            dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, drop_last = True)
            
            optimizer_auto = optim.Adam(pre_model.parameters(), lr=lr)   
            scheduler = StepLR(optimizer_auto, step_size=step_size, gamma= gamma)
            path_name_pre = path_name + '_pretrain_fold_' + str(j) + '_lat_' + str(latent_dim) +'_init_'+str(i)+'.pth'
            pretrain_epoch= pretrain_epoch
            pretrain_fit(dataloader, pre_model, optimizer_auto,scheduler,path_name_pre,pretrain_epoch,monte_number,num_cluster,device)
            
            if torch.cuda.is_available():
                model_train = model.cuda(device)        
            saveWeightPath = path_name_pre
            
            optimizer = optim.Adam(model_train.parameters(), lr=lr)
            scheduler = StepLR(optimizer, step_size=step_size, gamma= gamma)
            model_train.load_state_dict(torch.load(saveWeightPath))
            epoch_train=epoch_train
            loss = train_model(dataloader, dataloader_test, model_train, optimizer,scheduler,epoch_train,batch_size,monte_number,mean_num,device)
            dict_list['components:'+str(num_cluster)+' fold:'+ str(j) + ' initial:'+str(i) + ' dim:' + str(latent_dim) ] = loss

    return (dict_list)