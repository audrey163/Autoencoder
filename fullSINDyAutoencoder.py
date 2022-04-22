import numpy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pprint as pp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math

from regression import PolynomialLibrary, TrigLibrary
import sindy_helper

class FullSINDyAutoencoder(nn.Module):
    def __init__(self,params=None):
        super().__init__()
        self.params = params if params is not None else self.get_params()
        self.Theta = sindy_helper.Theta(self.params['training']['batch_size'], self.params['architecture']['latent_dimension'])
        self.theta = sindy_helper.Theta.theta
        self.encoder = nn.Sequential(
            nn.Linear(self.params['architecture']['input_dimension'], 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, self.params['architecture']['latent_dimension'])
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.params['architecture']['latent_dimension'],4),
            nn.ReLU(),
            nn.Linear(4,8),
            nn.ReLU(),
            nn.Linear(8,self.params['architecture']['input_dimension'])
        )
        self.SINDy_layer = nn.Sequential(nn.Linear(len(self.Theta.candidate_terms),self.params['architecture']['latent_dimension'],bias=False))
    def forward(self, x, dx):
        ret = { 'X' : x, 'dX' : dx}
        ret['Z'] = self.encoder(x)
        ret['X_pred'] = self.decoder(ret['Z'])
        if self.params['optimization']['loss_reg']['SINDy'] > 0 or self.params['optimization']['loss_reg']['dX'] > 0:
            ret['dZ'] = self.get_dZ(x,dx)
            ret['dZ_pred'] = self.sindy(ret['Z'])
        if self.params['optimization']['loss_reg']['dX'] > 0:
            ret['dX_pred'] = self.get_dX(ret['Z'],ret['dZ_pred'])
            ret['dX'] = dx
        if self.params['optimization']['loss_reg']['Xi1'] > 0 or self.params['optimization']['loss_reg']['Xi2'] > 0 :
            ret['Xi'] = self.SINDy_layer[0].weight
        return ret

    def get_dZ(self,x,dx):
        dZ = torch.zeros(x.shape[0],self.params['architecture']['latent_dimension'])
        J = torch.autograd.functional.jacobian(self.encoder, x)
        for i in range(0,J.shape[0]):
            dZ[i,:] += torch.matmul(J[i,:,i,:],dx[i,:])
        return dZ

    def get_dX(self,z,dz):
        J = torch.autograd.functional.jacobian(self.decoder, z)
        dX = torch.zeros(J.shape[0],J.shape[1])
        for i in range(0,J.shape[0]):
            dX[i,:] += torch.matmul(J[i,:,i,:],dz[i,:])
        return dX

    def sindy(self,Z):
        return torch.matmul(self.Theta.theta(Z),self.SINDy_layer[0].weight.T)
        
    def loss(self,args):
        l = {}
        l['X'] =  self.params['optimization']['loss_reg']['X']*torch.nn.functional.l2_loss(args['X'] - args['X_pred'])
        if self.params['optimization']['loss_reg']['SINDy'] > 0:
            l['dZ'] = self.params['optimization']['loss_reg']['SINDy']*torch.nn.functional.l2_loss((args['dZ_pred'] - args['dZ'])
        if self.params['optimization']['loss_reg']['dX'] > 0:
            l['dX'] = self.params['optimization']['loss_reg']['dX']*torch.nn.functional.l2_loss(args['dX_pred'] - args['dX'],ord=2)
        if self.params['optimization']['loss_reg']['Xi1'] > 0:
            l['Xi1'] = self.params['optimization']['loss_reg']['Xi1']*torch.nn.functional.l2_loss((args['Xi'])
        if self.params['optimization']['loss_reg']['Xi2'] > 0:
            l['Xi2'] = self.params['optimization']['loss_reg']['Xi2']*torch.nn.functional.l1_loss(args['Xi'])  
        total = 0
        _str = ''
        for name in list(l):
            total += l[name]
            l[name] = float(l[name].detach().numpy())
        return total, l
    
    def show(self):
        rows = torch.Tensor(self.Xi_weights()).tolist()
        equations = [[round(coeff,7) for coeff in row] for row in rows]
        for i,eq in enumerate(equations):
            x = f'z{i+1}'
            rhs = ' + '.join(f'{coeff} {name}' for coeff, name in zip(eq, self.Theta.candidate_names))
            print(f'({x}.) = {rhs}.')
            
    
    def Xi_weights(self):
        return self.SINDy_layer[0].weight
    
    def save(self):
        import datetime
        timestamp = str(datetime.datetime.now())
        PATH = 'models/' + timestamp
        torch.save(self.state_dict(),PATH)
        print(f"Saved model to : {PATH}")

    def load(self,state_dict_PATH=None):
        if state_dict_PATH is not None:
            if state_dict_PATH == ':latest':
                from os import listdir
                from os.path import isfile, join
                onlyfiles = [f for f in listdir('models/') if isfile(join('models/', f))]
                onlyfiles.sort(reverse=True)
                state_dict_PATH = 'models/' + onlyfiles[0]
            self.load_state_dict(torch.load(state_dict_PATH))
            print(f"Using Model: {state_dict_PATH}")

    def train(self,dataset,dataloader,params=None):
        if params is not None:
            self.params = params
        optimizer = torch.optim.Adam(self.parameters(),lr=self.params['optimization']['learn_rate'])
        losses = []
        total_samples = len(dataset)
        n_iter = math.ceil(total_samples / self.params['training']['batch_size'])
        for epoch in range(self.params['training']['epochs']):
            print(f'Epoch {str(epoch)}')
            l = []
            for i, (X, dX) in enumerate(dataloader):
                if X.shape[0] == self.params['training']['batch_size']:
                    loss_dict = None
                    res = self.forward(X,dX)
                    loss, _loss_dict = self.loss(res) 
                    if loss_dict is None:
                        loss_dict = _loss_dict
                        loss_dict_n = 1
                    else:
                        for l in list(loss_dict):
                            loss_dict[l] = loss_dict[l]*(loss_dict_n-1) + _loss_dict[l]
                        loss_dict_n += 1
                    
                    #optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                l.append(loss.item())
            losses.append(np.mean(np.array(l)))
            if self.params['outputs']['loss']:
                print(f"\tLoss: {loss_dict}")
            if self.params['outputs']['sindy_show']:
                self.show()
            # if epoch % 3 == 0:
            #     self.show()
            if epoch % self.params['training']['save_freq'] == self.params['training']['save_freq'] - 1:
                 self.save()
            # if epoch % 50 == 0:
            #     X = torch.Tensor(dataset.Z.T)
            #     Z = self.encoder(X).detach().numpy()
            #     plt.plot(Z[:,1],Z[:,0])
            #     plt.show()
        
        plt.plot(np.log(np.array(losses)))
        plt.title("log loss")


    def train_dynamic(self,dataset,dataloader,maxiter=1e5):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.params['optimization']['learn_rate'])
        losses = []
        total_samples = len(dataset)
        n_iter = math.ceil(total_samples / self.params['training']['batch_size'])
        for s in list(self.params['optimization']['loss_reg']):
            self.params['optimization']['loss_reg'][s] = 0
        n = 0
        for s in  list(self.params['optimization']['loss_reg']):
            self.params['optimization']['loss_reg'][s] = 1
            n += 1
            print(f"Adding {s}. To Loss function")
            epoch = 0
            epoch_loss = 1e10
            while epoch_loss > n**2 and epoch < maxiter:
                print(f"Epoch {epoch}")
                l = []
                for i, (X, dX) in enumerate(dataloader):
                    if X.shape[0] == self.params['training']['batch_size']:
                        loss_dict = None
                        res = self.forward(X,dX)
                        loss, _loss_dict = self.loss(res) 
                        if loss_dict is None:
                            loss_dict = _loss_dict
                            loss_dict_n = 1
                        else:
                            for l in list(loss_dict):
                                loss_dict[l] = loss_dict[l]*(loss_dict_n-1) + _loss_dict[l]
                            loss_dict_n += 1
                    
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    l.append(loss.item())
                epoch_loss = sum([loss_dict[item] for item in list(loss_dict)])
                losses.append(epoch_loss)
                if self.params['outputs']['loss']:
                    print(f'\tLoss: { np.round(10*epoch_loss)/10 } \t {loss_dict}')
                if self.params['outputs']['sindy_show']:
                    self.show()
                # if epoch % 3 == 0:
                #     self.show()
                if epoch % self.params['training']['save_freq'] == self.params['training']['save_freq'] - 1:
                     self.save()
                epoch += 1
        
            plt.plot(np.log(np.array(losses)))
            plt.title("log loss")

        
        
        def get_params(self):
            import json
            with open('FullSINDyAutoencoder.conf') as json_file:
                 return json.load(json_file)
