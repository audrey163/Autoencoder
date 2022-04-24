import numpy as np
import pprint as pp
import matplotlib.pyplot as plt
#from scipy.stats import special_ortho_group
from scipy.integrate import solve_ivp
import torch
from torch.utils.data import Dataset, DataLoader

class DynamicalSystem(Dataset):
    def __init__(self,state_size,time_steps,time,f,x0,beta,u=None,random_seed=None,embed_params=None):
        '''
            Params:
                state_size - the number of variables in your dynamical system
                time_steps - the number of time steps in the discrete time system
                f - this is the vector field of the dynamical system
                x0 - the initial condition of type tuple
                u - this is the control perameter for conitrol systems
                beta - this is the tuple of parameters for the construction of the system,
                        e.g. beta = (9.81, 2.997e8, 8.99e10)
        '''
        assert isinstance(time, tuple), "Error: Dynamical System - time should be tuple ( t_i , t_f )"
        assert isinstance(beta, tuple), "Error: Dynamical System - beta should be tuple (b0,...,bn)"
        assert isinstance(x0, tuple), "Error: Dynamical System - x0 should be tuple"
        # The dimminsions of X
        self.random_seed = random_seed
        self.state_size = state_size
        self.time_steps = time_steps
        self.time = time
        self.x0 = x0
        # X is the states
        self.X, self.dX = self.solve()
        self.embed_params = embed_params
        if embed_params is not None:#n,mu=0,sigma=0,mu1=0,sigma1=0,mat='RANDN'
            #embed_params = {'embed_dim' : 10, 'mu' : 0,'sigma' : 0,'mu1' : 0,'sigma1' : 0,'mat' : 'RANDN'}
            self.embed()

    def __getitem__(self,index):
        return self.Z[:,index],self.dZ[:,index]
        if self.embed_params is not None:
            return self.Z[:,index],self.dZ[:,index]
        else:
            return self.X[:,index],self.dX[:,index]
    def __len__(self):
        return self.time_steps
    def solve(self):
        # time
        t = np.linspace(self.time[0],self.time[1],self.time_steps)
        # set inital condition
        soln = solve_ivp(self.f, self.time, self.x0, dense_output = True, rtol=1e-8, atol=1e-8)
        x = soln.sol(t)
        dx = np.array([self.f(t[i],x[:,i]) for i in range(0,t.shape[0]) ]).T
        return x, dx

    def embed(self):
        #embed_params = {'embed_dim' : 10, 'mu' : 0,'sigma' : 0,'mu1' : 0,'sigma1' : 0,'mat' : 'RANDN'}
        if self.embed_params['mat'] == 'SO':
            self.embed_mat = special_ortho_group.rvs(self.embed_params['embed_dim'])
        elif self.embed_params['mat'] == 'RANDN':
            self.embed_mat = np.random.randn(self.embed_params['embed_dim'],self.state_size)
        # X_shape = (self.state_size,self.time_steps)
        def emb(w):
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            n0 = self.embed_params['sigma'] * np.random.randn(self.state_size,self.time_steps) + self.embed_params['mu']
            n1 = self.embed_params['sigma1']*np.random.randn(self.embed_mat.shape[0],self.time_steps) + self.embed_params['mu1']
            return np.dot(self.embed_mat[:,:self.state_size], w + n0)+n1
        self.Z = emb(self.X)
        self.dZ = emb(self.dX)
    def unembed(self):
        def unemb(w):        
            if self.embed_params['mat'] == 'SO':
                X = np.dot(T,self.embed_mat[:,:self.state_size]).T
            elif self.embed_params['mat'] == 'RANDN':
                pinv = np.linalg.pinv(np.dot(self.embed_mat.T,self.embed_mat)) #(m,m)
                rinv = np.dot(self.embed_mat,pinv)
                X = np.dot(w.T,rinv).T
        X = (unemb(self.Z))
        dX = unemb(self.dZ)
        return {'X' : X, 'X_err' : np.sum(np.abs(X-self.X)), 'dX' : dX, 'dX_err' : np.sum(np.abs(dX-self.dX))}

    def to(self,device):
        self.X = torch.Tensor(self.X)
        self.dX = torch.Tensor(self.dX)
        self.Z = torch.Tensor(self.Z)
        self.dZ = torch.Tensor(self.dZ)
        self.X.to(device)
        self.dX.to(device)
        self.Z.to(device)
        self.dZ.to(device)

class Lorenz(DynamicalSystem):
    def __init__(self):
        self.state_size = 3
        self.time_steps = 10000
        self.time = (0,100)
        self.x0 = (0,1,1.05)
        self.beta = (10.0,2.667,28.0)
        self.embed_params = {'embed_dim' : 10, 'mu' : 0,'sigma' : 0,'mu1' : 0,'sigma1' : 0,'mat' : 'RANDN'}
        super().__init__(self.state_size,self.time_steps,self.time,self.f,self.x0,self.beta,u=None,random_seed=1,embed_params=self.embed_params)
    def f(self,t,X):
        x,y,z = X
        return np.array([
                self.beta[0] * (y-x),
                x * (self.beta[2] - z) - y,
                x * y - self.beta[1]*z
            ])
    # def plot(self, WIDTH = 1000, HEIGHT = 750, DPI =  100,X=None):
    #     fig = plt.figure(figsize=(WIDTH/DPI, HEIGHT/DPI))
    #     fig.add_subplot(projection='3d')
    #     ax = fig.gca()
    #     ax.set_facecolor('k')
    #     fig.subplots_adjust(left=0, right=1, bottom=-1, top=1)
    #     # Make the line multi-coloured by plotting it in segments of length s which
    #     # change in colour across the whole time series.
    #     s = 10
    #     X = self.X.detach().numpy()
    #     for i in range(0,self.time_steps,s):
    #         ax.plot(X[0][i:i+s+1], X[1][i:i+s+1], X[2][i:i+s+1], alpha=0.4)

class SimplePendulum(DynamicalSystem):
    def __init__(self,g,l,mu,theta0,random_seed=None,embed_params=None,time_steps=10000,time=10):
        self.state_size = 2
        self.time_steps = time_steps
        self.time = (0,time)
        self.x0 = (theta0,0)
        self.beta = beta = (g,l,mu)
        self.embed_params = {'embed_dim' : 10, 'mu' : 0,'sigma' : 0,'mu1' : 0,'sigma1' : 0,'mat' : 'RANDN'}
        super().__init__(self.state_size,self.time_steps,self.time,self.f,self.x0,self.beta,u=None,random_seed=1,embed_params=self.embed_params)
    def f(self,t,X):
        theta, theta_dot = X
        return np.array([ theta_dot, -self.beta[0] / self.beta[1] * np.sin(theta) - self.beta[2] * theta_dot ])
    def get_xy(self):
        return np.array([ np.sin(self.X[0,:]), np.cos(self.X[0,:]) ])

