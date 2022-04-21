import numpy as np
import pprint as pp
import matplotlib.pyplot as plt
from scipy.stats import special_ortho_group
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
        return torch.Tensor(self.Z[:,index]),torch.Tensor(self.dZ[:,index])
        if self.embed_params is not None:
            return torch.Tensor(self.Z[:,index]),torch.Tensor(self.dZ[:,index])
        else:
            return torch.Tensor(self.X[:,index]),torch.Tensor(self.dX[:,index])
    def __len__(self):
        return self.time_steps
    def solve(self):
        # time
        t = np.linspace(self.time[0],self.time[1],self.time_steps)
        # set inital condition
        soln = solve_ivp(self.f, self.time, self.x0, dense_output = True, rtol=1e-8, atol=1e-8)
        x = soln.sol(t)
        dx = np.array([self.f(t[i],x[:,i]) for i in range(0,t.shape[0]) ]).T
        return torch.Tensor(x), torch.Tensor(dx)

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
            return np.dot(self.embed_mat[:,:self.state_size], w + self.embed_params['sigma'] * np.random.randn(self.state_size,self.time_steps) + self.embed_params['mu']) + self.embed_params['sigma1']*np.random.randn(self.embed_mat.shape[0],self.time_steps) + self.embed_params['mu1']
        self.Z = emb(self.X)
        self.dZ = emb(self.dX)
    def unembed(self):
        def unemb(w):        
            if self.embed_params['mat'] == 'SO':
                X = np.dot(w.T,self.embed_mat[:,:self.state_size]).T
            elif self.embed_params['mat'] == 'RANDN':
                pinv = np.linalg.pinv(np.dot(self.embed_mat.T,self.embed_mat)) #(m,m)
                rinv = np.dot(self.embed_mat,pinv)
                X = np.dot(w.T,rinv).T
        X = torch.Tensor(unemb(self.Z))
        dX = torch.Tensor(unemb(self.dZ))
        return {'X' : X, 'X_err' : np.sum(np.abs(X-self.X)), 'dX' : dX, 'dX_err' : np.sum(np.abs(dX-self.dX))}

    
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
    def plot(self, WIDTH = 1000, HEIGHT = 750, DPI =  100,X=None):
        fig = plt.figure(figsize=(WIDTH/DPI, HEIGHT/DPI))
        fig.add_subplot(projection='3d')
        ax = fig.gca()
        ax.set_facecolor('k')
        fig.subplots_adjust(left=0, right=1, bottom=-1, top=1)
        # Make the line multi-coloured by plotting it in segments of length s which
        # change in colour across the whole time series.
        s = 10
        X = self.X.detach().numpy()
        for i in range(0,self.time_steps,s):
            ax.plot(X[0][i:i+s+1], X[1][i:i+s+1], X[2][i:i+s+1], alpha=0.4)

class SimplePendulum(DynamicalSystem):
    def __init__(self,g,l,mu,theta0):
        self.state_size = 2
        self.time_steps = 10000
        self.time = (0,10)
        self.x0 = (theta0,0)
        self.beta = beta = (g,l,mu)
        self.embed_params = {'embed_dim' : 10, 'mu' : 0,'sigma' : 0,'mu1' : 0,'sigma1' : 0,'mat' : 'RANDN'}
        super().__init__(self.state_size,self.time_steps,self.time,self.f,self.x0,self.beta,u=None,random_seed=1,embed_params=self.embed_params)
    def f(self,t,X):
        theta, theta_dot = X
        return np.array([ theta_dot, -self.beta[0] / self.beta[1] * np.sin(theta) - self.beta[2] * theta_dot ])
    def get_xy(self):
        return np.array([ np.sin(self.X[0,:]), np.cos(self.X[0,:]) ])
    


# def get_training_data(embed_dim = 10):
#     dynamical_system = SimplePendulum(g=9.8,l=2,mu=0.5,theta0=np.pi/2)
#     dynamical_system.embed(n=embed_dim,mat='RANDN',mu=0,sigma=0) #embed the pendulum in a higer diminsional space with noise
#     return { 'X' : torch.Tensor(dynamical_system.Z.T) , 'dX' : torch.Tensor(dynamical_system.dZ.T)}
# class TrainDataset(Dataset):
#     def __init__(self):
#         data = get_training_data()
#         self.X = data['X']
#         self.dX = data['dX']
#         self.n_samples = data['X'].shape[0]
#     def __getitem__(self,index):
#         return self.X[index,:],self.dX[index,:]
#     def __len__(self):
#         return self.n_samples
