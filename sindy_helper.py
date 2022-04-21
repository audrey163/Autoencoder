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


class Theta:
    def __init__(self,num_snapshots, num_features):
        from regression import PolynomialLibrary, TrigLibrary
        self.num_snapshots, self.num_features  = num_snapshots , num_features 
        self.feature_names = [f'x{i+1}' for i in range(self.num_features)]
        self.candidate_terms = [ lambda x: torch.ones(self.num_snapshots) ]
        self.candidate_names = ['1']
        self.libs = [ PolynomialLibrary(max_degree=1), TrigLibrary() ]
        for lib in self.libs:
            lib_candidates = lib.get_candidates(self.num_features, self.feature_names)
            for term, name in lib_candidates:
                self.candidate_terms.append(term)
                self.candidate_names.append(name)
    def theta(self,X):
         return torch.stack(tuple(f(X) for f in self.candidate_terms), axis=1)
