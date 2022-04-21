import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pprint as pp

class PolynomialLibrary:
    def __init__(self, max_degree=2, cross_terms=True):
        self.max_degree = max_degree
        self.cross_terms = cross_terms
        
    def get_candidates(self, dim, feature_names):
        self.feature_names = feature_names
        return [self.__polynomial(degree_sequence) 
                    for degree in range(1,self.max_degree+1)
                        for degree_sequence in self.__get_degree_sequences(degree, dim)]
    
    
    def __polynomial(self, degree_sequence):
        def fn(X):
            terms = torch.stack( tuple(X[:,i]**d for i,d in enumerate(degree_sequence)), axis=1 )
            return torch.prod(terms, dim=1)
        fn_name = ' '.join(self.__display_term(self.feature_names[i],d) for i,d in enumerate(degree_sequence) if d)    
        return (fn, fn_name)
    
    def __display_term(self, feature_name, d):
        if d == 1:
            return f'{feature_name}'
        return f'{feature_name}^{d}'
    
    def __get_degree_sequences(self, degree, num_terms):
        if num_terms == 1:  return [[degree]]
        if degree == 0:     return [[0 for _ in range(num_terms)]]
        res = []
        for d in reversed(range(degree+1)):
            for seq in self.__get_degree_sequences(degree-d, num_terms-1):
                res.append([d, *seq])
        return res

class TrigLibrary:
    def __init__(self):
        self.max_freq = 1
    
    def get_candidates(self, dim, feature_names):
        self.feature_names = feature_names
        return [trig(i) for trig in [self.__sin, self.__cos] for i in range(dim)]

    def __sin(self,i):
        fn      = lambda X: torch.sin(X[:,i])
        fn_name = f'sin({self.feature_names[i]})'
        return (fn, fn_name)

    def __cos(self,i):
        fn      = lambda X: torch.cos(X[:,i])
        fn_name = f'cos({self.feature_names[i]})'
        return (
            fn,
            fn_name
        )
