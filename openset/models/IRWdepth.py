#!/usr/bin/env python3

import numpy as np
from numpy.random import default_rng



class IRWdepth():
    '''Integrated Rank-Weighted depth
    based on NIPS2022 paper P.Colombo et al., Beyond Mahalanobis-Based Scores for Textual OOD Detection
    (Monte Carlo approximation - page 5 of main paper, and Algorithm A.1 on page 18 of Supplementary Material)
    '''

    #def fit(self, X: np.ndarray, nproj: int | None = 1000) -> bool:
    def fit(self, X: np.ndarray, nproj = 1000):
        """ 
        X nxd - n training samples in d dimensions
        nproj - number of random vectors of hypersphere S
        """
        self.nproj = nproj

        self.n = X.shape[0]
        self.d = X.shape[1]

        self.U = np.empty((self.d, self.nproj))      # d x nproj, nproj random directions on hypersphere S^(d-1)
        self.M = np.empty((self.n, self.nproj))      # = XU, n x nproj, element M[i,j] = <xi, Uj>, where xi = i-th row of X, Uj = j-th column of U

        
        # generate U
        rng = default_rng()
        mi = np.zeros(self.d)
        cov = np.identity(self.d)
        U_ = rng.multivariate_normal(mi, cov, nproj).T
        self.U = U_/np.linalg.norm(U_, axis=0)       # normalized to length 1
        self.M = np.dot(X,self.U)

        print(f'fitted IRWdepth model with {self.nproj} projections in {self.n} dimensions')
        
        return True


    def score(self, x: np.ndarray) -> np.ndarray:    # formula D_IRW(x,Sn) on page 5 of NIPS paper

        v = np.dot(x, self.U)
        M_v = self.M - v

        suma = 0
        for i in range(self.nproj):         # TODO: szybsza itearcja po kolumnach macierzy...
            b = M_v[:,i]
            suma += min((b<=0).sum(), (b>0).sum())
            
        D_IRW = suma/(self.n * self.nproj)
        
        return D_IRW
 


