from __future__ import division
import numpy as np
import sys
import pandas as pd

data = pd.read_csv(sys.argv[1])
data = data.pivot(index = data.columns[0], columns = data.columns[1], 
                  values = data.columns[2])
M_mask = ~data.isnull().as_matrix()
train_data = data.as_matrix()

lam = 2
sigma2 = 0.1
d = 5
iterations = 50

# Implement function here
def PMF(train_data, M_mask, lam, sigma2, d, iterations):
    L  = np.zeros(iterations)
    Nu = train_data.shape[0]
    Nv = train_data.shape[1]
    U  = np.zeros((iterations, Nu, d))
    V  = np.zeros((iterations, Nv, d))
    mean = np.zeros(d)
    cov  = (1/lam) * np.identity(d)
    V[0] = np.random.multivariate_normal(mean, cov, Nv)
    
    for itr in range(iterations):
        
        if itr == 0:
            l = 0
        else:
            l = itr - 1
        
        for i in range(Nu):
            Z1 = lam * sigma2 * np.identity(d)
            Z2 = np.zeros(d)
            for j in range(Nv):
                if train_data[i, j] == True:
                    Z1 += np.outer(V[l, j], V[l, j]) #movie rated by the user
                    Z2 += train_data[i, j] * V[l, j]    
            
            U[itr, i] = np.dot(np.linalg.inv(Z1), Z2)
        
        
        for j in range(Nv):
            Z1 = lam * sigma2 * np.identity(d)
            Z2  = np.zeros(d)
            for i in range(Nu):
                if train_data[i, j] == True:
                    Z1 += np.outer(U[itr, i], U[itr, i])
                    Z2 += train_data[i, j] * U[itr, i]
            
            V[itr, j] = np.dot(np.linalg.inv(Z1), Z2)
            
        temp = 0 
        for i in range(Nu):
            
            for j in range(Nv):
                
                if train_data[i ,j] == True:
                    temp -= np.square(train_data[i ,j] - np.dot(U[itr, i].T, V[itr, j]))
                
                
            
        temp = (1/(2*sigma2)) * temp
        
        temp -= (lam/2) * (np.square(np.linalg.norm(U[itr])) + np.square(np.linalg.norm(V[itr])))
        
        L[itr] = temp
               zrzd 
    return L, U, V
# Assuming the PMF function returns Loss L, U_matrices and V_matrices (refer to lecture)
L, U_matrices, V_matrices = PMF(train_data, M_mask, lam, sigma2, d, iterations)

np.savetxt("objective.csv", L, delimiter=",")

np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
np.savetxt("U-50.csv", U_matrices[49], delimiter=",")

np.savetxt("V-10.csv", V_matrices[9], delimiter=",")
np.savetxt("V-25.csv", V_matrices[24], delimiter=",")
np.savetxt("V-50.csv", V_matrices[49], delimiter=",")
