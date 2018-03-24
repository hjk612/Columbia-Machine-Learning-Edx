from __future__ import division
import numpy as np
import sys
from scipy.stats import multivariate_normal

X_train = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2])
X_test = np.genfromtxt(sys.argv[3], delimiter=",")

def pluginClassifier(X_train, y_train, X_test): 
    N = len(set(y_train))
    prior_probs = [y_train.tolist().count(val)/N for val in set(y_train)]
    class_mean = []
    class_cov = []
    class_conditional = []
    for val in set(y_train):
        class_mean.append(np.mean(X_train[np.where(y_train==val)[0]],axis=0))
        class_cov.append(np.cov(X_train[np.where(y_train==val)[0]],rowvar=0))
        class_conditional.append(multivariate_normal(mean=class_mean[-1],cov=class_cov[-1]))
    posterior_prob = []
    for test in X_test:
        numerator = [prior_probs[i]*class_conditional[i].pdf(test) for i in range(N)]
        prob = [numerator[i]/sum(numerator) for i in range(N)]
        posterior_prob.append(prob)
    
    return np.array(posterior_prob)
 

final_outputs = pluginClassifier(X_train, y_train, X_test) # assuming final_outputs is returned from function

np.savetxt("probs_test.csv", final_outputs, delimiter=",") # write output to file