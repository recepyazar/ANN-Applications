import numpy as np
import random
import math
import pickle

inDim = 1
hidDim = 8
outDim = 1

inDim_ = 2
hidDim_ = 5
outDim_ = 1


in_weights = np.random.normal(0,(inDim*hidDim)**(-0.8),(hidDim,inDim))
out_weights = np.random.normal(0,(hidDim*outDim)**(-0.8),(outDim,hidDim))
con_weights = np.zeros((hidDim,hidDim))
context_layer = np.zeros((hidDim,outDim))

in_weights_old = np.random.normal(0,(inDim_*hidDim_)**(-0.8),(hidDim_,inDim_))
out_weights_old = np.random.normal(0,(hidDim_*outDim_)**(-0.8),(outDim_,hidDim_))

f = open("tursu","wb") 
pickle.dump(in_weights,f)     
pickle.dump(out_weights,f)
pickle.dump(con_weights,f) 
pickle.dump(context_layer,f) 
f.close()
f = open("tursu_old","wb")   
pickle.dump(in_weights_old,f)     
pickle.dump(out_weights_old,f)
f.close()