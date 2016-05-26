# module giving input parameters 
import numpy as np

kL = 1.0
b = 0.275*np.pi
gmax = np.array([15, 15, 15])
gstep = np.array([1, 1, 1])
ratiox = 1.5

Vmin = np.array([35.0, 15.0, 15.0])
Vmax = np.array([35.0, 15.0, 15.0])
Vstep = np.array([0.5, 0.5, 0.5])
Vchosen = np.array([35.0, 15.0, 15.0])

dk = np.array([0.01/kL, 0.02/kL, 0.02/kL])

nband = np.array([3, 3, 3])    
