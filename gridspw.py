# module for constructing quasi-momentum and lattice depth grids

import numpy as np
import scipy as sp
import scitools.numpyutils as scinp
import inputspw as inp

# --------- position, quasi-momentum, lattice depth, etc grids --------------
class Grids:
    """ 
    This class creates floating point 1D grid, with user defined 
    floats min, max, and step size. Also, we can simultaneously 
    create such independent 1D grids in 'dim' dimensions. 
    dim = 0 -> x-axis, dim = 1 -> y-axis, etc.
    """
    
    def __init__(self,min,max,step):
        self.min, self.max, self.step = min,max,step
        
    def __call__(self,dim):
        if isinstance(self.max[dim],float):
            return scinp.seq( self.min[dim],self.max[dim],self.step[dim] )
        else:
            return scinp.iseq( self.min[dim],self.max[dim],self.step[dim] )
# -------------------------- xxx -------------------- grids
