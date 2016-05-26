# Hamiltonian module
import numpy as np
import scipy.integrate as spintegrate
import scitools.numpyutils as scinp
import inputspw as inp
import gridspw as grd


# -------------------------- inputs -------------------------------
kL = inp.kL
b = inp.b
gmax = inp.gmax
gstep = inp.gstep
npw = 2*gmax + 1
pwbasis = grd.Grids(-gmax,gmax,gstep)
# ------------------------ xxx ------------------- inputs


# ---------------------- optical lattice ------------------------
LatticeVector = np.array([ np.pi/kL, np.pi/(2.0*kL), np.pi/(2.0*kL) ])
ReciprocalVector = 2.0*np.pi/LatticeVector
BZ = 0.5*ReciprocalVector
# ------------------------ xxx --------------- optical lattice


# ------------------- Hamiltonian in plane wave basis ------------------
class Hamiltonian:
    def __init__(self,k,LatticeDepth):
        self.k = k
        self.LatticeDepth = LatticeDepth

    # constructing the lower triangle of the Hermitian matrix
    # by first constructing the diagonal arrays
    # and then the H-matrix is constructed by using np.diag
    # this removes element wise operations -> faster code (vectorize)
    def hxy(self):
        k = self.k
        LatticeDepth = self.LatticeDepth
        keq = k/kL
        v0 = LatticeDepth[0]
        v1 = LatticeDepth[1]
        v2 = LatticeDepth[2]
        keq = k/kL
        cnst = 4.0*kL*b

        diaghx0 = ( -0.5*(v0+v1) + keq**2 )*np.ones(npw[0],float) \
                  + 4.0*keq*pwbasis(0) + 4.0*np.square(pwbasis(0))  

        diaghx1 = -0.25*v0*np.ones(npw[0]-1,float)
        
        tmpx = -0.25*v1*complex( np.cos(cnst),np.sin(cnst) )
        diaghx2 = tmpx*np.ones(npw[0]-2,complex)
        
        hx = np.diag(diaghx0) + np.diag(diaghx1,-1) + np.diag(diaghx2,-2)
        
        diaghy0 = ( -0.5*v2 + keq**2 )*np.ones(npw[1],float) \
                  + 8.0*keq*pwbasis(1) + 16.0*np.square(pwbasis(1))
        
        diaghy1 = -0.25*v2*np.ones(npw[1]-1,float)
        hy = np.diag(diaghy0) + np.diag(diaghy1,-1)
        
        return hx, hy

    def __call__(self,dim):
        hx, hy = Hamiltonian.hxy(self)
        if dim==0:
            return hx
        elif dim==1 or dim==2:
            return hy
# ---------------------- xxx ------------ Hamiltonian


# ------------------------ tunneling energies -----------------------
class Tunneling:
    def __init__(self,energy,kgrid,a):
        # input energy is the band energy array, \epsilon_\alpha(k)
        # where,alpha is the band index
        # and, k is the quasi-momentum
        # kgrid is the quasi-momentum grid 
        # on which we have the band energy
        # all the above are entered for a particular axis 'dim'
        # a is the lattice spacing along axis 'dim'
        self.energy = energy
        self.kgrid = kgrid
        self.a = a
        
    def hop(self):
        nk = self.kgrid.shape[0]
        hopping = np.zeros(nk)
        for n in xrange(nk):
            tmp1 = np.cos(n*self.kgrid*self.a)
            tmp = np.multiply( tmp1,self.energy )
            hopping[n] = -0.5*(tmp[0]+tmp[nk-1])+np.sum( tmp )
            
            # if using scipy.integrate (slower!)
            # \sum_k -> L*\int_{-pi/a}^{pi/a}dk/2pi
            # thus, \sum_k -> (M/2)*\int_{-1}^{1}dk
            # thus, (1/M)\sum_k -> (1/2)*\int_{-1}^{1}dk
            # hence, we have the prefactor of 0.5
            # hopping[n] = 0.5*spintegrate.simps(tmp,self.kgrid)

        hopping = -(1.0/float(nk-1))*hopping
        # each element 'n' of the hop gives the n^th neighbor hop
        # n=0, i.e., the first element is the avg energy of any band
        return hopping
        
    def __call__(self):
        hop = Tunneling.hop(self)
        return hop
# ------------------------ xxx ---------------- hopping




