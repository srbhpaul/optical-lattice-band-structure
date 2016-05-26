#!/usr/bin/env python
import numpy as np
import scipy.linalg as linalg
import scipy.integrate as integrate
import scitools.numpyutils as scinp
import inputspw as inp
import hamiltonianpw as ham
import gridspw as grd


# ------------------------ inputs and paramters --------------------
kL = inp.kL
a = ham.LatticeVector
b = inp.b
ratiox = inp.ratiox
Vchosen = inp.Vchosen
Vmin = inp.Vmin
Vmax = inp.Vmax
Vstep = inp.Vstep
npw = ham.npw
nband = inp.nband

Vgrid = grd.Grids(Vmin,Vmax,Vstep)
qmgrid = grd.Grids(-ham.BZ,ham.BZ,inp.dk)
Vichosen = np.array([0, 0, 0])
for j in range(3):
    if Vchosen[j]<Vmin[j] or Vchosen[j]>Vmax[j]:
        Vchosen[j] = 0.5*(Vmin[j]+Vmax[j])
        Vichosen[j] = np.floor( (Vchosen[j]-Vmin[j])/Vstep[j] )
    else:
        Vichosen[j] = np.floor( (Vchosen[j]-Vmin[j])/Vstep[j] )
for j in range(3):
    Vchosen[j] = Vgrid(j)[Vichosen[j]]
# -------------------------- xxx -------------------- input parameters


# ------------------------ Energy Bands ---------------------------
for j in xrange(3):
    ev = np.zeros((3,Vgrid(j).shape[0],qmgrid(j).shape[0],npw[j]))

for j in xrange(3): 
    for Vi in xrange( Vgrid(j).shape[0] ): # range(30,31):
        v0 = Vgrid(j)[Vi]
        v1 = ratiox*v0
        v2 = Vgrid(j)[Vi]
        LatticeDepth = np.array([ v0, v1, v2 ])
        # note: if j/=1, v0,v1 as obtained above will be wrong  
        # however, it doesn't matter since for j/=1, 
        # hamiltonian H(y,z) doesn't depend on them
        # similarly, v2 will be wrong when j=1
        # again, H(x) is independent of v2, so it doesn't matter
        for ki in xrange( qmgrid(j).shape[0] ):
            k = qmgrid(j)[ki]
            h = ham.Hamiltonian(k,LatticeDepth)
            hevalues = linalg.eigvalsh(h(j))
            hevalues = hevalues.real 
            ev[j,Vi,ki,:] = hevalues
# ---------------------- xxx ----------------- energy bands


# ------------------------ tunneling energies -----------------------
for j in xrange(3):
    hop = np.zeros((3,Vgrid(j).shape[0],nband[j],qmgrid(j).shape[0]))

for j in xrange(3):
    for Vi in xrange( Vgrid(j).shape[0] ):
        for band in xrange( nband[j] ):
            energy = ev[j,Vi,:,band]
            kgrid = qmgrid(j)
            hop[j,Vi,band,:] = ham.Tunneling(energy,kgrid,a[j])()
# ------------------------ xxx ---------------- hopping

#print hop[0,Vichosen[0],0]

# ------------------------ print energy bands -----------------------
title = ["xenergybandpw.out","yenergybandpw.out","zenergybandpw.out" ]

for j in xrange(3):
    outfile = open(title[j],"w")
    outfile.write ("#   lattice symmetry parameter b = %6.4f \n" %b)
    outfile.write ("#   V1/V0 = %6.3f \n" %ratiox)
    outfile.write ("#   Lattice depths =  ")
    for i in xrange(3):
        outfile.write ( "%8.3f" %(Vchosen[i]), )
    outfile.write("\n")
    outfile.write("# \n")
    outfile.write ("%s" %("#    k            band1                band2\
            band3 ...\n"))
    for ki in xrange( qmgrid(j).shape[0] ):
        outfile.write( "%8.3f" %(qmgrid(j)[ki]), )
        for pwi in xrange(nband[j]):
            outfile.write("%21.10e" %(ev[j,Vichosen[j],ki,pwi]), )
        outfile.write("\n")
    outfile.close()
# ---------------------------- xxx --------------- energy bands print


