
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 17:08:14 2023

@author: sahar
"""

import numpy
import scipy
import scipy.optimize
from scipy import integrate
from scipy.interpolate import interp1d
import string
import matplotlib
import matplotlib.pyplot as plt
import os
import sys

plt.rcParams['figure.figsize'] = [12, 8]

# This script solves the coupled transport equations
#  for the diffusion of He in a thermal gradient

# Function for constant boundaries and time evolutioninteraction parameters
def interModel(EcohA, EcohB, Omega, z):

    e11 = 2.0*EcohA / z
    e22 = 2.0*EcohB / z
    e12 = 0.5*(2.0*Omega/z + e11 + e22)
    
    return e11,e22,e12

# Total energy of the system
def energy(atypes,e11, e22, e12,nx):

    E = 0
    for i in range(0,nx):
        for j in range(0,nx):

            itype = atypes[i,j]

            # Neighbor i+1
            if(i+1 >= nx):
                jtype = atypes[0,j]
            elif(i+1 < 0):
                jtype = atypes[nx-1,j]
            else:
                jtype = atypes[i+1,j]
            if(itype == 1 and itype == jtype):
                E += e11
            elif(itype == 2 and itype == jtype):
                E += e22
            else:
                E += e12

            # Neighbor i-1
            if(i-1 >= nx):
                jtype = atypes[0,j]
            elif(i-1 < 0):
                jtype = atypes[nx-1,j]
            else:
                jtype = atypes[i-1,j]
            if(itype == 1 and itype == jtype):
                E += e11
            elif(itype == 2 and itype == jtype):
                E += e22
            else:
                E += e12

            # Neighbor j+1
            if(j+1 >= nx):
                jtype = atypes[i,0]
            elif(j+1 < 0):
                jtype = atypes[i,nx-1]
            else:
                jtype = atypes[i,j+1]
            if(itype == 1 and itype == jtype):
                E += e11
            elif(itype == 2 and itype == jtype):
                E += e22
            else:
                E += e12

            # Neighbor j-1
            if(j-1 >= nx):
                jtype = atypes[i,0]
            elif(j-1 < 0):
                jtype = atypes[i,nx-1]
            else:
                jtype = atypes[i,j-1]
            if(itype == 1 and itype == jtype):
                E += e11
            elif(itype == 2 and itype == jtype):
                E += e22
            else:
                E += e12

    return E
    
def do_mmc(atypes,e11,e22,e12,kb,T,nx,naccept,nreject):

    # compute initial energy
    ienergy = energy(atypes,e11,e22,e12,nx)
    
    # Pick two atoms at random with different types
    xi = numpy.random.randint(0,nx)
    yi = numpy.random.randint(0,nx)
    itype = atypes[xi,yi]
    jtype = 0
    xf = -1
    yf = -1
    while(True):
        xf = numpy.random.randint(0,nx)
        yf = numpy.random.randint(0,nx)
        if(atypes[xf,yf] != itype):
            jtype = atypes[xf,yf]
            break

    # Exchange types
    atypes[xi,yi] = jtype
    atypes[xf,yf] = itype

    # Compute final energy
    fenergy = energy(atypes,e11,e22,e12,nx)

    # accept or reject according to Boltzmann
    #  draw a random number and check if it is larger than
    #  the Boltzmann factor. If it is, reject the move
    #  accept otherwise.
    rnd =  numpy.random.uniform(0,1.0)
    deltaE = fenergy - ienergy
    if(rnd > numpy.exp(-deltaE/(kb*T))):
        atypes[xi,yi] = itype
        atypes[xf,yf] = jtype
        nreject += 1
    else:
        naccept += 1
    
    return atypes,naccept,nreject

def main():

    # seed random number generator
     # if len(sys.argv) > 1:
     seed = 100
     # else:
    # seed = 5000000  # or any other default value
     numpy.random.seed(seed)
    
    # dimension
     dimension = 2

    # rod size, nm
     nx = 50

    # Number of neighbors
     z = 4
    
    # order energy (eV)
     Omega = 0.08

    # Cohesive energies
     EcohA = -4.0
     EcohB = -3.0

    # Interaction parameters
     e11,e22,e12 = interModel(EcohA, EcohB, Omega, z)

    # Initial concentration of B     
     c0 = 0.8     #changed

    #Temperature (K)
     T = 1400

    # Boltzmann constant (eV/K)
     kb = 0.00008617
     
     #chemical concentration
     mu_A=EcohA-kb*T
     mu_A=EcohA-kb*T

    # Files for output
     floutC = open("confAB.dump","w")
     floutA = open("acceptance.txt","w")
     name = "step\tNumberAccept\tNumberReject\n"
     floutA.write(name)
    
     naccept = 0
     nreject = 0
     atypes = numpy.zeros((nx,nx))     # concentration

    # Initialize atomic types at random
     for i in range(0,nx):
        for j in range(0,nx):
            rnd = numpy.random.uniform(0,1.0)
            if(rnd < c0):
                atypes[i,j] = 2
            else:
                atypes[i,j] = 1
                
    # Number of timesteps
     nsteps = 10000000

    # Print values every these many steps
     printSteps = 100
     fignum = 0
     nprint = numpy.arange(0,nsteps,printSteps)
     for m in range(0,nsteps):
        if m in nprint:
            # Print configuration
            name = "ITEM: TIMESTEP\n"
            floutC.write(name)
            name = "%d\n" % (m)
            floutC.write(name)
            name = "ITEM: NUMBER OF ATOMS\n"
            floutC.write(name)
            name = "%d\n" % (nx*nx)
            floutC.write(name)
            name = "ITEM: BOX BOUNDS pp pp pp\n"
            floutC.write(name)
            name = "%f\t%f\n" % (0,nx)
            floutC.write(name)
            name = "%f\t%f\n" % (0,nx)
            floutC.write(name)
            name = "%f\t%f\n" % (-1,1)
            floutC.write(name)
            name = "ITEM: ATOMS id type x y z\n"
            floutC.write(name)
            for i in range(nx):
                for j in range(nx):
                    name = "%d\t%d\t%f\t%f\t%f\n" % (i*nx+j,atypes[i,j],i,j,0.0)
                    floutC.write(name)

            name = "%d\t%d\t%d\n" % (m,naccept,nreject)
            floutA.write(name)
            
        atypes,naccept,nreject = do_mmc(atypes,e11,e22,e12,kb,T,nx,naccept,nreject)

        floutC.close()
if __name__ == "__main__":
    main()
