# -*- coding: utf-8 -*-
"""
Basis construction for the Marsigli flow problem governed by the 
2D Boussinesq equations
This correpsonds to Example 2 for the following paper:
    "Multifidelity computing for coupling full and reduced order models",
     PLOS ONE, 2020
     
For questions, comments, or suggestions, please contact Shady Ahmed,
PhD candidate, School of Mechanical and Aerospace Engineering, 
Oklahoma State University. @ shady.ahmed@okstate.edu
last checked: 11/27/2020
"""


#%% Import libraries
import numpy as np
import os
from scipy.fftpack import dst, idst
from numpy import linalg as LA

#%% Define functions


# compute jacobian using arakawa scheme
# computed at all internal physical domain points (1:nx-1,1:ny-1)
def jacobian(nx,ny,dx,dy,q,s):
    gg = 1.0/(4.0*dx*dy)
    hh = 1.0/3.0
    #Arakawa 1:nx,1:ny   
    j1 = gg*( (q[2:nx+1,1:ny]-q[0:nx-1,1:ny])*(s[1:nx,2:ny+1]-s[1:nx,0:ny-1]) \
             -(q[1:nx,2:ny+1]-q[1:nx,0:ny-1])*(s[2:nx+1,1:ny]-s[0:nx-1,1:ny]))

    j2 = gg*( q[2:nx+1,1:ny]*(s[2:nx+1,2:ny+1]-s[2:nx+1,0:ny-1]) \
            - q[0:nx-1,1:ny]*(s[0:nx-1,2:ny+1]-s[0:nx-1,0:ny-1]) \
            - q[1:nx,2:ny+1]*(s[2:nx+1,2:ny+1]-s[0:nx-1,2:ny+1]) \
            + q[1:nx,0:ny-1]*(s[2:nx+1,0:ny-1]-s[0:nx-1,0:ny-1]))
    
    j3 = gg*( q[2:nx+1,2:ny+1]*(s[1:nx,2:ny+1]-s[2:nx+1,1:ny]) \
            - q[0:nx-1,0:ny-1]*(s[0:nx-1,1:ny]-s[1:nx,0:ny-1]) \
            - q[0:nx-1,2:ny+1]*(s[1:nx,2:ny+1]-s[0:nx-1,1:ny]) \
            + q[2:nx+1,0:ny-1]*(s[2:nx+1,1:ny]-s[1:nx,0:ny-1]) )
    jac = (j1+j2+j3)*hh
    return jac

def laplacian(nx,ny,dx,dy,w):
    aa = 1.0/(dx*dx)
    bb = 1.0/(dy*dy)
    lap = aa*(w[2:nx+1,1:ny]-2.0*w[1:nx,1:ny]+w[0:nx-1,1:ny]) \
        + bb*(w[1:nx,2:ny+1]-2.0*w[1:nx,1:ny]+w[1:nx,0:ny-1])
    return lap    


#Elliptic coupled system solver:
#For 2D Boussinesq equation:
def poisson_fst(nx,ny,dx,dy,w):

    f = np.zeros([nx-1,ny-1])
    f = np.copy(-w[1:nx,1:ny])

    #DST: forward transform
    ff = np.zeros([nx-1,ny-1])
    ff = dst(f, axis = 1, type = 1)
    ff = dst(ff, axis = 0, type = 1) 
    
    m = np.linspace(1,nx-1,nx-1).reshape([-1,1])
    n = np.linspace(1,ny-1,ny-1).reshape([1,-1])
    
    alpha = (2.0/(dx*dx))*(np.cos(np.pi*m/nx) - 1.0) + (2.0/(dy*dy))*(np.cos(np.pi*n/ny) - 1.0)           
    u1 = ff/alpha
        
    #IDST: inverse transform
    u = idst(u1, axis = 1, type = 1)
    u = idst(u, axis = 0, type = 1)
    u = u/((2.0*nx)*(2.0*ny))

    ue = np.zeros([nx+1,ny+1])
    ue[1:nx,1:ny] = u
    
    return ue

def import_data(nx,ny,n):
    folder = 'data_'+ str(nx) + '_' + str(ny)              
    filename = './Results/'+folder+'/data_' + str(int(n))+'.npz'
    data = np.load(filename)
    w = data['w']
    s = data['s']
    t = data['t']
    return w,s,t

def POD_svd(nx,ny,dx,dy,nstart,nend,nstep,nr):
    ns = int((nend-nstart)/nstep)
    #compute temporal correlation matrix
    Aw = np.zeros([(nx+1)*(ny+1),ns+1]) #vorticity
    At = np.zeros([(nx+1)*(ny+1),ns+1]) #temperature
    ii = 0
    for i in range(nstart,nend+1,nstep):
        w,s,t = import_data(nx,ny,i)
        Aw[:,ii] = w.reshape([-1,])
        At[:,ii] = t.reshape([-1,])
        ii = ii + 1
    
    #mean subtraction
    wm = np.mean(Aw,axis=1)
    tm = np.mean(At,axis=1)

    Aw = Aw - wm.reshape([-1,1])
    At = At - tm.reshape([-1,1])
    
    #singular value decomposition
    Uw, Sw, Vhw = LA.svd(Aw, full_matrices=False)
    Ut, St, Vht = LA.svd(At, full_matrices=False)
   
    Phiw = Uw[:,:nr]  
    Lw = Sw**2
    #compute RIC (relative importance index)
    RICw = sum(Lw[:nr])/sum(Lw)*100   

    Phit = Ut[:,:nr]  
    Lt = St**2
    #compute RIC (relative importance index)
    RICt = sum(Lt[:nr])/sum(Lt)*100   
    
    return wm,Phiw,Lw/sum(Lw),RICw , tm,Phit,Lt/sum(Lt),RICt 

def PODproj_svd(u,Phi): #Projection
    a = np.dot(u.T,Phi)  # u = Phi * a.T if shape of a is [ns,nr]
    return a

def PODrec_svd(a,Phi): #Reconstruction    
    u = np.dot(Phi,a.T)    
    return u

#%% Main program
# Inputs
lx = 8
ly = 1
nx = 4096
ny = int(nx/8)

Re = 1e4
Ri = 4
Pr = 1

Tm = 8
dt = 5e-4
nt = np.int(np.round(Tm/dt))

ns = 800
freq = np.int(nt/ns)

#%% grid

dx = lx/nx
dy = ly/ny

x = np.linspace(0.0,lx,nx+1)
y = np.linspace(0.0,ly,ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

#%% POD basis generation
nstart= 0
nend = nt
nstep = freq
nr = 50 #number of basis to store [we might not need to *use* all of them]
#compute  mean field and basis functions for potential voriticity
wm,Phiw,Lw,RICw , tm,Phit,Lt,RICt  = POD_svd(nx,ny,dx,dy,nstart,nend,nstep,nr)


#%% Compute Streamfunction mean and basis functions
# from those of potential vorticity using Poisson equation

tmp = wm.reshape([nx+1,ny+1])
tmp = poisson_fst(nx,ny,dx,dy,tmp)
sm = tmp.reshape([-1,])

Phis = np.zeros([(nx+1)*(ny+1),nr])
for k in range(nr):
    tmp = np.copy(Phiw[:,k]).reshape([nx+1,ny+1])
    tmp = poisson_fst(nx,ny,dx,dy,tmp)
    Phis[:,k] = tmp.reshape([-1,])
    
    
#%% compute true modal coefficients
nstart= 0
nend = nt
nstep = freq

ns = int((nend-nstart)/nstep)

aTrue = np.zeros([ns+1,nr])
bTrue = np.zeros([ns+1,nr])

ii = 0
for i in range(nstart,nend+1,nstep):
    w,s,t = import_data(nx,ny,i)
    tmp = w.reshape([-1,])-wm
    aTrue[ii,:] = PODproj_svd(tmp,Phiw)
    
    tmp = t.reshape([-1,])-tm
    bTrue[ii,:] = PODproj_svd(tmp,Phit)

    ii = ii + 1
    
    if ii%100 == 0:
        print(ii)


#%% Save data
folder = 'data_'+ str(nx) + '_' + str(ny)       
if not os.path.exists('./POD/'+folder):
    os.makedirs('./POD/'+folder)

filename = './POD/'+folder+'/POD_data.npz'
np.savez(filename, wm = wm, Phiw = Phiw,  sm = sm, Phis = Phis,  \
                   tm = tm, Phit = Phit, \
                   aTrue = aTrue, bTrue = bTrue,\
                   Lw = Lw, Lt = Lt)
         