# -*- coding: utf-8 -*-
"""
Calculation of tensorial Galerkin ROM coefficients for vorticity equation
of the Marsigli flow problem governed by the 2D Boussinesq equations
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


def const_term(nr,wm,sm,tm,Phiw,Re,Ri):
    #jacobian
    w = wm.reshape([nx+1,ny+1])
    s = sm.reshape([nx+1,ny+1])
    tmp1 = -jacobian(nx,ny,dx,dy,w,s)
        
    #laplacian
    w = wm.reshape([nx+1,ny+1])
    tmp2 = (1/Re)*laplacian(nx,ny,dx,dy,w)
    
    #conduction
    dd = 1.0/(2.0*dx)
    t = tm.reshape([nx+1,ny+1])
    tmp3 = Ri*dd*(t[2:nx+1,1:ny]-t[0:nx-1,1:ny])
    
    #compute constant term
    b_c = np.zeros(nr)
    for k in range(nr):
        tmp = np.zeros([nx+1,ny+1])
        tmp[1:nx,1:ny] = (tmp1 + tmp2 + tmp3)
        tmp = tmp.reshape([(nx+1)*(ny+1),])
        b_c[k] = tmp.T @ Phiw[:,k]
    
    return b_c

def lin_term(nr,wm,sm,tm,Phiw,Phis,Phit,Re):
    b_lw = np.zeros([nr,nr])
    b_lt= np.zeros([nr,nr])

    for i in range(nr):
        #L1
        w = np.copy(Phiw[:,i].reshape([nx+1,ny+1]))
        tmp1 = (1/Re)*laplacian(nx,ny,dx,dy,w)
        
        w = np.copy(Phiw[:,i].reshape([nx+1,ny+1]))
        s = np.copy(sm.reshape([nx+1,ny+1]))
        tmp2 = -jacobian(nx,ny,dx,dy,w,s)
        
        w = np.copy(wm.reshape([nx+1,ny+1]))
        s = np.copy(Phis[:,i].reshape([nx+1,ny+1]))
        tmp3 = -jacobian(nx,ny,dx,dy,w,s)
        
        #L2
        dd = 1.0/(2.0*dx)
        t = np.copy(Phit[:,i].reshape([nx+1,ny+1]))
        tmp4 = Ri*dd*(t[2:nx+1,1:ny]-t[0:nx-1,1:ny])
        
        for k in range(nr):
            tmp = np.zeros([nx+1,ny+1])
            tmp[1:nx,1:ny] = np.copy(tmp1+tmp2+tmp3)
            tmp = tmp.reshape([(nx+1)*(ny+1),])
            b_lw[i,k] = tmp.T @ Phiw[:,k]   

            tmp = np.zeros([nx+1,ny+1])
            tmp[1:nx,1:ny] = np.copy(tmp4)
            tmp = tmp.reshape([(nx+1)*(ny+1),])
            b_lt[i,k] = tmp.T @ Phiw[:,k]  
            
    return b_lw, b_lt

def nonline_term(nr,Phiw,Phis,nx,ny):
    b_nl = np.zeros([nr,nr,nr])
    
    for i in range(nr):
        w = np.copy(Phiw[:,i].reshape([nx+1,ny+1]))
        for j in range(nr):
            s = np.copy(Phis[:,j].reshape([nx+1,ny+1]))
            tmp1 = -jacobian(nx,ny,dx,dy,w,s)
            tmp = np.zeros([nx+1,ny+1])
            tmp[1:nx,1:ny] = np.copy(tmp1)
            for k in range(nr):
                tmp = tmp.reshape([(nx+1)*(ny+1),])
                b_nl[i,j,k] = tmp.T @ Phiw[:,k]   
    return b_nl


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

#%% Load POD data

folder = 'data_'+ str(nx) + '_' + str(ny)       
filename = './POD/'+folder+'/POD_data.npz'
data = np.load(filename)

wm = data['wm']        
Phiw = data['Phiw']
sm = data['sm']        
Phis = data['Phis']
tm = data['tm']        
Phit = data['Phit']

#%% Select the first nr
nr = 8
Phiw = Phiw[:,:nr]
Phis = Phis[:,:nr]
Phit = Phit[:,:nr]

#%% Compute GP coefficients for w-equation
b_c = const_term(nr,wm,sm,tm,Phiw,Re,Ri)
b_lw, b_lt = lin_term(nr,wm,sm,tm,Phiw,Phis,Phit,Re)
b_nl = nonline_term(nr,Phiw,Phis,nx,ny)

#%% Save data
folder = 'data_'+ str(nx) + '_' + str(ny)       
filename = './POD/'+folder+'/GP_data_nr='+str(nr)+'.npz'
np.savez(filename, b_c = b_c, b_lw = b_lw, b_lt = b_lt, b_nl = b_nl) 
         

