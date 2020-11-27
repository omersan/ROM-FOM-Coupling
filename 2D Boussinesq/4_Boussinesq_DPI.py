# -*- coding: utf-8 -*-
"""
Direct prolongation interface (DPI) approach for ROM-FOM coupling
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


def initial(nx,ny):
    #resting flow
    w = np.zeros([nx+1,ny+1])
    s = np.zeros([nx+1,ny+1])
    
    #masrigli flow [for temperature IC]
    t = np.zeros([nx+1,ny+1])
    t[:int(nx/2)+1,:] = 1.5
    t[int(nx/2)+1:,:] = 1
    
    return w,s,t


# time integration using third-order Runge Kutta method
def RK3t(rhs,nx,ny,dx,dy,Re,Pr,Ri,w,s,t,dt):
    aa = 1.0/3.0
    bb = 2.0/3.0

    tt = np.zeros([nx+1,ny+1])
    tt = np.copy(t)
    
    #stage-1
    rt = rhs(nx,ny,dx,dy,Re,Pr,Ri,w,s,t)
    tt[1:nx,1:ny] = t[1:nx,1:ny] + dt*rt
    tt = tbc(tt)

    #stage-2
    rt = rhs(nx,ny,dx,dy,Re,Pr,Ri,w,s,tt)
    tt[1:nx,1:ny] = 0.75*t[1:nx,1:ny] + 0.25*tt[1:nx,1:ny] + 0.25*dt*rt
    tt = tbc(tt)

    #stage-3
    rt = rhs(nx,ny,dx,dy,Re,Pr,Ri,w,s,tt)
    t[1:nx,1:ny] = aa*t[1:nx,1:ny] + bb*tt[1:nx,1:ny] + bb*dt*rt
    t = tbc(t)
    return t


def tbc(t):
    t[0,:] = t[1,:]
    t[-1,:] = t[-2,:]
    t[:,0] = t[:,1]
    t[:,-1] = t[:,-2]
    
    return t

def BoussRHS_t(nx,ny,dx,dy,Re,Pr,Ri,w,s,t):
    
    #t-equation
    rt = np.zeros([nx-1,ny-1])
    #laplacian terms
    Lt = laplacian(nx,ny,dx,dy,t)

    #Jacobian terms
    Jt = jacobian(nx,ny,dx,dy,t,s)
    rt = -Jt + (1/(Re*Pr))*Lt
    return rt

#compute velocity components from streamfunction (internal points)
def velocity(nx,ny,dx,dy,s):
    u =  np.zeros([nx-1,ny-1])
    v =  np.zeros([nx-1,ny-1])
    # u = ds/dy
    u = (s[1:nx,2:ny+1] - s[1:nx,0:ny-1])/(2*dy)
    # v = -ds/dx
    u = -(s[2:nx+1,1:ny] - s[0:nx-1,1:ny])/(2*dx)
    
    return u,v


def import_data(nx,ny,n):
    folder = 'data_'+ str(nx) + '_' + str(ny)              
    filename = './Results/'+folder+'/data_' + str(int(n))+'.npz'
    data = np.load(filename)
    w = data['w']
    s = data['s']
    t = data['t']
    return w,s,t


def PODproj_svd(u,Phi): #Projection
    a = np.dot(u.T,Phi)  # u = Phi * a.T
    return a

def PODrec_svd(a,Phi): #Reconstruction    
    u = np.dot(Phi,a.T)    
    return u

############################ GP Routines #####################################
# Galerkin Projection
# Right Handside of Galerkin Projection
def GROMrhs(nr, b_c, b_lw, b_lt, b_nl, a,b): 
    r1, r2, r3 = [np.zeros(nr) for _ in range(3)]
    
    r1 = b_c
    
    a = a.ravel()
    b = b.ravel()

    for k in range(nr):
        r2[k] = np.sum(b_lw[:,k]*a) + np.sum(b_lt[:,k]*b) 
    
    for k in range(nr):
        for i in range(nr):
            r3[k] = r3[k] + np.sum(b_nl[i,:,k]*a)*a[i]

    r = r1 + r2 + r3
    return r

# time integration using third-order Runge Kutta method
def M(nr, rhs, b_c, b_lw,b_lt, b_nl, a,b, dt):
    c1 = 1.0/3.0
    c2 = 2.0/3.0
    
    #stage-1
    r = rhs(nr, b_c, b_lw, b_lt, b_nl, a,b)
    a0 = a + dt*r
    
    #stage-2
    r = rhs(nr, b_c, b_lw, b_lt, b_nl, a0,b)
    a0 = 0.75*a + 0.25*a0 + 0.25*dt*r
    
    #stage-3
    r = rhs(nr, b_c, b_lw, b_lt, b_nl, a0,b)
    a = c1*a + c2*a0 + c2*dt*r
    
    return a   
            

def export_data_DPI(nx,ny,n,w,s,t):
    folder = 'data_'+ str(nx) + '_' + str(ny)       
    if not os.path.exists('./DPI/'+folder):
        os.makedirs('./DPI/'+folder)
        
    filename = './DPI/'+folder+'/data_' + str(int(n))+'.npz'
    np.savez(filename,w=w,s=s,t=t)
    
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
aTrue = data['aTrue']
bTrue = data['bTrue']

#%% Select the first nr
nr = 8
Phiw = Phiw[:,:nr]
Phis = Phis[:,:nr]
Phit = Phit[:,:nr]
aTrue = aTrue[:,:nr]
bTrue = bTrue[:,:nr]

#%% Load GP coefficients for w-equation
folder = 'data_'+ str(nx) + '_' + str(ny)       
filename = './POD/'+folder+'/GP_data_nr='+str(nr)+'.npz'
data = np.load(filename) 
b_c = data['b_c']
b_lw = data['b_lw']
b_lt = data['b_lt']
b_nl = data['b_nl']

#%% Initialize
nstart= 0
nend = nt
nstep = 1

ns = int((nend-nstart)/nstep)
print(ns)

aDPI = np.zeros([ns+1,nr])
bDPI = np.zeros([ns+1,nr])

w,s,t = import_data(nx,ny,nstart)
tmp = w.reshape([-1,])-wm
aDPI[0,:] = PODproj_svd(tmp,Phiw)

tmp = t.reshape([-1,])-tm
bDPI[0,:] = PODproj_svd(tmp,Phit)

time=0
for i in range(ns):
    time = time+dt
    
    tmp = t.reshape([-1,])-tm
    bDPI[i,:] = PODproj_svd(tmp,Phit)

    aDPI[i+1,:] = M(nr, GROMrhs, b_c, b_lw, b_lt, b_nl, aDPI[i,:], bDPI[i,:], dt).ravel()
        
    w = PODrec_svd(aDPI[i+1,:],Phiw) + wm
    s = PODrec_svd(aDPI[i+1,:],Phis) + sm

    w = w.reshape([nx+1,ny+1])
    s = s.reshape([nx+1,ny+1])
    t = RK3t(BoussRHS_t,nx,ny,dx,dy,Re,Pr,Ri,w,s,t,dt)

    if (i+1)%freq==0:
        export_data_DPI(nx,ny,nstart+i+1,w,s,t)
    
    u,v = velocity(nx,ny,dx,dy,s)
    umax = np.max(np.abs(u))
    vmax = np.max(np.abs(v))
    cfl = np.max([umax*dt/dx, vmax*dt/dy])
    
    if cfl >= 0.8:
        print('CFL exceeds maximum value')
        break
    
    if (i+1)%200==0:
        print(i+1, " ", time, " ", np.max(w), " ", cfl)
     
tmp = t.reshape([-1,])-tm
bDPI[i+1,:] = PODproj_svd(tmp,Phit)

#%% Save DATA
folder = 'data_'+ str(nx) + '_' + str(ny)       
filename = './DPI/'+folder+'/DPI_data_nr='+str(nr)+'.npz'
np.savez(filename,aDPI = aDPI, bDPI = bDPI) 

