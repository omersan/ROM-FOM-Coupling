# -*- coding: utf-8 -*-
"""
ROM-FOM coupling for the 1D Burgers problem with zonal heterogenity
This correpsonds to Example 1 for the following paper:
    "Multifidelity computing for coupling full and reduced order models",
     PLOS ONE, 2020
     
For questions, comments, or suggestions, please contact Shady Ahmed,
PhD candidate, School of Mechanical and Aerospace Engineering, 
Oklahoma State University. @ shady.ahmed@okstate.edu
last checked: 11/27/2020
"""

#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.linalg import block_diag

import os
import sys
#%% Define Functions

#-----------------------------------------------------------------------------!
#compute rhs for numerical solutions
#  r = -u*u' + nu*u''
#-----------------------------------------------------------------------------!
def rhsR(nx,dx,nu1,nu2,g1,g2,nxb,u):
    r = np.zeros(nx-nxb+1)

    r[1:nx-nxb] = (nu2/(dx*dx))*(u[2:nx-nxb+1] - 2.0*u[1:nx-nxb] + u[0:nx-nxb-1]) \
                 - g2*u[1:nx-nxb]\
                 - (1.0/3.0)*(u[2:nx-nxb+1]+u[0:nx-nxb-1]+u[1:nx-nxb])*(u[2:nx-nxb+1]-u[0:nx-nxb-1])/(2.0*dx)
    return r


#%% Main program:
    
# Inputs
nx =  4*1024  #spatial resolution
lx = 1.0    #spatial domain
dx = lx/nx
x = np.linspace(0, lx, nx+1)

nu1 = 1e-2   #control dissipation
nu2 = 1e-4   #control dissipation
g1 = 0 #friction
g2 = 1 #friction

tm = 2      #maximum time
dt = 2.5e-4   #solver timestep
nt = round(tm/dt)
t = np.linspace(0, tm, nt+1)

ns = 8000 #number of snapshots to save

#%% Reading data
nxb= int(6*nx/8)

nr = 8
data = np.load('./Data/uFOM_xb='+str(nxb/nx)+'_.npy')
uFOM = data
data = np.load('./ROM_Truncated/uPOD_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')
uTPI = data[nxb,:]
data = np.load('./ROM_Truncated/uDPI_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')
uDPI = data[nxb,:]
data = np.load('./ROM_Truncated/uPCI_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')
uPCI = data[:,0]
data = np.load('./ROM_Truncated/uCPI_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')
uCPI = data[nxb,:]
data = np.load('./ROM_Truncated/uUPI_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')
uUPI = data[nxb,:]



#%%
    
uTP = np.zeros((nx-nxb+1,ns+1))
uDP = np.zeros((nx-nxb+1,ns+1))
uPC = np.zeros((nx-nxb+1,ns+1))
uCP = np.zeros((nx-nxb+1,ns+1))
uUP = np.zeros((nx-nxb+1,ns+1))

#compute initial conditions
uu = np.zeros(nx-nxb+1)       
aa = 0.75
bb = 0.005
uu = 0.5-0.5*np.tanh((x[nxb:]-aa)/bb)

# boundary conditions: b.c.
u1 = np.zeros(nx-nxb+1)
uTP[0,:] = uTPI
uDP[0,:] = uDPI
uPC[0,:] = uPCI
uCP[0,:] = uCPI
uUP[0,:] = uUPI

uu[-1] = 0.0
u1[-1]= 0.0       
    
#check for stability
neu = np.max([nu2])*dt/(dx*dx)
cfl = np.max(uu)*dt/dx
if (neu >= 0.5):
    print('Neu condition: reduce dt')
    sys.exit()
if (cfl >=  1.0):
    print('CFL condition: reduce dt')
    sys.exit()
      
#time integration
uTP[:,0] = uu 
uDP[:,0] = uu 
uPC[:,0] = uu 
uCP[:,0] = uu 
uUP[:,0] = uu 

#%
for jj in range(1,ns+1):
    
    # TP
    #RK3 scheme
    # first step    
    uu = np.copy(uTP[:,jj-1])    
    u1 = np.copy(uTP[:,jj-1])  
    rr = rhsR(nx,dx,nu1,nu2,g1,g2,nxb,uu)
    u1[1:-1] = uu[1:-1] + dt*rr[1:-1]
    
    # second step
    rr = rhsR(nx,dx,nu1,nu2,g1,g2,nxb,u1)
    u1[1:-1] = 0.75*uu[1:-1] + 0.25*u1[1:-1] + 0.25*dt*rr[1:-1]
    	
    # third step
    rr = rhsR(nx,dx,nu1,nu2,g1,g2,nxb,u1)
    uu[1:-1] = 1.0/3.0*uu[1:-1] + 2.0/3.0*u1[1:-1] + 2.0/3.0*dt*rr[1:-1]
                 
    uTP[1:-1,jj] = np.copy(uu[1:-1])

    # check for CFL
    neu = np.max([nu2])*dt/(dx*dx)
    #check for numerical stability
    if (neu > 0.5):
        print('Error: CFL limit exceeded -- TP')
        #break

    # DP
    #RK3 scheme
    # first step    
    uu = np.copy(uDP[:,jj-1])    
    u1 = np.copy(uDP[:,jj-1])    

    rr = rhsR(nx,dx,nu1,nu2,g1,g2,nxb,uu)
    u1[1:-1] = uu[1:-1] + dt*rr[1:-1]
    
    # second step
    rr = rhsR(nx,dx,nu1,nu2,g1,g2,nxb,u1)
    u1[1:-1] = 0.75*uu[1:-1] + 0.25*u1[1:-1] + 0.25*dt*rr[1:-1]
    	
    # third step
    rr = rhsR(nx,dx,nu1,nu2,g1,g2,nxb,u1)
    uu[1:-1] = 1.0/3.0*uu[1:-1] + 2.0/3.0*u1[1:-1] + 2.0/3.0*dt*rr[1:-1]
                 
    uDP[1:-1,jj] = np.copy(uu[1:-1])

    # check for CFL
    neu = np.max([nu2])*dt/(dx*dx)
    #check for numerical stability
    if (neu > 0.5):
        print('Error: CFL limit exceeded -- GP')
        break
    
    
    # PC
    #RK3 scheme
    # first step    
    uu = np.copy(uPC[:,jj-1] )   
    u1 = np.copy(uPC[:,jj-1] )   
    
    rr = rhsR(nx,dx,nu1,nu2,g1,g2,nxb,uu)
    u1[1:-1] = uu[1:-1] + dt*rr[1:-1]
    
    # second step
    rr = rhsR(nx,dx,nu1,nu2,g1,g2,nxb,u1)
    u1[1:-1] = 0.75*uu[1:-1] + 0.25*u1[1:-1] + 0.25*dt*rr[1:-1]
    	
    # third step
    rr = rhsR(nx,dx,nu1,nu2,g1,g2,nxb,u1)
    uu[1:-1] = 1.0/3.0*uu[1:-1] + 2.0/3.0*u1[1:-1] + 2.0/3.0*dt*rr[1:-1]
                 
    uPC[1:-1,jj] = np.copy(uu[1:-1] )

    # check for CFL
    neu = np.max([nu2])*dt/(dx*dx)
    #check for numerical stability
    if (neu > 0.5):
        print('Error: CFL limit exceeded -- PC')
        break
    
    # CP
    #RK3 scheme
    # first step    
    uu = np.copy(uCP[:,jj-1]    )
    u1 = np.copy(uCP[:,jj-1]    )

    rr = rhsR(nx,dx,nu1,nu2,g1,g2,nxb,uu)
    u1[1:-1] = uu[1:-1] + dt*rr[1:-1]
    
    # second step
    rr = rhsR(nx,dx,nu1,nu2,g1,g2,nxb,u1)
    u1[1:-1] = 0.75*uu[1:-1] + 0.25*u1[1:-1] + 0.25*dt*rr[1:-1]
    	
    # third step
    rr = rhsR(nx,dx,nu1,nu2,g1,g2,nxb,u1)
    uu[1:-1] = 1.0/3.0*uu[1:-1] + 2.0/3.0*u1[1:-1] + 2.0/3.0*dt*rr[1:-1]
                 
    uCP[1:-1,jj] = np.copy(uu[1:-1])

    # check for CFL
    neu = np.max([nu2])*dt/(dx*dx)
    #check for numerical stability
    if (neu > 0.5):
        print('Error: CFL limit exceeded -- CP')
        break
    
    # UP
    #RK3 scheme
    # first step    
    uu = np.copy(uUP[:,jj-1]    )
    u1 = np.copy(uUP[:,jj-1]    )

    rr = rhsR(nx,im,dx,nu1,nu2,g1,g2,nxb,uu)
    u1[1:-1] = uu[1:-1] + dt*rr[1:-1]
    
    # second step
    rr = rhsR(nx,im,dx,nu1,nu2,g1,g2,nxb,u1)
    u1[1:-1] = 0.75*uu[1:-1] + 0.25*u1[1:-1] + 0.25*dt*rr[1:-1]
    	
    # third step
    rr = rhsR(nx,im,dx,nu1,nu2,g1,g2,nxb,u1)
    uu[1:-1] = 1.0/3.0*uu[1:-1] + 2.0/3.0*u1[1:-1] + 2.0/3.0*dt*rr[1:-1]
                
    uUP[1:-1,jj] = np.copy(uu[1:-1])

    # check for CFL
    neu = np.max([nu2])*dt/(dx*dx)
    #check for numerical stability
    if (neu > 0.5):
        print('Error: CFL limit exceeded -- UP')
        break
    
    
        
#%% Saving data

#create data folder
if os.path.isdir("./Data"):
    print('Data folder already exists')
else: 
    print('Creating data folder')
    os.makedirs("./Data")
 
print('Saving data')      
np.save('./Data/uTP_r='+str(nr)+'_.npy',uTP)
np.save('./Data/uGP_r='+str(nr)+'_.npy',uGP)
np.save('./Data/uPC_r='+str(nr)+'_.npy',uPC)
np.save('./Data/uCP_r='+str(nr)+'_.npy',uCP)
np.save('./Data/uUP_r='+str(nr)+'_.npy',uUP)


#%% surface plots of spatio-temporal evolution
import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
mpl.rcParams['text.latex.preamble'] = [r'\boldmath']
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
mpl.rc('font', **font)


from mpl_toolkits.mplot3d import Axes3D  

X, Y = np.meshgrid(t,x[nxb:])
r = 100 #change to 1 for high resolution, but it will take some time
c = 100 #change to 1 for high resolution, but it will take some time

fig = plt.figure(figsize=(10,6))

######### FOM #########
ax = fig.add_subplot(2, 3, 1, projection='3d')
surf = ax.plot_surface(X, Y, uFOM[nxb:,:], cmap='coolwarm', edgecolor='none',
                           linewidth=1, shade=False,antialiased=False,rstride=r,
                           cstride=c,rasterized=True)

surf.set_edgecolors(surf.to_rgba(surf._A))
surf.set_facecolors("white")

ax.set_title(r'\bf{FOM}', fontsize = 14)
ax.set_xticks([0,1.0,2.0])
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(11)
    
ax.set_yticks([0.7,0.85,1.0])
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(11)
    
ax.set_zlim([0,0.9])
ax.set_zticks([0,0.4,0.8])
for tick in ax.zaxis.get_major_ticks():
    tick.label.set_fontsize(11)
    
ax.set_xlabel('$t$', fontsize = 14)#, labelpad=2)
ax.set_ylabel('$x$', fontsize = 14, labelpad=6)
ax.set_zlabel('$u(x,t)$', fontsize = 14)#, labelpad=-2)


######### True POD #########
ax = fig.add_subplot(2, 3, 2, projection='3d')
surf = ax.plot_surface(X, Y, uTP, cmap='coolwarm',
                           linewidth=1, shade=False, antialiased=False,rstride=r,
                            cstride=c,rasterized=True)
surf.set_edgecolors(surf.to_rgba(surf._A))
surf.set_facecolors("white")
  
ax.set_title(r'\bf{TP}', fontsize = 14)
ax.set_xticks([0,1.0,2.0])
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(11)
    
ax.set_yticks([0.7,0.85,1.0])
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(11)
    
ax.set_zlim([0,0.9])
ax.set_zticks([0,0.4,0.8])
for tick in ax.zaxis.get_major_ticks():
    tick.label.set_fontsize(11)
    
ax.set_xlabel('$t$', fontsize = 14)#, labelpad=2)
ax.set_ylabel('$x$', fontsize = 14, labelpad=6)
ax.set_zlabel('$u(x,t)$', fontsize = 14)#, labelpad=-2)


######### DP #########
ax = fig.add_subplot(2, 3, 3, projection='3d')
surf = ax.plot_surface(X, Y, uDP, cmap='coolwarm',
                           linewidth=1, shade=False, antialiased=False,rstride=r,
                            cstride=c,rasterized=True)
surf.set_edgecolors(surf.to_rgba(surf._A))
surf.set_facecolors("white")
  
ax.set_title(r'\bf{DPI}', fontsize = 14)
ax.set_xticks([0,1.0,2.0])
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(11)
    
ax.set_yticks([0.7,0.85,1.0])
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(11)
    
ax.set_zlim([0,0.9])
ax.set_zticks([0,0.4,0.8])
for tick in ax.zaxis.get_major_ticks():
    tick.label.set_fontsize(11)
    
ax.set_xlabel('$t$', fontsize = 14)#, labelpad=2)
ax.set_ylabel('$x$', fontsize = 14, labelpad=6)
ax.set_zlabel('$u(x,t)$', fontsize = 14)#, labelpad=-2)


######### PCI #########
ax = fig.add_subplot(2, 3, 4, projection='3d')
surf = ax.plot_surface(X, Y, uPC, cmap='coolwarm',
                           linewidth=1, shade=False, antialiased=False,rstride=r,
                            cstride=c,rasterized=True)
surf.set_edgecolors(surf.to_rgba(surf._A))
surf.set_facecolors("white")
  
ax.set_title(r'\bf{PCI}', fontsize = 14)
ax.set_xticks([0,1.0,2.0])
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(11)
    
ax.set_yticks([0.7,0.85,1.0])
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(11)
    
ax.set_zlim([0,0.9])
ax.set_zticks([0,0.4,0.8])
for tick in ax.zaxis.get_major_ticks():
    tick.label.set_fontsize(11)
    
ax.set_xlabel('$t$', fontsize = 14)#, labelpad=2)
ax.set_ylabel('$x$', fontsize = 14, labelpad=6)
ax.set_zlabel('$u(x,t)$', fontsize = 14)#, labelpad=-2)


######### CPI #########
ax = fig.add_subplot(2, 3, 5, projection='3d')
surf = ax.plot_surface(X, Y, uCP, cmap='coolwarm',
                           linewidth=1, shade=False, antialiased=False,rstride=r,
                            cstride=c,rasterized=True)
surf.set_edgecolors(surf.to_rgba(surf._A))
surf.set_facecolors("white")
  
ax.set_title(r'\bf{CPI}', fontsize = 14)
ax.set_xticks([0,1.0,2.0])
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(11)
    
ax.set_yticks([0.7,0.85,1.0])
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(11)
    
ax.set_zlim([0,0.9])
ax.set_zticks([0,0.4,0.8])
for tick in ax.zaxis.get_major_ticks():
    tick.label.set_fontsize(11)
    
ax.set_xlabel('$t$', fontsize = 14)#, labelpad=2)
ax.set_ylabel('$x$', fontsize = 14, labelpad=6)
ax.set_zlabel('$u(x,t)$', fontsize = 14)#, labelpad=-2)
ax.set_zlabel('$u(x,t)$', fontsize = 14)#, labelpad=-2)


######### UPI #########
ax = fig.add_subplot(2, 3, 6, projection='3d')
surf = ax.plot_surface(X, Y, uUP, cmap='coolwarm',
                           linewidth=1, shade=False, antialiased=False,rstride=r,
                            cstride=c,rasterized=True)
surf.set_edgecolors(surf.to_rgba(surf._A))
surf.set_facecolors("white")
  
ax.set_title(r'\bf{UPI}', fontsize = 14)
ax.set_xticks([0,1.0,2.0])
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(11)
    
ax.set_yticks([0.7,0.85,1.0])
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(11)
    
ax.set_zlim([0,0.9])
ax.set_zticks([0,0.4,0.8])
for tick in ax.zaxis.get_major_ticks():
    tick.label.set_fontsize(11)
    
ax.set_xlabel('$t$', fontsize = 14)#, labelpad=2)
ax.set_ylabel('$x$', fontsize = 14, labelpad=6)
ax.set_zlabel('$u(x,t)$', fontsize = 14)#, labelpad=-2)


fig.subplots_adjust(bottom=0.15,hspace=0.25, wspace=0.2)
cbar_ax = fig.add_axes([0.33, 0.01, 0.4, 0.04])
fig.colorbar(surf,cax=cbar_ax,orientation='horizontal')
cbar_ax.tick_params(labelsize=18)

fig.add_axes([0.9,0.2,0.04,0.1]).axis("off")

#%
fig.savefig('Plots/surf_r' + str(nr) + '.png',\
            dpi = 500, bbox_inches = 'tight')
