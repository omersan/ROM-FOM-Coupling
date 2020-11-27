# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:50:00 2020

@author: Shady
"""



import numpy as np
from numpy.random import seed
seed(1)
import matplotlib.pyplot as plt 
#import time as clck
import os
from scipy.fftpack import dst, idst
from numpy import linalg as LA

#%% Define functions


def PODproj_svd(u,Phi): #Projection
    a = np.dot(u.T,Phi)  # u = Phi * a.T
    return a

def PODrec_svd(a,Phi): #Reconstruction    
    u = np.dot(Phi,a.T)    
    return u

def import_data(nx,ny,n):
    folder = 'data_'+ str(nx) + '_' + str(ny)              
    filename = './Results/'+folder+'/data_' + str(int(n))+'.npz'
    data = np.load(filename)
    w = data['w']
    s = data['s']
    t = data['t']
    return w,s,t


def import_data_DPI(nx,ny,n):
    folder = 'data_'+ str(nx) + '_' + str(ny)               
    filename = './DPI/'+folder+'/data_' + str(int(n))+'.npz'
    data = np.load(filename)
    w = data['w']
    s = data['s']
    t = data['t']
    return w,s,t


def import_data_CPI(nx,ny,n):
    folder = 'data_'+ str(nx) + '_' + str(ny)               
    filename = './CPI/'+folder+'/data_' + str(int(n))+'.npz'
    data = np.load(filename)
    w = data['w']
    s = data['s']
    t = data['t']
    return w,s,t
    

def import_data_UPI(nx,ny,n):
    folder = 'data_'+ str(nx) + '_' + str(ny)               
    filename = './UPI/'+folder+'/data_' + str(int(n))+'.npz'
    data = np.load(filename)
    w = data['w']
    s = data['s']
    t = data['t']
    return w,s,t
    

    
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
nq = 16
Phiw = Phiw[:,:nq]
Phis = Phis[:,:nq]
Phit = Phit[:,:nq]
aTP = aTrue[:,:nq]
bTP = bTrue[:,:nq]

#%%
if not os.path.exists('./Plots/'):
    os.makedirs('./Plots/')


#%% Load data 
# Load DPI data
folder = 'data_'+ str(nx) + '_' + str(ny)       
filename = './DPI/'+folder+'/DPI_data_nr='+str(nr)+'.npz'
data = np.load(filename) 
aDPI = data['aDPI']
bDPI = data['bDPI']

filename = './CPI/'+folder+'/CPI_data_nr='+str(nr)+'.npz'
data = np.load(filename) 
aCPI = data['aCPI']
bCPI = data['bCPI']

filename = './UPI/'+folder+'/UPI_data_nr='+str(nr)+'.npz'
data = np.load(filename) 
aUPI = data['aUPI']
bUPI = data['bUPI']

#%%
import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
mpl.rcParams['text.latex.preamble'] = [r'\boldmath']
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}
mpl.rc('font', **font)

time = np.linspace(0,Tm,nt+1)
freq = 20
x_ticks = [0,2,4,6,8]

fig, axs = plt.subplots(4,2,figsize=(12,10))
axs= axs.flat
for k in range(8):
    axs[k].plot(time[::freq],bTP[:,k],label=r'\bf{FOM}', color = 'k', linewidth=3)
    axs[k].plot(time,bDPI[:,k],'-.',label=r'\bf{DPI}', color = 'r', linewidth=3)
    axs[k].plot(time,bCPI[:,k],'--',label=r'\bf{CPI}', color = 'b', linewidth=3)
    axs[k].plot(time,bUPI[:,k],'--',label=r'\bf{UPI}', color = 'g', linewidth=3)

    axs[k].set_xticks(x_ticks)
    axs[k].set_xlabel(r'$t$')
    axs[k].set_ylabel(r'$\beta_{'+str(k+1)+'}(t)$')

axs[2].set_yticks([-100,0,100])
axs[7].set_yticks([-50,0,50])

axs[4].yaxis.labelpad = 12
axs[6].yaxis.labelpad = 12

axs[0].legend(loc="center", bbox_to_anchor=(1.18,1.3),ncol =4)#,fontsize=15)

fig.subplots_adjust(hspace=0.7, wspace=0.35)

plt.savefig('./Plots/BSbeta.png', dpi = 500, bbox_inches = 'tight')


#%% Load FOM results for t=0,2,4,8
n=0 #t=0
w0,s0,t0 = import_data(nx,ny,n)

n=int(2*nt/8) #t=2
w2,s2,t2 = import_data(nx,ny,n)

n=int(4*nt/8) #t=4
w4,s4,t4 = import_data(nx,ny,n)

n=int(8*nt/8) #t=8
w8,s8,t8 = import_data(nx,ny,n)



#%%

import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
mpl.rcParams['text.latex.preamble'] = [r'\boldmath']
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}
mpl.rc('font', **font)


nlvls = 31
x_ticks = [0,1,2,3,4,5,6,7,8]
y_ticks = [0,1]

colormap = 'viridis'

#colormap = 'gnuplot'
#colormap = 'inferno'
colormap = 'seismic'

v = np.linspace(1.05, 1.45, nlvls, endpoint=True)
ctick = np.linspace(1.05, 1.45, 5, endpoint=True)

fig, axs = plt.subplots(4,1,figsize=(10,8.5))
axs= axs.flat

cs = axs[0].contour(X,Y,t0,v,cmap=colormap,linewidths=3)
cs.set_clim([1.05, 1.45])

cs = axs[1].contour(X,Y,t2,v,cmap=colormap,linewidths=0.5)
cs.set_clim([1.05, 1.45])

cs = axs[2].contour(X,Y,t4,v,cmap=colormap,linewidths=0.5)
cs.set_clim([1.05, 1.45])

cs = axs[3].contour(X,Y,t8,v,cmap=colormap,linewidths=0.5)#, rasterized=True)
cs.set_clim([1.05, 1.45])


for i in range(4):
    axs[i].set_xticks(x_ticks)
    axs[i].set_xlabel('$x$')
    axs[i].set_yticks(y_ticks)
    axs[i].set_ylabel('$y$')

# Add titles
fig.text(0.92, 0.83, '$t=0$', va='center')
fig.text(0.92, 0.63, '$t=2$', va='center')
fig.text(0.92, 0.43, '$t=4$', va='center')
fig.text(0.92, 0.23, '$t=8$', va='center')

    
fig.subplots_adjust(bottom=0.18, hspace=1)
cbar_ax = fig.add_axes([0.125, 0.03, 0.775, 0.045])
CB = fig.colorbar(cs, cax = cbar_ax, ticks=ctick, orientation='horizontal')
CB.ax.get_children()[0].set_linewidths(3.0)

plt.savefig('./Plots/BSFOM.png', dpi = 500, bbox_inches = 'tight')


#%% Load coupling results at final time

n=int(8*nt/8) #t=8
wFOM,sFOM,tFOM = import_data(nx,ny,n)
wDPI,sDPI,tDPI = import_data_DPI(nx,ny,n)
wCPI,sCPI,tCPI = import_data_CPI(nx,ny,n)
wUPI,sUPI,tUPI = import_data_UPI(nx,ny,n)

#%%
# tUPI = tm + PODrec_svd(bTP[-1,:],Phit)
# tUPI = tUPI.reshape([nx+1,ny+1])
#%%

import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
mpl.rcParams['text.latex.preamble'] = [r'\boldmath']
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}
mpl.rc('font', **font)


nlvls = 30
x_ticks = [0,1,2,3,4,5,6,7,8]
y_ticks = [0,1]

colormap = 'viridis'

colormap = 'inferno'

colormap = 'seismic'

v = np.linspace(1.05, 1.45, nlvls, endpoint=True)
ctick = np.linspace(1.05, 1.45, 5, endpoint=True)

fig, axs = plt.subplots(4,1,figsize=(10,8.5))
axs= axs.flat

cs = axs[0].contour(X,Y,tFOM,v,cmap=colormap,linewidths=0.5)
cs.set_clim([1.05, 1.45])

cs = axs[1].contour(X,Y,tDPI,v,cmap=colormap,linewidths=0.5)
cs.set_clim([1.05, 1.45])

cs = axs[2].contour(X,Y,tCPI,v,cmap=colormap,linewidths=0.5)
cs.set_clim([1.05, 1.45])

cs = axs[3].contour(X,Y,tUPI,v,cmap=colormap,linewidths=0.5)#, rasterized=True)
cs.set_clim([1.05, 1.45])


for i in range(4):
    axs[i].set_xticks(x_ticks)
    axs[i].set_xlabel('$x$')
    axs[i].set_yticks(y_ticks)
    axs[i].set_ylabel('$y$')
    
# Add titles
fig.text(0.92, 0.83, r'\bf{FOM}', va='center')
fig.text(0.92, 0.63, r'\bf{DPI}', va='center')
fig.text(0.92, 0.43, r'\bf{CPI}', va='center')
fig.text(0.92, 0.23, r'\bf{UPI}', va='center')


fig.subplots_adjust(bottom=0.18,hspace=1)
cbar_ax = fig.add_axes([0.125, 0.03, 0.775, 0.045])
CB = fig.colorbar(cs, cax = cbar_ax, ticks=ctick, orientation='horizontal')
CB.ax.get_children()[0].set_linewidths(3.0)


plt.savefig('./Plots/BScoupling.png', dpi = 500, bbox_inches = 'tight')

