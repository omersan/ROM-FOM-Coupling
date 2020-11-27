# -*- coding: utf-8 -*-
"""
Plotting results for the 1D Burgers problem with zonal heterogenity
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
import os

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

tm = 2        #maximum time
dt = 2.5e-4   #solver timestep
nt = round(tm/dt)
t = np.linspace(0, tm, nt+1)

ns = 8000 #number of snapshots to load
    
#%% Reading Data
print('Reading FOM snapshots...')

nxb= int(6*nx/8)
data = np.load('./Data/uFOM_xb='+str(nxb/nx)+'_.npy')
uFOM = data[:nxb+1,:]


print('Reading ROM-FOM data')

nr = 2
Phi1 = np.load('./ROM_Truncated/Phi_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')
Phis1 = np.load('./ROM_Truncated/Phis_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')

aPOD2 = np.load('./ROM_Truncated/aPOD_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')
aPODs2 = np.load('./ROM_Truncated/aPODs_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')
aGPI2 = np.load('./ROM_Truncated/aGP_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')
aCPI2 = np.load('./ROM_Truncated/aCPI_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')
aUPI2 = np.load('./ROM_Truncated/aUPI_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')

uPOD2 = np.load('./ROM_Truncated/uPOD_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')
uGPI2 = np.load('./ROM_Truncated/uGPI_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')
uPCI2 = np.load('./ROM_Truncated/uPCI_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')
uCPI2 = np.load('./ROM_Truncated/uCPI_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')
uUPI2 = np.load('./ROM_Truncated/uUPI_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')


nr = 4
Phi4 = np.load('./ROM_Truncated/Phi_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')
Phis4 = np.load('./ROM_Truncated/Phis_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')

aPOD4 = np.load('./ROM_Truncated/aPOD_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')
aPODs4 = np.load('./ROM_Truncated/aPODs_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')
aGPI4 = np.load('./ROM_Truncated/aGP_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')
aCPI4 = np.load('./ROM_Truncated/aCPI_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')
aUPI4 = np.load('./ROM_Truncated/aUPI_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')

uPOD4 = np.load('./ROM_Truncated/uPOD_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')
uGPI4 = np.load('./ROM_Truncated/uGPI_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')
uPCI4 = np.load('./ROM_Truncated/uPCI_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')
uCPI4 = np.load('./ROM_Truncated/uCPI_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')
uUPI4 = np.load('./ROM_Truncated/uUPI_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy')

#%%
import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
mpl.rcParams['text.latex.preamble'] = [r'\boldmath']
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
mpl.rc('font', **font)

fig, ax = plt.subplots(nrows=2,ncols=4, figsize=(18,8))
ax = ax.flat

nr = 2
ax[0].plot(t,uFOM[nxb,:],label=r'\bf{FOM}', color = 'k', linewidth=3)
ax[0].plot(t,uPOD2[nxb,:],'--', label=r'\bf{TP}', color = 'b', linewidth=3)
ax[0].plot(t,uGPI2[nxb,:],':', label=r'\bf{DPI}', color = 'r', linewidth=3)
ax[0].set_title(r'\bf{DPI} $(r=2)$',fontsize=18)


ax[1].plot(t,uFOM[nxb,:],label=r'\bf{FOM}', color = 'k', linewidth=3)
ax[1].plot(t,uPOD2[nxb,:],'--', label=r'\bf{TP}', color = 'b', linewidth=3)
ax[1].plot(t,uPCI2[:],':', label=r'\bf{PCI}', color = 'r', linewidth=3)
ax[1].set_title(r'\bf{PCI} $(r=2)$',fontsize=18)


ax[2].plot(t,uFOM[nxb,:],label=r'\bf{FOM}', color = 'k', linewidth=3)
ax[2].plot(t,uPOD2[nxb,:],'--', label=r'\bf{TP}', color = 'b', linewidth=3)
ax[2].plot(t,uCPI2[nxb,:],':', label=r'\bf{CPI}', color = 'r', linewidth=3)
ax[2].set_title(r'\bf{CPI} $(r=2)$',fontsize=18)


ax[3].plot(t,uFOM[nxb,:],label=r'\bf{FOM}', color = 'k', linewidth=3)
ax[3].plot(t,uPOD2[nxb,:],'--', label=r'\bf{TP}', color = 'b', linewidth=3)
ax[3].plot(t,uUPI2[nxb,:],':', label=r'\bf{UPI}', color = 'r', linewidth=3)
ax[3].set_title(r'\bf{UPI} $(r=2)$',fontsize=18)


nr = 4
ax[4].plot(t,uFOM[nxb,:],label=r'\bf{FOM}', color = 'k', linewidth=3)
ax[4].plot(t,uPOD4[nxb,:],'--', label=r'\bf{TP}', color = 'b', linewidth=3)
ax[4].plot(t,uGPI4[nxb,:],':', label=r'\bf{DPI}', color = 'r', linewidth=3)
ax[4].set_title(r'\bf{DPI} $(r=4)$',fontsize=18)


ax[5].plot(t,uFOM[nxb,:],label=r'\bf{FOM}', color = 'k', linewidth=3)
ax[5].plot(t,uPOD4[nxb,:],'--', label=r'\bf{TP}', color = 'b', linewidth=3)
ax[5].plot(t,uPCI4[:],':', label=r'\bf{PCI}', color = 'r', linewidth=3)
ax[5].set_title(r'\bf{PCI} $(r=4)$',fontsize=18)


ax[6].plot(t,uFOM[nxb,:],label=r'\bf{FOM}', color = 'k', linewidth=3)
ax[6].plot(t,uPOD4[nxb,:],'--', label=r'\bf{TP}', color = 'b', linewidth=3)
ax[6].plot(t,uCPI4[nxb,:],':', label=r'\bf{CPI}', color = 'r', linewidth=3)
ax[6].set_title(r'\bf{CPI} $(r=4)$',fontsize=18)


ax[7].plot(t,uFOM[nxb,:],label=r'\bf{FOM}', color = 'k', linewidth=3)
ax[7].plot(t,uPOD4[nxb,:],'--', label=r'\bf{TP}', color = 'b', linewidth=3)
ax[7].plot(t,uUPI4[nxb,:],':', label=r'\bf{UPI}', color = 'r', linewidth=3)
ax[7].set_title(r'\bf{UPI} $(r=4)$',fontsize=18)


for k in range(8):
    ax[k].set_xlabel(r'$t$')
    ax[k].set_ylabel(r'$u(x_b,t)$')#, labelpad=0)  
    ax[k].set_xlim([0.0,2.0])
    ax[k].set_ylim([0.25,1.15])
    ax[k].axvspan(0, 1.0, color='y', alpha=0.2, lw=0)
    ax[k].legend(fontsize=14.5,handletextpad=0.4)



fig.subplots_adjust(hspace=0.45,wspace=0.5)     
plt.savefig('./Plots/u_t_r24.png', dpi = 500, bbox_inches = 'tight')

fig.show()
