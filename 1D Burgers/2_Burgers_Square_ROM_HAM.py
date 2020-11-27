# -*- coding: utf-8 -*-
"""
Interface learning from ROM solver for the 1D Burgers problem with zonal heterogenity
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
from numpy.random import seed
seed(0)
import os

import tensorflow as tf
tf.random.set_seed(1)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

#%% Define Functions
###############################################################################
#POD Routines
###############################################################################         
def POD(u,R): #Basis Construction
    n,ns = u.shape
    U,S,Vh = LA.svd(u, full_matrices=False)
    Phi = U[:,:R]
    L = S**2
    #compute RIC (relative inportance index)
    RIC = sum(L[:R])/sum(L)*100   
    return Phi,L,RIC

def PODproj(u,Phi): #Projection
    a = np.dot(u.T,Phi)  # u = Phi * a.T
    return a

def PODrec(a,Phi): #Reconstruction    
    u = np.dot(Phi,a.T)    
    return u

###############################################################################
# Numerical Routines
###############################################################################
# Thomas algorithm for solving tridiagonal systems:    
def tdma(a, b, c, r, up, s, e):
    for i in range(s+1,e+1):
        b[i] = b[i] - a[i]/b[i-1]*c[i-1]
        r[i] = r[i] - a[i]/b[i-1]*r[i-1]   
    up[e] = r[e]/b[e]   
    for i in range(e-1,s-1,-1):
        up[i] = (r[i]-c[i]*up[i+1])/b[i]

# Computing first derivatives using the fourth order compact scheme:  
def pade4d(u, h, n):
    a, b, c, r = [np.zeros(n+1) for _ in range(4)]
    ud = np.zeros(n+1)
    i = 0
    b[i] = 1.0
    c[i] = 2.0
    r[i] = (-5.0*u[i] + 4.0*u[i+1] + u[i+2])/(2.0*h)
    for i in range(1,n):
        a[i] = 1.0
        b[i] = 4.0
        c[i] = 1.0
        r[i] = 3.0*(u[i+1] - u[i-1])/h
    i = n
    a[i] = 2.0
    b[i] = 1.0
    r[i] = (-5.0*u[i] + 4.0*u[i-1] + u[i-2])/(-2.0*h)
    tdma(a, b, c, r, ud, 0, n)
    return ud
    
# Computing second derivatives using the foruth order compact scheme:  
def pade4dd(u, h, n):
    a, b, c, r = [np.zeros(n+1) for _ in range(4)]
    udd = np.zeros(n+1)
    i = 0
    b[i] = 1.0
    c[i] = 11.0
    r[i] = (13.0*u[i] - 27.0*u[i+1] + 15.0*u[i+2] - u[i+3])/(h*h)
    for i in range(1,n):
        a[i] = 0.1
        b[i] = 1.0
        c[i] = 0.1
        r[i] = 1.2*(u[i+1] - 2.0*u[i] + u[i-1])/(h*h)
    i = n
    a[i] = 11.0
    b[i] = 1.0
    r[i] = (13.0*u[i] - 27.0*u[i-1] + 15.0*u[i-2] - u[i-3])/(h*h)
    
    tdma(a, b, c, r, udd, 0, n)
    return udd

# Galerkin Projection
def rhs(nr, b_l, b_nl, a): # Right Handside of Galerkin Projection
    r2, r3, r = [np.zeros(nr) for _ in range(3)]
    
    a = a.ravel()
    for k in range(nr):
        r2[k] = np.sum(b_l[:,k]*a)
    
    for k in range(nr):
        for i in range(nr):
            r3[k] = r3[k] + np.sum(b_nl[i,:,k]*a)*a[i]

    r = r2 + r3
    r = r.reshape(-1,1)    
    return r

###############################################################################
# Model map
def M(nr, b_l, b_nl, a0):
    global dt
    
    a0 = a0.reshape(-1,1)
    k1 = rhs(nr, b_l, b_nl, a0)
    k2 = rhs(nr, b_l, b_nl, a0+k1*dt/2)
    k3 = rhs(nr, b_l, b_nl, a0+k2*dt/2)
    k4 = rhs(nr, b_l, b_nl, a0+k3*dt)
    a = a0 + (dt/6)*(k1+2*k2+2*k3+k4)
    return a

#-----------------------------------------------------------------------------#
# Neural network Routines
#-----------------------------------------------------------------------------#
def create_training_data_lstm(features,labels, m, n, lookback):
    # m : number of snapshots 
    # n: number of states
    ytrain = [labels[i,:] for i in range(lookback-1,m)]
    ytrain = np.array(ytrain)    
    
    xtrain = np.zeros((m-lookback+1,lookback,n))
    for i in range(m-lookback+1):
        a = features[i,:]
        for j in range(1,lookback):
            a = np.vstack((a,features[i+j,:]))
        xtrain[i,:,:] = a
    return xtrain , ytrain


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

ns = 200 #number of snapshots to load

nr = 2    #number of modes
nrs = 2*nr     #number of modes for uplift

training = 'true'
#%% FOM snapshot generation for training
print('Reading FOM snapshots...')

nxb= int(6*nx/8)
data = np.load('./Data/uFOM.npy')
uFOMr = data[:nxb+1,:4001:20] #train with data for 0<t<1
uFOM = data[:nxb+1,:]

#%% POD basis computation for training data
print('Computing POD basis...')
L = np.zeros((ns+1)) #Eigenvalues      
Phi, L, RIC  = POD(uFOMr, nr)
Phis, L, RICs  = POD(uFOMr, nrs)
        
#%% Calculating true POD modal coefficients
aTrue = np.zeros((ns+1,nr))
print('Computing true POD coefficients...')
aPOD = PODproj(uFOM,Phi)
#Unifying signs for proper training and interpolation
Phi = Phi/np.sign(aPOD[0,:])
aPOD = aPOD/np.sign(aPOD[0,:])

aPODs = PODproj(uFOM,Phis)
#Unifying signs for proper training and interpolation
Phis = Phis/np.sign(aPODs[0,:])
aPODs = aPODs/np.sign(aPODs[0,:])

#%% Model 1 - DPI: Direct Prolongation Interface 
# direct construction from ROM to FOM

#Galerkin Projection 
b_l = np.zeros((nr,nr))
b_nl = np.zeros((nr,nr,nr))
Phid = np.zeros((nxb+1,nr))
Phidd = np.zeros((nxb+1,nr))

for i in range(nr):
    Phid[:,i] = pade4d(Phi[:nxb+1,i],dx,nxb)
    Phidd[:,i] = pade4dd(Phi[:nxb+1,i],dx,nxb)
    Phidd[:nxb+1,i] = Phidd[:nxb+1,i] * nu1  - Phi[:nxb+1,i] * g1

# linear term   
for k in range(nr):
    for i in range(nr):
        b_l[i,k] = np.dot(Phidd[:,i].T , Phi[:,k]) 
                   
# nonlinear term
for k in range(nr):
    for j in range(nr):
        for i in range(nr):
            temp = Phi[:,i]*Phid[:,j]
            b_nl[i,j,k] = - np.dot( temp.T, Phi[:,k] ) 

# solving ROM for tfreq timesteps            
# time integration using fourth-order Runge Kutta method
aDPI = np.zeros((nt+1,nr))
aDPI[0,:] = aPOD[0,:]
for j in range(nt):
    aDPI[j+1,:] = (M(nr, b_l, b_nl, aDPI[j,:])).ravel()

uDPI = PODrec(aDPI,Phi) #Reconstruction   

#%% TPI: True projection
uPOD = PODrec(aPOD,Phi) #Reconstruction    

#%% Model 2 - PCI: Prolongation followed by Correction Interface 
# here we learn BC on FOM space

# Create input/output
xi = np.zeros((nt+1,3))
xi[:,0] = x[nxb]
xi[:,1] = t
xi[:,2] = np.copy(uDPI[nxb,:])

yi = np.zeros((nt+1,1))
yi[:,0] = np.copy(uFOM[nxb,:] - uDPI[nxb,:])

lookback = 1
features = xi[:int(nt/2)+1,:] #train with data for 0<t<1
labels = yi[:int(nt/2)+1,:]   #train with data for 0<t<1
xtrain, ytrain = create_training_data_lstm(features, labels,  features.shape[0], \
                                      features.shape[1], lookback)
                
# Scaling data
m,n = ytrain.shape # m is number of training samples, n is number of output features
scalerOut = MinMaxScaler(feature_range=(-1,1))
scalerOut = scalerOut.fit(ytrain)
ytrain = scalerOut.transform(ytrain)

for k in range(lookback):
    if k == 0:
        tmp = xtrain[:,k,:]
    else:
        tmp = np.vstack([tmp,xtrain[:,k,:]])
        
scalerIn = MinMaxScaler(feature_range=(-1,1))
scalerIn = scalerIn.fit(tmp)
for i in range(m):
    xtrain[i,:,:] = scalerIn.transform(xtrain[i,:,:])

# Shuffling data
perm = np.random.permutation(m)
xtrain = xtrain[perm,:,:]
ytrain = ytrain[perm,:]

if training == 'true': 
    #create folder
    if os.path.isdir("./LSTM Model"):
        print('LSTM models folder already exists')
    else: 
        print('Creating LSTM models folder')
        os.makedirs("./LSTM Model")
        
        
    #Removing old models
    model_name = 'LSTM Model/LSTM_PCI_' + str(nr) + '.h5'
    if os.path.isfile(model_name):
       os.remove(model_name)
    
    #create the LSTM architecture
    model = Sequential()
    model.add(LSTM(20, input_shape=(lookback, features.shape[1]), return_sequences=True, activation='tanh'))
    model.add(LSTM(20, input_shape=(lookback, features.shape[1]), activation='tanh'))
    model.add(Dense(labels.shape[1]))
    
    #compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    
    #run the model
    history = model.fit(xtrain, ytrain, epochs=500, batch_size=64, validation_split=0.25)
    
    #evaluate the model
    scores = model.evaluate(xtrain, ytrain, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure()
    epochs = range(1, len(loss) + 1)
    plt.semilogy(epochs, loss, 'b', label='Training loss')
    plt.semilogy(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    filename = 'LSTM Model/loss_PCI_' + str(nr) + '.png'
    plt.savefig(filename, dpi = 200)
    plt.show()
    
    #Save the model
    model.save(model_name)
    #Save the scales
    filename = 'LSTM Model/input_scaler_PCI_' + str(nr) + '.save'
    joblib.dump(scalerIn,filename) 
    filename = 'LSTM Model/output_scaler_PCI_' + str(nr) + '.save'
    joblib.dump(scalerOut,filename) 

        
#%% Testing
model_name = 'LSTM Model/LSTM_PCI_' + str(nr) + '.h5'
model = load_model(model_name)  

# load scales
filename = 'LSTM Model/input_scaler_PCI_' + str(nr) + '.save'
scalerIn = joblib.load(filename)  
filename = 'LSTM Model/output_scaler_PCI_' + str(nr) + '.save'
scalerOut = joblib.load(filename) 
        
uPCI = np.zeros((nt+1,1))

xtest = np.zeros((1,lookback,features.shape[1]))
for i in range(lookback):
    uPCI[i,:] = uFOM[nxb,i]
    tmp = np.hstack([x[nxb] , t[i], uDPI[nxb,i]])

    xtest[0,i,:] = scalerIn.transform(tmp.reshape((1,-1))) 

for i in range(lookback-1,nt):
    # update xtest
    for ii in range(lookback-1):
        xtest[0,ii,:] = xtest[0,ii+1,:]
        
    tmp = np.hstack([x[nxb] , t[i+1], uDPI[nxb,i+1]])
    xtest[0,lookback-1,:] = scalerIn.transform(tmp.reshape((1,-1))) 
    ytest = model.predict(xtest)
    ytest = scalerOut.inverse_transform(ytest) # rescale  
    uPCI[i+1,:] = uDPI[nxb,i+1] + ytest
    
       
#%% Model 3 - CPI: Correction followed by Prolongation Interface 
#here we learn correction on ROM space in such a way that once prolongated it match the BC
       
#Create input/output
xi = np.zeros((nt+1,nr+2))
xi[:,0] = x[nxb]
xi[:,1] = t
xi[:,2:] = aDPI

yi = np.zeros((nt+1,nr))
yi = aPOD-aDPI

lookback = 1
features = xi[:int(nt/2)+1,:] #train with data for 0<t<1
labels = yi[:int(nt/2)+1,:]   #train with data for 0<t<1
xtrain, ytrain = create_training_data_lstm(features, labels,  features.shape[0], \
                                      features.shape[1], lookback)
    
#Scaling data
m,n = ytrain.shape # m is number of training samples, n is number of output features
scalerOut = MinMaxScaler(feature_range=(-1,1))
scalerOut = scalerOut.fit(ytrain)
ytrain = scalerOut.transform(ytrain)

for k in range(lookback):
    if k == 0:
        tmp = xtrain[:,k,:]
    else:
        tmp = np.vstack([tmp,xtrain[:,k,:]])
        
scalerIn = MinMaxScaler(feature_range=(-1,1))
scalerIn = scalerIn.fit(tmp)
for i in range(m):
    xtrain[i,:,:] = scalerIn.transform(xtrain[i,:,:])

#Shuffling data
perm = np.random.permutation(m)
xtrain = xtrain[perm,:,:]
ytrain = ytrain[perm,:]


if training == 'true': 
    #create folder
    if os.path.isdir("./LSTM Model"):
        print('LSTM models folder already exists')
    else: 
        print('Creating LSTM models folder')
        os.makedirs("./LSTM Model")
        
    #Removing old models
    model_name = 'LSTM Model/LSTM_CPI_' + str(nr) + '.h5'
    if os.path.isfile(model_name):
       os.remove(model_name)
    
    
    #create the LSTM architecture
    model = Sequential()
    model.add(LSTM(20, input_shape=(lookback, features.shape[1]), return_sequences=True, activation='tanh'))
    model.add(LSTM(20, input_shape=(lookback, features.shape[1]), activation='tanh'))
    model.add(Dense(labels.shape[1]))
    #compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    
    #run the model
    history = model.fit(xtrain, ytrain, epochs=500, batch_size=64, validation_split=0.20)
    
    #evaluate the model
    scores = model.evaluate(xtrain, ytrain, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure()
    epochs = range(1, len(loss) + 1)
    plt.semilogy(epochs, loss, 'b', label='Training loss')
    plt.semilogy(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    filename = 'LSTM Model/loss_CPI_' + str(nr) + '.png'
    plt.savefig(filename, dpi = 200)
    plt.show()
    
    #Save the model
    model.save(model_name)
    #Save the scales
    filename = 'LSTM Model/input_scaler_CPI_' + str(nr) + '.save'
    joblib.dump(scalerIn,filename) 
    filename = 'LSTM Model/output_scaler_CPI_' + str(nr) + '.save'
    joblib.dump(scalerOut,filename) 

       
#%% Testing

model_name = 'LSTM Model/LSTM_CPI_' + str(nr) + '.h5'
model = load_model(model_name)  

# load scales
filename = 'LSTM Model/input_scaler_CPI_' + str(nr) + '.save'
scalerIn = joblib.load(filename)  
filename = 'LSTM Model/output_scaler_CPI_' + str(nr) + '.save'
scalerOut = joblib.load(filename) 
        
aCPI = np.zeros((nt+1,nr))

xtest = np.zeros((1,lookback,features.shape[1]))
for i in range(lookback):
    aCPI[i,:] = aPOD[i,:]

    tmp = np.hstack([x[nxb] , t[i], aPOD[i,:]])
    xtest[0,i,:] = scalerIn.transform(tmp.reshape((1,-1))) 

for i in range(lookback-1,nt):
    # update xtest
    for ii in range(lookback-1):
        xtest[0,ii,:] = xtest[0,ii+1,:]
        
    tmp = np.hstack([x[nxb] , t[i+1], aDPI[i+1,:]])
    xtest[0,lookback-1,:] = scalerIn.transform(tmp.reshape((1,-1))) 

    ytest = model.predict(xtest)
    ytest = scalerOut.inverse_transform(ytest) # rescale  
    aCPI[i+1,:] = aDPI[i+1,:] + ytest
    
uCPI = PODrec(aCPI,Phi) #Reconstruction    
    

#%% Model 4 - UPI: Uplifted Prolongation Interface 
# here we apply uplifting to enhance number of ROM variables and used prolongation to meet FOM bc
    
#Create input/output
xi = np.zeros((nt+1,nr+2))
xi[:,0] = x[nxb]
xi[:,1] = t
xi[:,2:] = aDPI

yi = np.zeros((nt+1,nrs))
yi = aPODs

lookback = 1
features = xi[:int(nt/2)+1,:] #train with data for 0<t<1
labels = yi[:int(nt/2)+1,:]  #train with data for 0<t<1
xtrain, ytrain = create_training_data_lstm(features, labels, features.shape[0], \
                                      features.shape[1], lookback)
#Scaling data
m,n = ytrain.shape # m is number of training samples, n is number of output features
scalerOut = MinMaxScaler(feature_range=(-1,1))
scalerOut = scalerOut.fit(ytrain)
ytrain = scalerOut.transform(ytrain)

for k in range(lookback):
    if k == 0:
        tmp = xtrain[:,k,:]
    else:
        tmp = np.vstack([tmp,xtrain[:,k,:]])
        
scalerIn = MinMaxScaler(feature_range=(-1,1))
scalerIn = scalerIn.fit(tmp)
for i in range(m):
    xtrain[i,:,:] = scalerIn.transform(xtrain[i,:,:])

#Shuffling data
perm = np.random.permutation(m)
xtrain = xtrain[perm,:,:]
ytrain = ytrain[perm,:]

if training == 'true': 
    #create folder
    if os.path.isdir("./LSTM Model"):
        print('LSTM models folder already exists')
    else: 
        print('Creating LSTM models folder')
        os.makedirs("./LSTM Model")
        
    #Removing old models
    model_name = 'LSTM Model/LSTM_UPI_' + str(nr) + '.h5'
    if os.path.isfile(model_name):
       os.remove(model_name)
    
    #create the LSTM architecture
    model = Sequential()
    model.add(LSTM(80, input_shape=(lookback, features.shape[1]), return_sequences=True, activation='tanh'))
    model.add(LSTM(80, input_shape=(lookback, features.shape[1]), activation='tanh'))
    model.add(Dense(labels.shape[1]))
    
    #compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    
    #run the model
    history = model.fit(xtrain, ytrain, epochs=500, batch_size=64, validation_split=0.20)
    
    #evaluate the model
    scores = model.evaluate(xtrain, ytrain, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure()
    epochs = range(1, len(loss) + 1)
    plt.semilogy(epochs, loss, 'b', label='Training loss')
    plt.semilogy(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    filename = 'LSTM Model/loss_UPI_' + str(nr) + '.png'
    plt.savefig(filename, dpi = 200)
    plt.show()
    
    #Save the model
    model.save(model_name)
    #Save the scales
    filename = 'LSTM Model/input_scaler_UPI_' + str(nr) + '.save'
    joblib.dump(scalerIn,filename) 
    filename = 'LSTM Model/output_scaler_UPI_' + str(nr) + '.save'
    joblib.dump(scalerOut,filename) 

        
#%% Testing

model_name = 'LSTM Model/LSTM_UPI_' + str(nr) + '.h5'
model = load_model(model_name)  
# load scales
filename = 'LSTM Model/input_scaler_UPI_' + str(nr) + '.save'
scalerIn = joblib.load(filename)  
filename = 'LSTM Model/output_scaler_UPI_' + str(nr) + '.save'
scalerOut = joblib.load(filename) 
        
aUPI = np.zeros((nt+1,nrs))

xtest = np.zeros((1,lookback,features.shape[1]))
for i in range(lookback):
    aUPI[i,:] = aPODs[i,:]

    tmp = np.hstack([x[nxb] , t[i], aPOD[i,:]])
    xtest[0,i,:] = scalerIn.transform(tmp.reshape((1,-1))) 

for i in range(lookback-1,nt):
    # update xtest
    for ii in range(lookback-1):
        xtest[0,ii,:] = xtest[0,ii+1,:]
        
    tmp = np.hstack([x[nxb] , t[i+1], aDPI[i+1,:]])
    xtest[0,lookback-1,:] = scalerIn.transform(tmp.reshape((1,-1))) 

    ytest = model.predict(xtest)
    ytest = scalerOut.inverse_transform(ytest) # rescale  
    aUPI[i+1,:] = ytest

uUPI = PODrec(aUPI,Phis) #Reconstruction    


#%%
print('Saving')

np.save('./ROM_Truncated/Phi_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy',Phi)
np.save('./ROM_Truncated/Phis_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy',Phis)

np.save('./ROM_Truncated/aPOD_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy',aPOD)
np.save('./ROM_Truncated/aPODs_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy',aPODs)

np.save('./ROM_Truncated/aDPI_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy',aDPI)
np.save('./ROM_Truncated/aCPI_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy',aCPI)
np.save('./ROM_Truncated/aUPI_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy',aUPI)

np.save('./ROM_Truncated/uPOD_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy',uPOD)
np.save('./ROM_Truncated/uDPI_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy',uDPI)
np.save('./ROM_Truncated/uPCI_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy',uPCI)
np.save('./ROM_Truncated/uCPI_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy',uCPI)
np.save('./ROM_Truncated/uUPI_xb='+str(nxb/nx)+'_r='+str(nr)+'.npy',uUPI)

