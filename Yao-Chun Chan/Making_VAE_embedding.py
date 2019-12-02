#This script skip the visualize part


import pandas as pd
import numpy as np
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential
from keras import optimizers

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

import argparse
import os

from VAE import*
pre_embedding_train=pd.read_csv('member_temp_ebedding_pre.csv')
original_dim= pre_embedding_train.shape[1]
input_shape = (original_dim, )
intermediate_dim = int(original_dim/2)
batch_size = 128
latent_dim = 50
epochs     = 100
epsilon_std = 1.0

x, eps, z_mu, x_pred = vae_arc(original_dim, intermediate_dim, latent_dim)
vae            = Model(inputs=[x, eps], outputs=x_pred)
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
vae.compile(optimizer='adam', loss=nll)



scaler    = MinMaxScaler()

pre_embedding_train_norm   = scaler.fit_transform(pre_embedding_train)
# 
X_train, X_test, y_train, y_test = train_test_split(pre_embedding_train_norm, pre_embedding_train_norm, 
                                                    test_size=0.2, random_state=42)
                                                    
filepath   ="vae_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
vae.fit(X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        validation_data=(X_test, X_test))

encoder = Model(x, z_mu)
z_df    = encoder.predict(pre_embedding_train_norm, batch_size=batch_size)


msno=pd.read_csv('members.csv')['msno']
res_dict={}
for ctr,item in enumerate(z_df):
    res_dict[msno.iloc[ctr]]=item
    
    
import pickle 

with open('user_embedding_dict.pickle', 'wb') as f:
     pickle.dump(res_dict, f, protocol=pickle.HIGHEST_PROTOCOL)



