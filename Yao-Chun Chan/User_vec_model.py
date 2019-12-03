import pandas as pd
import numpy as np
import pickle
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, GRU,GlobalAveragePooling1D,GlobalMaxPooling1D,Add
from keras.layers import TimeDistributed, RepeatVector, Input, subtract, Lambda, Conv1D,concatenate,multiply,CuDNNGRU
from keras import backend as K
from keras.engine import Layer
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from keras import initializers, regularizers, constraints, activations
from keras.layers import InputSpec,RepeatVector,Masking
from keras.layers import Bidirectional
from keras import optimizers
import math
from keras.regularizers import l1, l2, l1_l2


os.environ['KMP_DUPLICATE_LIB_OK']='True'
#import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



        
class Attention(Layer):
    """
    #Computes a weighted average of the different channels across timesteps.
    #Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(Attention, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        self.bias = self.add_weight(shape=(input_shape[2], 1),
                                      initializer='zeros',
                                      name='bias' )
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)+self.bias
        #logits= K.tanh(logits)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        #att_weights=K.softmax(logits)
        weighted_input=x* K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None        
        

class User_score_model():
      def __init__(self):
          pass  
          
          
      def deep_learning_component(self,input_channel,data_shape):
          #input_channel=Masking()(input_channel)
          gru = CuDNNGRU(data_shape[2],kernel_regularizer=l2(1e-6),return_sequences=True)(input_channel)
          fcn_output=Attention(name="user_vec_2d")(gru) 
          return fcn_output
                                             
     
      def cos_sim(self,a, b):
          """Takes 2 vectors a, b and returns the cosine similarity according 
          to the definition of the dot product
          """
          dot_product = np.dot(a, b)
          norm_a = np.linalg.norm(a)
          norm_b = np.linalg.norm(b)
          return dot_product / (norm_a * norm_b)


      def l2_norm(self,x, axis=None):
          square_sum = K.sum(K.square(x), axis=axis, keepdims=True)
          norm = K.sqrt(K.maximum(square_sum, K.epsilon()))
          return norm

      def pairwise_cos_sim(self,tensor):
          """
          t1 [batch x n x d] tensor of n rows with d dimensions
          t2 [batch x m x d] tensor of n rows with d dimensions
          returns:
          t12 [batch x n x m] tensor of cosine similarity scores between each point i<n, j<m
          """
          t1, t2 = tensor
          t1_mag = self.l2_norm(t1, axis=-1)
          t2_mag = self.l2_norm(t2, axis=-1)
          num = K.batch_dot(t1, K.permute_dimensions(t2, (0,2,1)))
          den = (t1_mag * K.permute_dimensions(t2_mag, (0,2,1)))
          t12 =  num / den
          return t12                                       
                                             
                                             
      def build_model(self,data_shape,print_model=True):
          input_pos = Input(shape=(data_shape[1],data_shape[2],), name="input_pos")
          input_neg = Input(shape=(data_shape[1],data_shape[2],), name="input_neg")
          #comb=subtract([input_pos,input_neg])
             
              
          user_vec=self.deep_learning_component(input_pos,data_shape,model=model_mode)
          #user_vec=Dropout(0.25)(user_vec)
          
          user_vec_d3    = Lambda(lambda x: K.reshape(x, (-1, 1,data_shape[2])), name = "user_vec_3d")(user_vec)
          batch_cos_pos_3d  = Lambda(self.pairwise_cos_sim, name="batch_cos_pos_3d")([user_vec_d3,input_pos])
          batch_cos_neg_3d  = Lambda(self.pairwise_cos_sim, name="batch_cos_neg_3d")([user_vec_d3,input_neg])
    
          batch_cos_pos_2d  = Lambda(lambda x: K.reshape(x, (-1, K.shape(x)[1])), name="batch_cos_pos_2d")(batch_cos_pos_3d)
          batch_cos_neg_2d  = Lambda(lambda x: K.reshape(x, (-1, K.shape(x)[1])), name="batch_cos_neg_2d")(batch_cos_neg_3d)
          batch_cos_diff_2d = subtract([batch_cos_pos_2d, batch_cos_neg_2d], name="batch_cos_diff_2d")

          output = Lambda(lambda x:K.reshape(x,(-1,data_shape[1])), name="output")(batch_cos_diff_2d)
          #output = Lambda( lambda x: K.sum(x, axis=1), name="output_sum")(output)
          #output=Dense(data_shape[1],activation='relu',use_bias=False,name="leverage")(output)
                                             
          model  = Model(inputs=[input_pos,input_neg], outputs=output)      
          if print_model:
             print(model.summary())
          sgd = optimizers.SGD(lr=0.3, decay=1e-6, momentum=0.9, nesterov=True)
          model.compile(loss='mae', optimizer='adam')
          return model
          
      def training(self,X_train,patience_num,epochs_num,batch_num):
          X_pos=X_train[1]
          #Y_train = np.ones((X_pos.shape[0],X_pos.shape[1]))+1
          Y_train= np.ones((X_pos.shape[0],X_pos.shape[1]))+1
          print(Y_train.shape)
          np.random.seed(0)
          model=self.build_model(X_pos.shape)
          checkpoint=ModelCheckpoint('temp_weight.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
          early = EarlyStopping(monitor="val_loss", patience=patience_num, verbose=1, mode="auto")
          callbacks_list = [early,checkpoint]    
          model.fit(X_train,
                       Y_train, 
                       epochs=epochs_num, 
                       batch_size=batch_num, validation_split=0.1, 
                       verbose=1, 
                       callbacks=callbacks_list)
          
          return model
          
            
      def get_the_user_vector(self,X_train,patience_num_para,epochs_num_para,batch_num_para):
          layer_name = 'user_vec_2d'
          model=self.training(X_train,
                              patience_num=patience_num_para,
                              epochs_num=epochs_num_para,
                              batch_num=batch_num_para)
          model.load_weights('temp_weight.hdf5')
          user_vec_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
          user_vec_output= user_vec_model.predict([X_train[0], X_train[1]])
          layer_name = "output"
          test_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
          #print(test_model.predict([X_train[0], X_train[1]]))
          print("user vec generated!!")
          for ctr,vec in enumerate(test_model.predict([X_train[0], X_train[1]])):
              if np.any(vec>2) or np.any(vec<-2):
                 print('counting no: ',ctr,'have numerical error')
          
          return user_vec_output
      
     
