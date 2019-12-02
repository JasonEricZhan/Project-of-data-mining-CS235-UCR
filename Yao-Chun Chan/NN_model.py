from Train_user_vec_model import *
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.merge import concatenate, dot, add, multiply, subtract
from keras.regularizers import l1, l2, l1_l2
import tensorflow as tf
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
#import os





class model_deepAndWide():
        def __init__(self):
             pass

            
            
        def component(self,data,ft,embedding_features,num_features):   
            if embedding_features is None:
               tmp_input_num = Input(shape=(len(num_features),), name=ft+'_numerical_features_input')
               #x=Dense(64, name='num_dense')(tmp_input_num)
               x_numerical=BatchNormalization()(tmp_input_num)
               return x_numerical,tmp_input_num
            
            if num_features is None:
               embedding_inputs = []
               embedding_outputs = []
               for ctr,feature_name in enumerate(embedding_features):
                  tmp_input = Input(shape=(1,), name='embedding_features_input_'+feature_name)
                  max_val=max(data[feature_name])+1
                  embedding_channel = Embedding(max_val, 12, 
                                              input_length=1,
                                              trainable=True,
                                              name='embedding_features_'+feature_name)(tmp_input)
                  Flatten_channel=Flatten()(embedding_channel)
                  embedding_inputs.append(tmp_input)
                  embedding_outputs.append(Flatten_channel)
               x_category=concatenate(embedding_outputs, name=ft+'_category_embedding_concate')
               return x_category,embedding_inputs
            
            embedding_inputs = []
            embedding_outputs = []
            for ctr,feature_name in enumerate(embedding_features):
                tmp_input = Input(shape=(1,), name='embedding_features_input_'+feature_name)
                max_val=max(data[feature_name])+1
                embedding_channel = Embedding(max_val, 10, 
                                              input_length=1,
                                              trainable=True,
                                              name='embedding_features_'+feature_name)(tmp_input)
                Flatten_channel=Flatten()(embedding_channel)
                embedding_inputs.append(tmp_input)
                embedding_outputs.append(Flatten_channel)
    
            x_numerical = Input(shape=(len(num_features),), name=ft+'_numerical_features_input')
             x_numerical=tmp_input_num
            
            x_category=concatenate(embedding_outputs, name=ft+'_category_embedding_concate')
            x_category=embedding_concat
            
            return x_numerical,x_category,tmp_input_num,embedding_inputs
            
                    
        
        def build_model_multi_channel(self,data,song,member,source,user_vec_dim):
            song_cat=song[0]
            song_num=song[1]
            member_cat=member[0]
            member_num=member[1]
            source_cat=source[0]
            
            #x_numerical=Activation('linear')(x)
            
            
            x_song_num,x_song_cat,song_input_num,song_cat_inputs=self.component(data,'song',song_cat,song_num)
            x_member_num,x_member_cat,member_input_num,member_cat_inputs=self.component(data,'member',member_cat,member_num)
            x_source_cat,source_input_cat=self.component(data,'source',source_cat,None)

            
            
            
            song_channel=concatenate([x_song_num,x_song_cat], name='song_channel')
            song_channel=BatchNormalization()(song_channel)
            song_channel=Dense(72,kernel_regularizer=l1(1e-6), name='song_channel_dense')(song_channel)
            song_channel=Activation('elu')(song_channel)
            member_channel=concatenate([x_member_num,x_member_cat], name='member_channel')
            member_channel=BatchNormalization()(member_channel)
            member_channel=Dense(50,kernel_regularizer=l1(1e-6), name='member_channel_dense')(member_channel)
            member_channel=Activation('elu')(member_channel)
            
            x_source_channel=BatchNormalization()(x_source_cat)
            x_source_channel=Dense(24,kernel_regularizer=l1(1e-6),  name='source_channel_dense')(x_source_channel)
            x_source_channel=Activation('elu')(x_source_channel)
            #x_source_channel=x_source_cat
            
            
            member_channel_lower_24=Dense(24,
                                          kernel_regularizer=l1(1e-6), 
                                          name='member_channel_lower_24d')(member_channel)
            member_channel_lower_24=Activation('linear')(member_channel_lower_24)
            
            song_channel_lower_24=Dense(24, 
                                        kernel_regularizer=l1(1e-6),
                                        name='song_channel_lower_24d')(song_channel)
            song_channel_lower_24=Activation('linear')(song_channel_lower_24)
            
            sq_song_lower_24= Lambda(lambda x: K.square(x),name = "square_song")(song_channel_lower_24)
            sq_member_lower_24= Lambda(lambda x: K.square(x),name = "square_member")(member_channel_lower_24)
            square_source= Lambda(lambda x: K.square(x),name = "square_source")(x_source_channel)
            
            multi_1=multiply([member_channel_lower_24,song_channel_lower_24])
            multi_1=Lambda(lambda x: x*2)(multi_1)
            multi_2=multiply([member_channel_lower_24,x_source_channel])
            multi_2=Lambda(lambda x: x*2)(multi_2)
            multi_3=multiply([song_channel_lower_24,x_source_channel])
            multi_3=Lambda(lambda x: x*2)(multi_3)
            
            CTR_inner_interact=add([sq_song_lower_24,
                                    sq_member_lower_24,
                                    square_source,
                                    multi_1,
                                    multi_2,
                                    multi_3])
            
            
            
            CTR_channel=concatenate([song_channel,
                                     member_channel,
                                     x_source_channel,
                                     CTR_inner_interact], name='CTR_channel')
     

    
            tmp_input_user_vec = Input(shape=(user_vec_dim,), name='user_vec_input')
            tmp_input_music_vec = Input(shape=(user_vec_dim,), name='music_vec_input')
            
            
            input_user_vec_lower_24=Dense(24,
                                          kernel_regularizer=l1(1e-6),
                                          name='input_user_vec_lower_24d')(tmp_input_user_vec)
            input_user_vec_lower_24=Activation('linear')(input_user_vec_lower_24)
            
            input_music_vec_lower_24=Dense(24, 
                                           kernel_regularizer=l1(1e-6),
                                           name='tmp_input_music_vec_lower_24d')(tmp_input_music_vec)
            input_music_vec_lower_24=Activation('linear')(input_music_vec_lower_24)
            
            sq_user_vec_lower_24= Lambda(lambda x: K.square(x),
                                         name = "square_user")(input_user_vec_lower_24)
            sq_music_vec_lower_24= Lambda(lambda x: K.square(x),
                                          name = "square_music")(input_music_vec_lower_24)
            
            
            multi_last_1=multiply([input_user_vec_lower_24,input_music_vec_lower_24])
            multi_last_1=Lambda(lambda x: x*2)(multi_last_1)
            multi_last_2=multiply([input_music_vec_lower_24,x_source_channel])
            multi_last_2=Lambda(lambda x: x*2)(multi_last_2)
            multi_last_3=multiply([input_user_vec_lower_24,x_source_channel])
            multi_last_3=Lambda(lambda x: x*2)(multi_last_3)
            rec_interact=add([sq_user_vec_lower_24,
                              sq_music_vec_lower_24,
                              square_source,
                                    multi_last_1,
                                    multi_last_2,
                                    multi_last_3])
        
            dot_reommend= dot([tmp_input_user_vec,tmp_input_music_vec],axes=1)
            
    
           
            
            byAttribute_user_member= multiply([input_user_vec_lower_24,
                                               member_channel_lower_24])                               
            byAttribute_song_music= multiply([input_music_vec_lower_24,
                                              song_channel_lower_24])
            Cross_user_song= multiply([input_user_vec_lower_24,
                                       song_channel_lower_24])                               
            Cross_member_music= multiply([input_music_vec_lower_24,
                                           member_channel_lower_24])
                             
            
            
          
            
 
           
            pred_=BatchNormalization()(CTR_channel)
            x=Dense(256, name='FCN_1_with_Bathnorm')(pred_)
            pred0=Activation('elu')(x)
            x=Dense(128, name='FCN_2_with_Bathnorm')(pred0)
            x=BatchNormalization()(x)
            x=Activation('elu')(x)
            #x=Dense(128,kernel_regularizer=l1(1e-4), name='FCN_3_with_Bathnorm')(x)
            #x=BatchNormalization()(x)
            #x=Activation('elu')(x)
            #x=Activation('elu')(x)
            x=concatenate([x,
                           byAttribute_user_member,
                           byAttribute_song_music,
                           Cross_user_song,
                           Cross_member_music,
                           tmp_input_user_vec,
                           tmp_input_music_vec,
                           dot_reommend],axis=1)
            x=BatchNormalization()(x)
            x=Dropout(0.4)(x)
            x=Dense(1,kernel_regularizer=l1(1e-5), name='output')(x)
            preds=Activation('sigmoid')(x)
    
            model = Model(inputs=[song_input_num]+
                                 song_cat_inputs+
                                 [member_input_num]+
                                 member_cat_inputs+
                                 source_input_cat+
                                 [tmp_input_user_vec]+
                                 [tmp_input_music_vec], outputs=preds)
            print(model.summary()) 
            sgd = optimizers.SGD(lr=0.0005, decay=1e-10, momentum=0.9, nesterov=True)
            #opt = RMSprop(lr=0.0001)
            model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['acc'])
            self.model=model
            
            return self.model
        
        
        def train_multi_channel(self,data,song,member,source,user_vec,music_vec,y):
            song_cat=song[0]
            song_num=song[1]
            member_cat=member[0]
            member_num=member[1]
            source_cat=source[0]
            
            feed_list=[]
            data_song_np=data[song_num].values
            feed_list.append(data_song_np)
            data_song_np=data[song_cat].values
            for i in range(data_song_np.shape[1]):
                feed_list.append(data_song_np[:,i])
                
            data_member_np=data[member_num].values
            feed_list.append(data_member_np)
            data_member_np=data[member_cat].values
            for i in range(data_member_np.shape[1]):
                feed_list.append(data_member_np[:,i])
            
            data_source_np=data[source_cat].values
            for i in range(data_source_np.shape[1]):
                feed_list.append(data_source_np[:,i])
            #feed_list.append(music_vec)
            feed_list.append(user_vec)
            feed_list.append(music_vec)
            
            checkpoint=ModelCheckpoint('weight_DeepAndWide.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            early = EarlyStopping(monitor="val_loss", patience=10, verbose=1, mode="min")
            callbacks_list = [early,checkpoint]    
            history =self.model.fit(feed_list,y, 
                       epochs=72, 
                       batch_size=128, validation_split=0.2,  
                       callbacks=callbacks_list,verbose=1)
            
            return  history
      
            
            
        
            
            
     
        def predict_multi_channel(self,data,song,member,source,user_vec,music_vec):
            song_cat=song[0]
            song_num=song[1]
            member_cat=member[0]
            member_num=member[1]
            source_cat=source[0]
            
            feed_list=[]
            data_song_np=data[song_num].values
            feed_list.append(data_song_np)
            data_song_np=data[song_cat].values
            for i in range(data_song_np.shape[1]):
                feed_list.append(data_song_np[:,i])
                
            data_member_np=data[member_num].values
            feed_list.append(data_member_np)
            data_member_np=data[member_cat].values
            for i in range(data_member_np.shape[1]):
                feed_list.append(data_member_np[:,i])
            
            data_source_np=data[source_cat].values
            for i in range(data_source_np.shape[1]):
                feed_list.append(data_source_np[:,i])
            #feed_list.append(music_vec)
            feed_list.append(user_vec)
            feed_list.append(music_vec)

            self.model.load_weights('weight_DeepAndWide.hdf5')
            return self.model.predict(feed_list)
          
        
        
            
         
