# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import pickle 
from gensim.models.doc2vec import TaggedDocument,Doc2Vec
import operator

#from 
import re
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, GRU,GlobalAveragePooling1D,GlobalMaxPooling1D
from keras.layers import TimeDistributed, RepeatVector, Input, subtract, Lambda, Conv1D, add
from keras import backend as K
from keras.engine import Layer
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from keras import initializers, regularizers, constraints, activations
from keras.layers import InputSpec,RepeatVector
from keras.layers import Bidirectional

from VAE import *

from numpy import dot
from numpy.linalg import norm
import gc

import copy 


class Vec_generator():
      def __init__(self):
          pass
          
      
            
      def generate_pos_set_True_neg_set_song_id(self,user_id,pands_table):
          temp=pands_table.iloc[np.where(pands_table['msno']==user_id)]
          pos, True_neg = temp[temp['target']==1], temp[temp['target']==0]
          pos_np, True_neg_np = pos['song_id'].values, True_neg['song_id'].values
          pos_np_idx, True_neg_np_idx = pos.index.tolist(),True_neg.index.tolist()
                     
          return pos_np, True_neg_np, pos_np_idx, True_neg_np_idx
                               
      def generate_pos_set_True_neg_set_song_id_batch(self,pands_table):
          all_pos_dict={}
          all_True_neg_dict={}
          user_id_list=sorted(pands_table.groupby('msno').groups.keys())
                               
          for ctr,user_id in enumerate(user_id_list):
              pos_np, True_neg_np, pos_np_idx, True_neg_np_idx=self.generate_pos_set_True_neg_set_song_id(user_id,pands_table) 
              pos_idxAndorder={}
              for ctr1,music_id in enumerate(pos_np):
                  pos_idxAndorder[music_id]=pos_np_idx[ctr1]
              neg_idxAndorder={}
              for ctr1,music_id in enumerate(True_neg_np):
                  neg_idxAndorder[music_id]=True_neg_np_idx[ctr1]
                
              all_pos_dict[user_id]=pos_idxAndorder
              all_True_neg_dict[user_id]=neg_idxAndorder   
              print('generate_pos_set_True_neg_set_song_id_batch User number: ', ctr+1, '/', len(user_id_list), '\r', end='')
                     
          return all_pos_dict,all_True_neg_dict
                               
        
        
      
    
      def isrc_to_year(self,isrc):
          if type(isrc) == str:
              if int(isrc[5:7]) > 17:
                 return 1900 + int(isrc[5:7])
              else:
                 return 2000 + int(isrc[5:7])
          else:
              return np.nan
        
          songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
       
                         
      def preprocessing(self,data,feature_name=['genre_ids',
                                                'artist_name',
                                                'composer',
                                                'lyricist',
                                                'language',
                                                'song_year']):
                               
                               
          for col in feature_name:
              #print(col)
              data[col]=data[col].astype('str')
              data[col]=data[col].fillna('NAN')                 
              data[col]=str(col)+' : '+data[col]
          
          data['article_type']=' '
          data['article_type']=data['article_type']                
          for col in feature_name:
              data['article_type']=data['article_type']+" , "+data[col]
                               
          return data['song_id'].values,data['article_type'].values
                               
      def generate_VecDict(self,
                           songs,
                           songs_extra,
                           save_dict=True,
                           embedding_method='doc2vec',
                           path_setting={'save_embeding_path':'all_doc2vecKKBOX.pickle',
                                         'save_model_path':'all_doc2vecKKBOX'},
                           other_concat=['song_length']):
                               
          if embedding_method=='doc2vec':
             #implement Doc2vec embedding   
             save_embeding_path=path_setting['save_embeding_path']
             save_model_path=path_setting['save_model_path']
                
             songs_extra['song_year'] = songs_extra['isrc'].apply(self.isrc_to_year)   
             song_data=songs.merge(songs_extra,on='song_id', how='left')
                             
             positin2idx,song_data_np=self.preprocessing(song_data)
            
                    
             #print(article_arr_con[0])
             #print([doc for i, doc in enumerate(article_arr_con)])
             
             documents = [TaggedDocument(re.split(r'[,|:||]',str(doc)), [i]) for i, doc in enumerate(song_data_np)]
             num1=49
             num2=6
             num3=1
             print("parameter for doc2vec, vector size:",num1,"windows:",num2,"min_count:",num3)
             model = Doc2Vec(documents,vector_size=num1,window=num2, min_count=num3,min_alpha=0.01,negative=5, workers=48)
             print("min count:", model.min_count)
             print("corpus count:", model.corpus_count)
             print("word count:", len(model.wv.vocab))
             model.train(documents,epochs=25,total_examples=model.corpus_count)
             model.save(save_model_path)
             
             #model= Doc2Vec.load(save_model_path)
             
             vec_dict={}
             if other_concat:
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                temp=np.zeros((len(song_data),len(other_concat)))
                for ctr,col in enumerate(other_concat):
                    song_data[col]=song_data[col].fillna(0)
                    temp[:,ctr]=scaler.fit_transform(song_data[col].values.reshape(-1, 1)).reshape(-1,)
                    print("scaler: ",scaler)
                for ctr,item_vec in enumerate(model.docvecs.vectors_docs):
                    idx=positin2idx[ctr] 
                    vec_dict[idx]=np.concatenate((item_vec,temp[ctr]), axis=0)
                    
             else:
                for ctr,item_vec in enumerate(model.docvecs.vectors_docs):
                    idx=positin2idx[ctr] 
                    vec_dict[idx]=item_vec
                
               
                
             with open(save_embeding_path, 'wb') as handle:
                pickle.dump(vec_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            
             return vec_dict    
            
          
             
      def generate_VecDict_User(self,data,latent_dim = 50,batch_size = 128, epochs= 100):
          original_dim= data.shape[1]
          input_shape = (original_dim, )
          intermediate_dim = int(original_dim/2)
          epsilon_std = 1.0
          x, eps, z_mu, x_pred = vae_arc(original_dim, intermediate_dim, latent_dim)
          vae = Model(inputs=[x, eps], outputs=x_pred)
          vae.compile(optimizer='adam', loss=nll)
          
          from sklearn.model_selection import train_test_split
          from sklearn.preprocessing import MinMaxScaler
          scaler    = MinMaxScaler()
          pre_embedding_train_norm   = scaler.fit_transform(data) 
          X_train, X_test, y_train, y_test = train_test_split(pre_embedding_train_norm, pre_embedding_train_norm, 
                                                    test_size=0.2, random_state=42)
          filepath   ="weights.hdf5"
          checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
          callbacks_list = [checkpoint]
          vae.fit(X_train, X_train,
          epochs=epochs,
          batch_size=batch_size,
          callbacks=callbacks_list,
          validation_data=(X_test, X_test))
  
          encoder = Model(x, z_mu)
          z_df= encoder.predict(pre_embedding_train_norm, batch_size=batch_size)
        
          msno_key=data['msno']
          res_dict={}
          for ctr,item in enumerate(z_df):
              res_dict[msnomsno_key.iloc[ctr]]=item

          with open('user_embedding_dict.pickle', 'wb') as f:
               pickle.dump(res_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            
            
            

     
        



    
    
class pos_neg_vec_generator(Vec_generator):
      def __init__(self,embedding_path='all_doc2vecKKBOX.pickle',user_embedding='user_embedding_dict.pickle'):
          print("loading.....")
          with open(embedding_path, 'rb') as f:
                   u = pickle._Unpickler(f)
                   u.encoding = 'latin1'
                   vec_dict = u.load()
                   
          print("loading music embedding completed.....")
          with open(user_embedding,'rb') as f:
               user_dict=pickle.load(f)
            
          
                
          print("loading user embedding completed.....")
          self.vec_dict=vec_dict   
          self.vec_user_dict=user_dict    
          self.temp_similarity=None
            

     
      def infer_UserVec(self,user_key,applied_dict):   
          vec_user_dict=copy.deepcopy(self.vec_user_dict)
          
          max_idx=0
          b=vec_user_dict[user_key]
          sim_dict={}
          for key in vec_user_dict.keys():
              if key is not user_key:
                 cos_sim = self.cos_sim(vec_user_dict[key],b)
                 sim_dict[key]=cos_sim
             
          temp_stack=sorted(sim_dict.items(),key=lambda x: x[1])
          idx=temp_stack.pop(-1)[0]
          while idx not in applied_dict:
                idx=temp_stack.pop(-1)[0]
                
          return idx
      

      
      def coldStart_byContent_sampling(self,target_music_list,refer_music_list,user_id,upper_bound):
          #count the windows, be more uniform
          # inputs minority_music_list majority_music_list are all [[music_id,order],.....]
          if type(refer_music_list)==list:
             refer_idx_list=refer_music_list
          else:
             refer_idx_list=[]
             for item in refer_music_list:
                 refer_idx_list.append(item[0])
                
          target_idx_list=[]
          for item in target_music_list:
              target_idx_list.append(item[0])
          
          
          
          if len(target_music_list)<upper_bound:
             all_set=set(self.vec_dict.keys())-set(refer_idx_list)-set(target_idx_list)
             sampling_ctr=0
             while len(target_idx_list)<upper_bound and sampling_ctr<1:
                   diff=upper_bound-len(target_idx_list)
                   sampling_candidate=np.random.choice(np.asarray(list(all_set)), diff*3, replace=False)
                   print("sampling_candidate len:",len(sampling_candidate),'\r', end='')
                   print("")
                   target_sim_dict={}
                   refer_sim_dict={}
                   #choosed_idx_list=[]
                   for music_idx_s in sampling_candidate:
                       vec=self.vec_dict[music_idx_s]
                       #print("sampling_candidate len:",len(sampling_candidate)) 
                       #print("minor_idx_list len:",len(minor_idx_list))
                       #print("major_idx_list len:",len(major_idx_list))
                       target_sim_sum=0
                       for music_idx_m in target_idx_list:
                           target_sim_sum+=self.cos_sim(self.vec_dict[music_idx_m],vec)
                       refer_sim_sum=0
                       for music_idx_m in refer_idx_list:
                           refer_sim_sum+=self.cos_sim(self.vec_dict[music_idx_m],vec)
                       #min_target_sim=sorted(target_sim_dict.items(),key=lambda x:x[1])[0][1]
                       #max_refer_sim=sorted(refer_sim_dict.items(),key=lambda x:x[1])[-1][1]
                       #print("max refer sim:", max_refer_sim ,"min target sim:", min_target_sim)
                       #print("minor id",music_idx_s)
                       if target_sim_sum/len(target_idx_list) > refer_sim_sum/len(refer_idx_list):
                          print("number of item been choose:", len(target_idx_list))
                          target_idx_list.append(music_idx_s)
                       if len(target_idx_list)==upper_bound:
                          break
                   if len(target_idx_list)==upper_bound:
                          break
                   
                   print(' time of sampling: ',sampling_ctr, '\r', end='')
                   sampling_ctr+=1
                   all_set=all_set-set(target_idx_list)
              
             #print(target_idx_list)
              
             
             diff=upper_bound-len(target_idx_list)
             if diff>0:
                sampling_candidate=np.random.choice(np.asarray(list(all_set)), diff, replace=False)
                for music_idx_s in sampling_candidate:
                    target_idx_list.append(music_idx_s)
                
             del all_set
             gc.collect()
          else:
             target_music_list=[ item[0] for item in target_music_list]
             target_idx_list=self.normal_sampling(target_music_list,upper_bound)
                        
          return target_idx_list                           
                            
           
          
                            
     
      def coldStart_byUser_sampling(self,music_list,user_id,upper_bound,pos):         
          #count the windows, be more uniform
          #exclude=[]
          sampling_ctr=0
          while len(music_list)<upper_bound:
                possible_candidate=self.user_similarity_query(user_id,pos)
                music_list=list(music_list)+list(possible_candidate)
                music_list=sorted(music_list,key=lambda x:x[1])  #[(music_id,order),....]
                #print(' length of music_list:', len(music_list), '\r', end='')
                print(' time of user sampling: ',sampling_ctr, '\r', end='')
                print(' length of music list:',len(music_list))
                sampling_ctr+=1
                
          
             
          music_list=[ item[0] for item in music_list]
          print("majority_music_list after user preprocess len: ",len(music_list))
          self.temp_similarity=None
        
          return self.normal_sampling(music_list,upper_bound)
          
            
    
      def normal_sampling(self,music_list,upper_bound):    
            
          period=int(len(music_list)/upper_bound)
          index=[]
          for i in range(0,len(music_list),period):
              base=i
              end=base+period
              index.append([base,end])
          
          if period==1:
             candidate_item_list=np.random.choice(music_list,upper_bound, replace=False)
             res=[ item for item in candidate_item_list]
             
          else:
             res=[]
             ctr=0        
             while len(res)<upper_bound:
                  base=index[ctr][0]
                  end=index[ctr][1]
                  print("base,end",base,end, '\r', end='')
                  if end > len(music_list):
                     end=len(music_list)
                  #print(np.asarray(majority_music_list[base:end]))
                  candidate=np.random.choice(music_list[base:end], 1, replace=False)
                  #print(candidate)
                  res.append(candidate[0])
                  #print('len:',len(res))
                  
                  ctr+=1
                
                
          #self.majority_res=res
        
          print('len of res:',len(res))
                
          return res
                
    
      
      def user_similarity_query(self,user_id,pos):
          
            
            
          vec_user_dict=copy.deepcopy(self.vec_user_dict)
         
            
          if self.temp_similarity:
             max_idx=self.temp_similarity_list.pop(-1)[0]
             ctr=0
             
            
          else:
             max_idx=0
             b=vec_user_dict[user_id]
             sim_dict={}
             for key in vec_user_dict.keys():
                 if key is not user_id:
                    cos_sim = self.cos_sim(vec_user_dict[key],b)
                    sim_dict[key]=cos_sim
                
                
             self.temp_similarity_list=sorted(sim_dict.items(),key=lambda x: x[1])
             max_idx=self.temp_similarity_list.pop(-1)[0]
             ctr=0
              
                
                
          if pos:
             while len(self.pos[max_idx])<2:
                   max_idx=self.temp_similarity_list.pop(-1)[0]
                   print("finding pos active user",'\r', end='')                      
                   
             return self.pos[max_idx].items()
          else:
             while len(self.neg[max_idx])<2:
                   max_idx=self.temp_similarity_list.pop(-1)[0]
                   print("finding neg active user",'\r', end='')
                         
             return self.neg[max_idx].items()
              
       
      
      def cos_sim(self,a,b):
          return dot(a,b)/(norm(a)*norm(b))
                               

      def generate_pos_neg_vec(self,user2music_pos_record_dict,
                                user2music_neg_record_dict,
                                embedding_dim=50,
                                upper_bound=50,
                                threhold=None,following_time=False):
          #suppose input already filtered by solar meta data movie set
          #XXX_res_music_list do not have order label
          dim2=upper_bound
          if (threhold is not None) and threhold>dim2:
             dim2=threhold
          #sampling
          
          total_length=len(user2music_pos_record_dict)
          
          out_pos_arr=np.zeros((total_length,dim2,embedding_dim))
          out_neg_arr=np.zeros((total_length,dim2,embedding_dim))
          
          #filter
          
          new_user2music_pos_record_dict={}
          for key in user2music_pos_record_dict:
              new_user2music_pos_record_dict[key]={}
              for music_id in user2music_pos_record_dict[key]:
                 print(' original len:',len(user2music_pos_record_dict[key]),'\r', end='')
                 if music_id in self.vec_dict:
                    new_user2music_pos_record_dict[key][music_id]=user2music_pos_record_dict[key][music_id]   
                 print(' new len:',len(new_user2music_pos_record_dict[key]),'\r', end='')
          
          
          new_user2music_neg_record_dict={}
          for key in user2music_neg_record_dict:
              new_user2music_neg_record_dict[key]={}
              for music_id in user2music_neg_record_dict[key]:
                 print(' original len:',len(user2music_neg_record_dict[key]),'\r', end='')
                 if music_id in self.vec_dict:
                    new_user2music_neg_record_dict[key][music_id]=user2music_neg_record_dict[key][music_id]
                 print(' new len:',len(new_user2music_neg_record_dict[key]),'\r', end='')
          
                    
                    
          self.pos=new_user2music_pos_record_dict
          self.neg=new_user2music_neg_record_dict
        
        
          ##change common variable session:
          key_list=list(self.vec_user_dict.keys())
          for key in key_list:   
              if (key not in self.pos) or (key not in self.neg):
                 del self.vec_user_dict[key]
          
          print("original user number:", len(key_list))
          print("active user number:", len(self.vec_user_dict))
          ##session end
            
          
          count_dim1=0                     
          pos_idx2vec_dict={}    
          neg_idx2vec_dict={}    
          ordered_keys=sorted(list(new_user2music_pos_record_dict.keys()))
          for ctr1,user_id in enumerate(ordered_keys):
              pos_dict=copy.deepcopy(new_user2music_pos_record_dict[user_id])
              neg_dict=copy.deepcopy(new_user2music_neg_record_dict[user_id])
              np.random.seed(0)
              if len(pos_dict)>=len(neg_dict):
                    
                 if len(neg_dict)==0:
                    neg_res_music_list=self.coldStart_byUser_sampling(neg_dict.items(),user_id,upper_bound,pos=False)
                    print("neg done")
                    pos_res_music_list=self.coldStart_byContent_sampling(pos_dict.items(),neg_res_music_list,user_id,upper_bound)
                    print("pos done")
                 else:
                    pos_res_music_list=self.coldStart_byContent_sampling(pos_dict.items(),neg_dict.items(),user_id,upper_bound)
                    print("pos done")
                    neg_res_music_list=self.coldStart_byContent_sampling(neg_dict.items(),pos_dict.items(),user_id,upper_bound) 
                    print("neg done")
              else:
                 if len(pos_dict)==0:
                    pos_res_music_list=self.coldStart_byUser_sampling(pos_dict.items(),user_id,upper_bound,pos=True)
                    print("pos done")
                    neg_res_music_list=self.coldStart_byContent_sampling(neg_dict.items(),pos_res_music_list,user_id,upper_bound)
                    print("neg done")
                 else:
                    neg_res_music_list=self.coldStart_byContent_sampling(neg_dict.items(),pos_dict.items(),user_id,upper_bound)
                    print("neg done")
                    pos_res_music_list=self.coldStart_byContent_sampling(pos_dict.items(),neg_dict.items(),user_id,upper_bound) 
                    print("pos done")
                 
              #output: list of [music_id,order] 
                 
              #print('pos length:', len(pos_res_music_list))
            
              #print('neg length:', len(neg_res_music_list))
                
                
              
              with open('test_1_idx.pickle', 'wb') as f:
                   pickle.dump(pos_res_music_list, f, protocol=pickle.HIGHEST_PROTOCOL)
                
              
              with open('test_2_idx.pickle', 'wb') as f:
                   pickle.dump(neg_res_music_list, f, protocol=pickle.HIGHEST_PROTOCOL)
  
              #exit(0)
              assert len(pos_res_music_list)==len(neg_res_music_list) and len(neg_res_music_list)==dim2
              count_dim2=0
              music_idx_list=[]
              for music_idx in pos_res_music_list:
                  #music_idx_list.append(music_idx)
                  out_pos_arr[count_dim1,count_dim2,:]=self.mapping_quary(music_idx,embedding_dim,sampling=False)        
                  count_dim2+=1
              
              pos_idx2vec_dict[user_id]=out_pos_arr[count_dim1]     #music_idx_list

              
              count_dim2=0
              music_idx_list=[]
              for music_idx in neg_res_music_list:
                  #music_idx_list.append(music_idx)
                  out_neg_arr[count_dim1,count_dim2,:]=self.mapping_quary(music_idx,embedding_dim,sampling=False)        
                  count_dim2+=1
               
              neg_idx2vec_dict[user_id]=out_neg_arr[count_dim1]      #music_idx_list
      
              count_dim1+=1
        
             

              print("")                     
              print(' User number: ', ctr1+1, '/', total_length, '\r', end='')
              print("")   
                        
              
                   
          print(" vectors generated!! ")
          return out_pos_arr,out_neg_arr,pos_idx2vec_dict,neg_idx2vec_dict
                               
          
       

      def mapping_quary(self,music_id,embedding_dim,sampling):
          #quary the vector, will consider the shell 
          if music_id not in self.vec_dict:
             print("unfound:", music_id)
             #exit(0)
                
          if not sampling:
             try:
                vec=self.vec_dict[music_id][:embedding_dim] #force to get the first 'embedding_dim' of vector
             except:
                vec=np.zeros((1,embedding_dim))
                return vec

          else:
             try:
                vec=self.vec_dict[music_id][:embedding_dim] #force to get the first 'embedding_dim' of vector
             except:
                return None
             
          return vec
          
              
