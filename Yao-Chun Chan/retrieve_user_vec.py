import pandas as pd
import numpy as np
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pickle
from Sampling import *

data_path=''
train = pd.read_csv(data_path + 'train.csv', dtype={'msno' : 'category',
                                                'source_system_tab' : 'category',
                                                  'source_screen_name' : 'category',
                                                  'source_type' : 'category',
                                                  'target' : np.uint8,
                                                  'song_id' : 'category'})
test = pd.read_csv(data_path + 'test.csv', dtype={'msno' : 'category',
                                                'source_system_tab' : 'category',
                                                'source_screen_name' : 'category',
                                                'source_type' : 'category',
                                                'song_id' : 'category'})



df=pd.concat([train,test])

print(len(df.groupby(['msno']).groups.keys()))

embedding_path='user_embedding_dict.pickle'
with open(embedding_path, 'rb') as f:
      u = pickle._Unpickler(f)
      u.encoding = 'latin1'
      vec_dict = u.load()

    
    
#PNVG=pos_neg_vec_generator()
        
user_input_vec=np.zeros((len(df),50))

user_embedding='User_vec_idx.pickle'
with open(user_embedding,'rb') as f:
     user_dict=pickle.load(f)

martix_b=np.zeros((len(vec_dict),50))
idxs=[]
for ctr,key in enumerate(vec_dict.keys()):
    martix_b[ctr]=vec_dict[key]
    idxs.append(key)
    
diff=-len(user_dict)+len(vec_dict)
martix_a=np.zeros((diff,50))
ctr=0
not_in_idx=[]
for key in vec_dict.keys():
    if key not in user_dict:
        martix_a[ctr]=vec_dict[key]
        not_in_idx.append(key)
        ctr+=1
    
#similarity matrix    
similarity = np.dot(martix_a, martix_b.T)
mag_vec_a = np.sqrt(np.sum(martix_a**2, axis=1))
mag_vec_b = np.sqrt(np.sum(martix_b**2, axis=1))
norm_den = np.outer(mag_vec_a, mag_vec_b)
#mag_matrix=np.dot(mag_vec,mag_vec.T)  
cosine=similarity/norm_den
di = np.diag_indices(diff)
cosine[di]=-1

print(cosine.shape)

assert cosine.shape[1]==len(idxs)

def get(key_in,cosine_matrix,idxs,not_in_idx,user_dict):
    if key_in in user_dict:
        return user_dict[key_in]
    else:
        key_idx=not_in_idx.index(key_in)
        sim_dict={}
        for i in range(0,cosine_matrix.shape[1],1):
            sim_dict[idxs[i]]=cosine_matrix[key_idx][i]
            
             
        #print(sim_dict)
        temp_stack=sorted(sim_dict.items(),key=lambda x: x[1])
        #print(temp_stack)
        key_subtitute=temp_stack.pop(-1)[0]
        while key_subtitute not in user_dict:
              key_subtitute=temp_stack.pop(-1)[0]
                
        return user_dict[key_subtitute]



for ctr,key in enumerate(df.groupby(['msno']).groups.keys()):
    print(" time:", ctr+1,'\r',end='')
    search_boolean=np.where(df['msno']==key)
    length=len(user_input_vec[search_boolean])
    user_input_vec[search_boolean]=np.asarray([ get(key,cosine,idxs,not_in_idx,user_dict) for i in range(length)])

    
np.save('user_input_vec.npy',user_input_vec)
