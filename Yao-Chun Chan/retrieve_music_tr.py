import pandas as pd
import numpy as np
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pickle

data_path=''
train = pd.read_csv(data_path + 'train.csv', dtype={'msno' : 'category',
                                                'source_system_tab' : 'category',
                                                  'source_screen_name' : 'category',
                                                  'source_type' : 'category',
                                                  'target' : np.uint8,
                                                  'song_id' : 'category'})

#df=pd.concat([train,test])

print(len(train.groupby(['song_id']).groups.keys()))

embedding_path='all_doc2vecKKBOX.pickle'
with open(embedding_path, 'rb') as f:
      u = pickle._Unpickler(f)
      u.encoding = 'latin1'
      vec_dict = u.load()

def get(key):
    if key in vec_dict:
        return vec_dict[key]
    else:
        return np.zeros((50,))
    
    


 
        
music_vec=np.zeros((len(train),50))
ordered_keys=sorted(list(train.groupby(['song_id']).groups.keys()))
#length=int(len(ordered_keys)/2)
for ctr,key in enumerate(ordered_keys):
    print(" time:", ctr+1,'\r',end='')
    search_boolean=np.where(train['song_id']==key)
    length=len(music_vec[search_boolean])
    music_vec[search_boolean]=np.asarray([ get(key) for i in range(length)])

    
np.save('music_vec_tr_len.npy',music_vec)
