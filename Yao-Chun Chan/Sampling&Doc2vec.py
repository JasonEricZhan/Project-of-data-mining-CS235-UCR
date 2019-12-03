import pandas as pd

import pandas as pd
import numpy as np
import pickle 

from Sampling import *

VG=Vec_generator()
VG.generate_VecDict(songs=pd.read_csv('songs.csv'),songs_extra=pd.read_csv('song_extra_info.csv'))
All_pos_dict,All_neg_dict=VG.generate_pos_set_True_neg_set_song_id_batch(pd.read_csv('train.csv'))


with open('All_pos_dict.pickle', 'wb') as f:
     pickle.dump(All_pos_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        
       
with open('All_neg_dict.pickle', 'wb') as f:
     pickle.dump(All_neg_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
     
     

with open('All_pos_dict.pickle', 'rb') as f:
     positive_sample=pickle.load(f)

with open('All_neg_dict.pickle', 'rb') as f:
     negative_sample=pickle.load(f)


VG=pos_neg_vec_generator()
out_pos_arr,out_neg_arr,pos_idx_dict,neg_idx_dict=VG.generate_pos_neg_vec(positive_sample,negative_sample)






with open('All_pos_vec_idx.pickle', 'wb') as f:
     pickle.dump(pos_idx_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('All_neg_vec_idx.pickle', 'wb') as f:
     pickle.dump(neg_idx_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        

np.save('pos_vec.npy',out_pos_arr)
        
np.save('neg_vec.npy',out_neg_arr)

