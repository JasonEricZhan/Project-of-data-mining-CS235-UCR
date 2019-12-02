from NN_model import *
import pandas as pd
import numpy as np
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split



pre_train=pd.read_csv('pre_train.csv')
pre_test=pd.read_csv('pre_test.csv')
#pre_train=pre_train.drop(columns=['msno','song_id'])
#cat_features=['source_system_tab', 'source_screen_name',
#'source_type', 'genre_ids', 'artist_name', 'composer', 'lyricist',
#'language', 'city', 'gender', 'registered_via']



song_cat=['song_id','genre_ids', 'artist_name', 'composer', 'lyricist','language']
source_cat=['source_system_tab', 'source_screen_name','source_type']
member_cat=['msno','city', 'gender', 'registered_via']
song_num=['song_length','lyricists_count','composer_count','artist_count','artist_composer',
         'artist_composer_lyricist', 'song_lang_boolean', 'smaller_song',
         'count_song_played', 'count_artist_played','is_featured','song_year']
member_num=['bd','expiration_date','membership_days',
            'registration_year','registration_month', 'registration_date', 'expiration_year',
            'expiration_month']

a=np.load('music_vec_tr_len.npy')
b=np.load('music_vec_ts_len.npy')

a_ave=np.sum(a,axis=0)/(len(a)-5700)
b_ave=np.sum(b,axis=0)/(len(b)-1250)



user_input_vec=np.load('user_input_vec.npy')




import copy
a_temp=copy.deepcopy(a)
b_temp=copy.deepcopy(b)

for idx in np.where(a_temp==np.zeros((50,)))[0]:
    a[idx]=a_ave+np.random.uniform(-0.25,0.25)
    
for idx in np.where(b_temp==np.zeros((50,)))[0]:
    b[idx]=b_ave+np.random.uniform(-0.25,0.25)
#music_input_vec=np.concatenate((a, b), axis=0)



from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
song_num_np=np.vstack((pre_train[song_num].values,pre_test[song_num].values))
song_num_np=sc.fit_transform(song_num_np)


member_num_np=np.vstack((pre_train[member_num].values,pre_test[member_num].values))
member_num_np=sc.fit_transform(member_num_np)


for ctr,col in enumerate(song_num):
    pre_train[col]=song_num_np[:len(pre_train),ctr]
    pre_test[col]=song_num_np[len(pre_train):,ctr]
    
for ctr,col in enumerate(member_num):
    pre_train[col]=song_num_np[:len(pre_train),ctr]
    pre_test[col]=song_num_np[len(pre_train):,ctr]
    
    


#num_features=list(set(pre_train.columns)-set(cat_features))
DAW=model_deepAndWide()
model=DAW.build_model_multi_channel(
                pre_train,
                [song_cat,song_num],
                [member_cat,member_num],
                [source_cat]
                ,50)




y=pd.read_csv('label.csv')

model_history=DAW.train_multi_channel(pre_train,[song_cat,song_num],[member_cat,member_num],
                    [source_cat],user_input_vec[:len(y)],a,y.values[:,-1])

ans=DAW.predict_multi_channel(pre_test,[song_cat,song_num],[member_cat,member_num],
                    [source_cat],user_input_vec[len(y):],b)

np.save("pred.npy",ans)
