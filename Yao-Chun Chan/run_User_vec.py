from Train_user_vec_model import *





with open('All_pos_vec_idx.pickle', 'rb') as f:
     pos_vec_dict=pickle.load(f)





with open('All_neg_vec_idx.pickle', 'rb') as f:
     neg_vec_dict=pickle.load(f)
        
  
ordered_keys=sorted(list(pos_vec_dict.keys()))
dim2=50
embedding_dim=50
pos_vec=np.zeros((len(ordered_keys),dim2,embedding_dim))
neg_vec=np.zeros((len(ordered_keys),dim2,embedding_dim))

for ctr,key in enumerate(ordered_keys):
    pos_vec[ctr]=pos_vec_dict[key]
for ctr,key in enumerate(ordered_keys):
    neg_vec[ctr]=neg_vec_dict[key]

X_train=[pos_vec,neg_vec]
UR=User_score_model()
user_vec=UR.get_the_user_vector(X_train,patience_num_para=15,epochs_num_para=200,batch_num_para=64)


user_vec_dict={}
for ctr,key in enumerate(ordered_keys):
    user_vec_dict[key]=user_vec[ctr]
    
    


with open('User_vec_idx.pickle', 'wb') as f:
     pickle.dump(user_vec_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
