#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import datetime
import operator
import math
from collections import Counter
from random import randrange
from math import sqrt
from timeit import default_timer as timer


# In[3]:


########read data
print('loading ./input/train.csv')
input_train = pd.read_csv('./input/train.csv', dtype = {
    'msno' : 'category',
    'song_id' : 'category',
    'source_system_tab' : 'category', # 9 diff values
    'source_screen_name' : 'category',
    'source_type' : 'category', # 12 diff values
    'target' : np.uint8})

print('loading ./input/test.csv')
input_test = pd.read_csv('./input/test.csv', dtype = {
    'msno' : 'category',
    'song_id' : 'category',
    'source_system_tab' : 'category',
    'source_screen_name' : 'category',
    'source_type' : 'category'}) 

print('loading ./input/songs.csv')
input_songs = pd.read_csv('./input/songs.csv', dtype = {
    'song_id' : 'category',
    #'song_length' : np.uint8,
    'genre_ids': 'category',
    'artist_name' : 'category',
    'composer' : 'category', #23% missing values
    'lyricist' : 'category',  #44% missing values
    'language' : 'category'})

print('loading ./input/members.csv')
input_members = pd.read_csv('./input/members.csv', dtype = {
    'msno' : 'category',
    'city' : 'category',
    'bd' : np.uint8, #losts of missing and misleading values
    'gender' : 'category', #lots of null value
    'registered_via' : 'category'},
    parse_dates=['registration_init_time', 'expiration_date'])

print('loading ./input/song_extra_info.csv')
input_songs_extra = pd.read_csv('./input/song_extra_info.csv')

print('Done reading files.')


# In[4]:


#count how many distinc values in the list
def countDistinct(arr):
    return len(Counter(arr).keys())


# In[5]:


print('Data preprocessing...')

########handle member data
print('Processing members data.')
input_members['length'] = input_members['expiration_date'].subtract(input_members['registration_init_time']).dt.days.astype(int)
input_members['reg_year'] = input_members['registration_init_time'].dt.year
input_members['reg_month'] = input_members['registration_init_time'].dt.month
input_members['reg_day'] = input_members['registration_init_time'].dt.day

input_members['exp_year'] = input_members['expiration_date'].dt.year
input_members['exp_month'] = input_members['expiration_date'].dt.month
input_members['exp_day'] = input_members['expiration_date'].dt.day
input_members = input_members.drop(['registration_init_time', 'expiration_date'], axis=1)
input_members = input_members.drop(['gender'], axis=1)
input_members = input_members.drop(['bd'], axis=1)
#input_members.loc[input_members.bd == 0, 'bd'] = input_members['bd'].mean()


# #classify membership length
#print(countDistinct(input_members['length']))
def membership_len_class(days):
    if days < 1:
        return 0
    elif days < 7:
        return 1
    elif days < 30:
        return 2
    elif days < 180:
        return 3
    elif days < 600:
        return 4
    elif days < 1500:
        return 5
    elif days < 2000:
        return 6
    elif days < 3500:
        return 7
    else:
        return 8
input_members['length_class'] = input_members['length'].apply(membership_len_class)
input_members['length_class']=(input_members['length_class']-input_members['length_class'].min())/(input_members['length_class'].max()-input_members['length_class'].min())
#print("# of element in input_members: ", Counter(input_members['length_class']))
input_members = input_members.drop(['length'], axis=1)


# In[6]:


######## handle songs data
print('Processing songs data.')
input_songs.song_length.fillna(200000, inplace=True)
input_songs.song_length = input_songs.song_length.astype(np.uint32)
def song_length_class(song_len):
    if int(song_len) < 20000:
        return 0
    elif int(song_len) < 600000:
        return round(int(song_len) / 120000) + 1
    else:
        return 7
input_songs['song_length_class'] = input_songs['song_length'].apply(song_length_class).astype(np.uint8)
input_songs['song_length_class']=(input_songs['song_length_class']-input_songs['song_length_class'].min())/(input_songs['song_length_class'].max()-input_songs['song_length_class'].min())
#print("# of element: ", Counter(input_songs['song_length_class']))
input_songs = input_songs.drop(['song_length'], axis=1)

#assign the first genre to songs that have multiple genre separate by | 
input_songs['genre_ids'] = input_songs['genre_ids'].apply(lambda genre: str(genre).split('|')[0])
input_songs['genre_ids'] = input_songs['genre_ids'].replace(np.nan, '0')
input_songs['language'] = input_songs['language'].replace(np.nan, '0')


# In[7]:


#handle missing values in lyricist, classify lyricist count
input_songs['lyricist'] = input_songs['lyricist'].replace(np.nan, 'no_lyricist')
def lyricist_count(lyricist):
    if lyricist == 'no_lyricist' or lyricist == 'no_composer' or lyricist == 'no_artist':
        return 0
    else:
        str_count=1
        str_list = ['|', '/', '\\', ';', ',', '&']
        for str in str_list:
            str_count = str_count + lyricist.count(str)
        return str_count
input_songs['lyricist_count'] = input_songs['lyricist'].apply(lyricist_count).astype(np.uint8)
#print("# of element: ", Counter(input_songs['lyricist_count']))
def lyricist_count_class(lyr):
    if lyr < 1:
        return 0
    elif lyr < 2:
        return 1
    elif lyr < 5:
        return 2
    elif lyr < 11:
        return 3
    else:
        return 4
input_songs['lyricist_count_class'] = input_songs['lyricist_count'].apply(lyricist_count_class).astype(np.uint8)
input_songs['lyricist_count_class']=(input_songs['lyricist_count_class']-input_songs['lyricist_count_class'].min())/(input_songs['lyricist_count_class'].max()-input_songs['lyricist_count_class'].min())
#print("# of element: ", Counter(input_songs['lyricist_count_class']))


# In[8]:


#handle composer
input_songs['composer'] = input_songs['composer'].replace(np.nan, 'no_composer')

input_songs['composer_count'] = input_songs['composer'].apply(lyricist_count).astype(np.uint8)
#print("# of element: ", Counter(input_songs['composer_count']))
input_songs['composer_count_class'] = input_songs['composer_count'].apply(lyricist_count_class).astype(np.uint8)
input_songs['composer_count_class']=(input_songs['composer_count_class']-input_songs['composer_count_class'].min())/(input_songs['composer_count_class'].max()-input_songs['composer_count_class'].min())
#print("# of element: ", Counter(input_songs['composer_count_class']))


# In[9]:


#handle artist_name
#some songs have "Various Artists" as artist_name?
input_songs['artist_name'] = input_songs['artist_name'].replace(np.nan, 'no_artist')

input_songs['artist_name_count'] = input_songs['artist_name'].apply(lyricist_count).astype(np.uint8)
#print("# of element: ", Counter(input_songs['artist_name_count']))
input_songs['artist_name_count_class'] = input_songs['artist_name_count'].apply(lyricist_count_class).astype(np.uint8)
input_songs['artist_name_count_class']=(input_songs['artist_name_count_class']-input_songs['artist_name_count_class'].min())/(input_songs['artist_name_count_class'].max()-input_songs['artist_name_count_class'].min())
#print("# of element: ", Counter(input_songs['artist_name_count_class']))


# In[10]:


#artist = composer
input_songs['artist_composer'] = (input_songs['artist_name'] == input_songs['composer']).astype(np.uint8)
input_songs['artist_composer']=(input_songs['artist_composer']-input_songs['artist_composer'].min())/(input_songs['artist_composer'].max()-input_songs['artist_composer'].min())
#artist = composer = lyricist
input_songs['artist_composer_lyricist'] = ((input_songs['artist_name'] == input_songs['composer']) &
                                            (input_songs['artist_name'] == input_songs['lyricist']) &
                                            (input_songs['composer'] == input_songs['lyricist'])).astype(np.uint8)
input_songs['artist_composer_lyricist']=(input_songs['artist_composer_lyricist']-input_songs['artist_composer_lyricist'].min())/(input_songs['artist_composer_lyricist'].max()-input_songs['artist_composer_lyricist'].min())


input_songs = input_songs.drop(['lyricist_count'], axis=1)
input_songs = input_songs.drop(['composer_count'], axis=1)
input_songs = input_songs.drop(['artist_name_count'], axis=1)


# In[11]:


######## handle song extra info data
print('Processing song extra info.')

# songs are all before 2017
def isrc_to_year(isrc): 
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan
def isrc_to_country(isrc):
    if type(isrc) == str:
        return isrc[:2]
    else:
        return np.nan
input_songs_extra['song_year'] = input_songs_extra['isrc'].apply(isrc_to_year)
song_year_mean = input_songs_extra['song_year'].mean(skipna=True)
input_songs_extra['song_year']=(input_songs_extra['song_year']-input_songs_extra['song_year'].min())/(input_songs_extra['song_year'].max()-input_songs_extra['song_year'].min())
input_songs_extra['song_country'] = input_songs_extra['isrc'].apply(isrc_to_country)
input_songs_extra.drop(['isrc', 'name',], axis=1, inplace=True)


# In[12]:


########merge data
print('Merging data.')

input_train = input_train.merge(input_songs, on='song_id', how='left')
input_test = input_test.merge(input_songs, on='song_id', how='left')

input_train = input_train.merge(input_members, on='msno', how='left')
input_test = input_test.merge(input_members, on='msno', how='left')

input_train = input_train.merge(input_songs_extra, on='song_id', how='left')
input_test = input_test.merge(input_songs_extra, on='song_id', how='left')

input_train['song_year'] = input_train['song_year'].replace(np.nan, song_year_mean)
input_test['song_year'] = input_test['song_year'].replace(np.nan, song_year_mean)

input_train['song_country'] = input_train['song_country'].replace(np.nan, 'none')
input_test['song_country'] = input_test['song_country'].replace(np.nan, 'none')

input_train['source_system_tab'] = input_train['source_system_tab'].replace(np.nan, '0')
input_train['source_screen_name'] = input_train['source_screen_name'].replace(np.nan, '0')
input_train['source_type'] = input_train['source_type'].replace(np.nan, '0')

input_test['source_system_tab'] = input_test['source_system_tab'].replace(np.nan, '0')
input_test['source_screen_name'] = input_test['source_screen_name'].replace(np.nan, '0')
input_test['source_type'] = input_test['source_type'].replace(np.nan, '0')

# In[13]:


######## features in train+test
print('Processing features in all samples.')
input_all = pd.concat([input_train.drop(['target'], axis=1), input_test.drop(['id'], axis=1)])


# In[14]:


#count how many times each user appears in all sample, then classify 
user_all_sample = input_all.groupby('msno').size()
# print(user_all_sample)
def user_frequency(msno):
    return user_all_sample[msno]
input_train['user_frequency'] = input_train['msno'].apply(user_frequency)
input_test['user_frequency'] = input_test['msno'].apply(user_frequency)
del user_all_sample

def user_frequency_class(fre):
    if fre < 20:
        return 0
    elif fre < 100:
        return 1
    elif fre < 500:
        return 2
    elif fre < 1000:
        return 3
    elif fre < 3000:
        return 4
    else:
        return 5
input_train['user_frequency_class'] = input_train['user_frequency'].apply(user_frequency_class)
input_train['user_frequency_class']=(input_train['user_frequency_class']-input_train['user_frequency_class'].min())/(input_train['user_frequency_class'].max()-input_train['user_frequency_class'].min())
input_test['user_frequency_class'] = input_test['user_frequency'].apply(user_frequency_class)
input_test['user_frequency_class']=(input_test['user_frequency_class']-input_test['user_frequency_class'].min())/(input_test['user_frequency_class'].max()-input_test['user_frequency_class'].min())

input_train = input_train.drop(['user_frequency'], axis=1)
input_test = input_test.drop(['user_frequency'], axis=1)
# print("# of element: ", Counter(input_train['user_frequency_class']))
# print("# of element: ", Counter(input_test['user_frequency_class']))


# In[15]:


#count how many times each song appears in all sample, then classify 
song_all_sample = input_all.groupby('song_id').size()
def song_frequency(id):
    return song_all_sample[id]
input_train['song_frequency'] = input_train['song_id'].apply(song_frequency)
input_test['song_frequency'] = input_test['song_id'].apply(song_frequency)
#del song_all_sample

def song_frequency_class(fre):
    if fre < 20:
        return 0
    elif fre < 100:
        return 1
    elif fre < 500:
        return 2
    elif fre < 1000:
        return 3
    elif fre < 3000:
        return 4
    elif fre < 9000:
        return 6
    else:
        return 5
input_train['song_frequency_class'] = input_train['song_frequency'].apply(song_frequency_class)
input_test['song_frequency_class'] = input_test['song_frequency'].apply(song_frequency_class)

input_train['song_frequency_class']=(input_train['song_frequency_class']-input_train['song_frequency_class'].min())/(input_train['song_frequency_class'].max()-input_train['song_frequency_class'].min())
input_test['song_frequency_class']=(input_test['song_frequency_class']-input_test['song_frequency_class'].min())/(input_test['song_frequency_class'].max()-input_test['song_frequency_class'].min())

input_train = input_train.drop(['song_frequency'], axis=1)
input_test = input_test.drop(['song_frequency'], axis=1)
# print("# of element: ", Counter(input_train['song_frequency_class']))
# print("# of element: ", Counter(input_test['song_frequency_class']))

#print(input_all.dtypes)
#print(input_train.dtypes)


# In[18]:


#some songs in input_test and input_train but not in input_songs, igonre them
#print(input_all['artist_name'].isna().sum()) #139 of them

input_train = input_train[pd.notnull(input_train['artist_name'])]
input_test = input_test[pd.notnull(input_test['artist_name'])]
input_all = input_all[pd.notnull(input_all['artist_name'])]
#print(input_all['artist_name'].isna().sum())

#count how many times each artist_name appears in all sample, then classify 
artist_name_all_sample = input_all.groupby('artist_name').size()
def artist_name_frequency(id):
    return artist_name_all_sample[id]
input_train['artist_frequency'] = input_train['artist_name'].apply(artist_name_frequency)
input_test['artist_frequency'] = input_test['artist_name'].apply(artist_name_frequency)
del artist_name_all_sample

def artist_frequency_class(fre):
    if fre < 20:
        return 0
    elif fre < 100:
        return 1
    elif fre < 500:
        return 2
    elif fre < 3000:
        return 3
    elif fre < 10000:
        return 4
    elif fre < 30000:
        return 5
    elif fre < 100000:
        return 6
    elif fre < 200000:
        return 7
    else:
        return 8
input_train['artist_frequency_class'] = input_train['artist_frequency'].apply(artist_frequency_class)
input_test['artist_frequency_class'] = input_test['artist_frequency'].apply(artist_frequency_class)

input_train['artist_frequency_class']=(input_train['artist_frequency_class']-input_train['artist_frequency_class'].min())/(input_train['artist_frequency_class'].max()-input_train['artist_frequency_class'].min())
input_test['artist_frequency_class']=(input_test['artist_frequency_class']-input_test['artist_frequency_class'].min())/(input_test['artist_frequency_class'].max()-input_test['artist_frequency_class'].min())

input_train = input_train.drop(['artist_frequency'], axis=1)
input_test = input_test.drop(['artist_frequency'], axis=1)
# print("# of element: ", Counter(input_train['artist_frequency_class']))
# print("# of element: ", Counter(input_test['artist_frequency_class']))


# In[19]:



#count how many times each composer appears in all sample, then classify 
composer_all_sample = input_all.groupby('composer').size()
def composer_frequency(id):
    return composer_all_sample[id]
input_train['composer_frequency'] = input_train['composer'].apply(composer_frequency)
input_test['composer_frequency'] = input_test['composer'].apply(composer_frequency)
del composer_all_sample

def composer_frequency_class(fre):
  if fre < 20:
    return 0
  elif fre < 100:
    return 1
  elif fre < 500:
    return 2
  elif fre < 3000:
    return 3
  elif fre < 10000:
    return 4
  elif fre < 30000:
    return 5
  elif fre < 100000:
    return 6
  elif fre < 200000:
    return 7
  elif fre < 500000:
    return 8
  elif fre < 1000000:
    return 9
  else:
    return 10
input_train['composer_frequency_class'] = input_train['composer_frequency'].apply(composer_frequency_class)
input_test['composer_frequency_class'] = input_test['composer_frequency'].apply(composer_frequency_class)

input_train['composer_frequency_class']=(input_train['composer_frequency_class']-input_train['composer_frequency_class'].min())/(input_train['composer_frequency_class'].max()-input_train['composer_frequency_class'].min())
input_test['composer_frequency_class']=(input_test['composer_frequency_class']-input_test['composer_frequency_class'].min())/(input_test['composer_frequency_class'].max()-input_test['composer_frequency_class'].min())

input_train = input_train.drop(['composer_frequency'], axis=1)
input_test = input_test.drop(['composer_frequency'], axis=1)
# print("# of element: ", Counter(input_train['composer_frequency_class']))
# print("# of element: ", Counter(input_test['composer_frequency_class']))


# In[ ]:


#count how many times each lyricist appears in all sample, then classify 
lyricist_all_sample = input_all.groupby('lyricist').size()
def lyricist_frequency(id):
  return lyricist_all_sample[id]
input_train['lyricist_frequency'] = input_train['lyricist'].apply(lyricist_frequency)
input_test['lyricist_frequency'] = input_test['lyricist'].apply(lyricist_frequency)
del lyricist_all_sample

input_train['lyricist_frequency_class'] = input_train['lyricist_frequency'].apply(composer_frequency_class)
input_test['lyricist_frequency_class'] = input_test['lyricist_frequency'].apply(composer_frequency_class)

input_train['lyricist_frequency_class']=(input_train['lyricist_frequency_class']-input_train['lyricist_frequency_class'].min())/(input_train['lyricist_frequency_class'].max()-input_train['lyricist_frequency_class'].min())
input_test['lyricist_frequency_class']=(input_test['lyricist_frequency_class']-input_test['lyricist_frequency_class'].min())/(input_test['lyricist_frequency_class'].max()-input_test['lyricist_frequency_class'].min())

input_train = input_train.drop(['lyricist_frequency'], axis=1)
input_test = input_test.drop(['lyricist_frequency'], axis=1)
# print("# of element: ", Counter(input_train['lyricist_frequency_class']))
# print("# of element: ", Counter(input_test['lyricist_frequency_class']))

# print(len(input_train['msno'])) 7377304
# print(len(input_test['id'])) 2556765

print('Processing pair features.')
#user song pair featrue

input_train['user_song_pair'] = input_train['msno'].astype(str) + input_train['song_id'].astype(str)
input_test['user_song_pair'] = input_test['msno'].astype(str) + input_test['song_id'].astype(str)
input_all = pd.concat([input_train.drop(['target'], axis=1), input_test.drop(['id'], axis=1)])

#count how many time a user listen to the same song 
user_song_pair_all_sample = input_all.groupby('user_song_pair').size()
def user_song_pair_frequency(id):
  return user_song_pair_all_sample[id]
input_train['user_song_pair_frequency'] = input_train['user_song_pair'].apply(user_song_pair_frequency)
input_test['user_song_pair_frequency'] = input_test['user_song_pair'].apply(user_song_pair_frequency)

input_train['user_song_pair_frequency']=(input_train['user_song_pair_frequency']-input_train['user_song_pair_frequency'].min())/(input_train['user_song_pair_frequency'].max()-input_train['user_song_pair_frequency'].min())
input_test['user_song_pair_frequency']=(input_test['user_song_pair_frequency']-input_test['user_song_pair_frequency'].min())/(input_test['user_song_pair_frequency'].max()-input_test['user_song_pair_frequency'].min())

del user_song_pair_all_sample
#print("# of element: ", Counter(input_train['user_song_pair_frequency']))
#print("# of element: ", Counter(input_test['user_song_pair_frequency']))
#either 1 or 2 times...


input_train = input_train.drop(['user_song_pair'],axis=1)
input_test = input_test.drop(['user_song_pair'],axis=1)

del input_all


#check the types of var and convert
########convert object to category
input_train = pd.concat([input_train.select_dtypes([], ['object']), 
                          input_train.select_dtypes(['object']).apply(pd.Series.astype, dtype = 'category')],
                           axis=1).reindex_axis(input_train.columns, axis=1)
input_test = pd.concat([input_test.select_dtypes([], ['object']),
                          input_test.select_dtypes(['object']).apply(pd.Series.astype, dtype = 'category')], 
                           axis=1).reindex_axis(input_test.columns, axis=1)

input_train['registered_via_cat'] = input_train['registered_via'].cat.codes
input_test['registered_via_cat'] = input_test['registered_via'].cat.codes
input_train['registered_via_cat']=(input_train['registered_via_cat']-input_train['registered_via_cat'].min())/(input_train['registered_via_cat'].max()-input_train['registered_via_cat'].min())
input_test['registered_via_cat']=(input_test['registered_via_cat']-input_test['registered_via_cat'].min())/(input_test['registered_via_cat'].max()-input_test['registered_via_cat'].min())


input_train['city_cat'] = input_train['city'].cat.codes
input_test['city_cat'] = input_test['city'].cat.codes
input_train['city_cat']=(input_train['city_cat']-input_train['city_cat'].min())/(input_train['city_cat'].max()-input_train['city_cat'].min())
input_test['city_cat']=(input_test['city_cat']-input_test['city_cat'].min())/(input_test['city_cat'].max()-input_test['city_cat'].min())

input_train['song_country_cat'] = input_train['song_country'].cat.codes
input_test['song_country_cat'] = input_test['song_country'].cat.codes
input_train['song_country_cat']=(input_train['song_country_cat']-input_train['song_country_cat'].min())/(input_train['song_country_cat'].max()-input_train['song_country_cat'].min())
input_test['song_country_cat']=(input_test['song_country_cat']-input_test['song_country_cat'].min())/(input_test['song_country_cat'].max()-input_test['song_country_cat'].min())

input_train['language_cat'] = input_train['language'].cat.codes
input_test['language_cat'] = input_test['language'].cat.codes
input_train['language_cat']=(input_train['language_cat']-input_train['language_cat'].min())/(input_train['language_cat'].max()-input_train['language_cat'].min())
input_test['language_cat']=(input_test['language_cat']-input_test['language_cat'].min())/(input_test['language_cat'].max()-input_test['language_cat'].min())

input_train['genre_ids_cat'] = input_train['genre_ids'].cat.codes
input_test['genre_ids_cat'] = input_test['genre_ids'].cat.codes
input_train['genre_ids_cat']=(input_train['genre_ids_cat']-input_train['genre_ids_cat'].min())/(input_train['genre_ids_cat'].max()-input_train['genre_ids_cat'].min())
input_test['genre_ids_cat']=(input_test['genre_ids_cat']-input_test['genre_ids_cat'].min())/(input_test['genre_ids_cat'].max()-input_test['genre_ids_cat'].min())

input_train['source_system_tab_cat'] = input_train['source_system_tab'].cat.codes
input_test['source_system_tab_cat'] = input_test['source_system_tab'].cat.codes
input_train['source_system_tab_cat']=(input_train['source_system_tab_cat']-input_train['source_system_tab_cat'].min())/(input_train['source_system_tab_cat'].max()-input_train['source_system_tab_cat'].min())
input_test['source_system_tab_cat']=(input_test['source_system_tab_cat']-input_test['source_system_tab_cat'].min())/(input_test['source_system_tab_cat'].max()-input_test['source_system_tab_cat'].min())

input_train['source_screen_name_cat'] = input_train['source_screen_name'].cat.codes
input_test['source_screen_name_cat'] = input_test['source_screen_name'].cat.codes
input_train['source_screen_name_cat']=(input_train['source_screen_name_cat']-input_train['source_screen_name_cat'].min())/(input_train['source_screen_name_cat'].max()-input_train['source_screen_name_cat'].min())
input_test['source_screen_name_cat']=(input_test['source_screen_name_cat']-input_test['source_screen_name_cat'].min())/(input_test['source_screen_name_cat'].max()-input_test['source_screen_name_cat'].min())

input_train['source_type_cat'] = input_train['source_type'].cat.codes
input_test['source_type_cat'] = input_test['source_type'].cat.codes
input_train['source_type_cat']=(input_train['source_type_cat']-input_train['source_type_cat'].min())/(input_train['source_type_cat'].max()-input_train['source_type_cat'].min())
input_test['source_type_cat']=(input_test['source_type_cat']-input_test['source_type_cat'].min())/(input_test['source_type_cat'].max()-input_test['source_type_cat'].min())

input_train['msno_cat'] = input_train['msno'].cat.codes
input_test['msno_cat'] = input_test['msno'].cat.codes
input_train['msno_cat']=(input_train['msno_cat']-input_train['msno_cat'].min())/(input_train['msno_cat'].max()-input_train['msno_cat'].min())
input_test['msno_cat']=(input_test['msno_cat']-input_test['msno_cat'].min())/(input_test['msno_cat'].max()-input_test['msno_cat'].min())

input_train['song_id_cat'] = input_train['song_id'].cat.codes
input_test['song_id_cat'] = input_test['song_id'].cat.codes
input_train['song_id_cat']=(input_train['song_id_cat']-input_train['song_id_cat'].min())/(input_train['song_id_cat'].max()-input_train['song_id_cat'].min())
input_test['song_id_cat']=(input_test['song_id_cat']-input_test['song_id_cat'].min())/(input_test['song_id_cat'].max()-input_test['song_id_cat'].min())



input_train = input_train.drop(['registered_via'], axis=1)
input_test = input_test.drop(['registered_via'], axis=1)
input_train = input_train.drop(['city'], axis=1)
input_test = input_test.drop(['city'], axis=1)
input_train = input_train.drop(['song_country'], axis=1)
input_test = input_test.drop(['song_country'], axis=1)
input_train = input_train.drop(['language'], axis=1)
input_test = input_test.drop(['language'], axis=1)
input_train = input_train.drop(['genre_ids'], axis=1)
input_test = input_test.drop(['genre_ids'], axis=1)
input_train = input_train.drop(['source_system_tab'], axis=1)
input_test = input_test.drop(['source_system_tab'], axis=1)
input_train = input_train.drop(['source_screen_name'], axis=1)
input_test = input_test.drop(['source_screen_name'], axis=1)
input_train = input_train.drop(['source_type'], axis=1)
input_test = input_test.drop(['source_type'], axis=1)
input_train = input_train.drop(['msno'], axis=1)
input_test = input_test.drop(['msno'], axis=1)
input_train = input_train.drop(['song_id'], axis=1)
input_test = input_test.drop(['song_id'], axis=1)
input_train = input_train.drop(['artist_name'], axis=1)
input_test = input_test.drop(['artist_name'], axis=1)
input_train = input_train.drop(['composer'], axis=1)
input_test = input_test.drop(['composer'], axis=1)
input_train = input_train.drop(['lyricist'], axis=1)
input_test = input_test.drop(['lyricist'], axis=1)

#check the types of var
print(input_train.dtypes)
# print(input_test.dtypes)
print('Finish processing features.')


#######split input_train into training and validation set

input_train_x = input_train.drop(['target'], axis=1)
input_train_y = input_train['target'].values
input_test_x = input_test.drop(['id'], axis=1)
input_test_id = input_test['id'].values

input_train_x = input_train_x.to_numpy()
input_test_x = input_test_x.to_numpy() # last step
print('Done.')

###append target to last col
# for i in range(len(input_train_x)):
#     np.append(input_train_x[i], input_train_y[i], axis=0)
input_train_x = np.concatenate((input_train_x, np.array(input_train_y)[:,None]),axis=1)

print(input_train_x[0])
print(len(input_train_x[0]))


lenA = round(len(input_train_x)*0.66)
lenB = len(input_train_x)-lenA
# print("len A is {}  len B is {}".format(lenA, lenB))
# print(lenA+lenB)
trainingSet = input_train_x[:int(lenA)]
testingSet = input_train_x[int(lenA):]

# y_train = input_train_y[:int(lenA)]
# y_test = input_train_y[int(lenA):]
print("training length: {}  testing length {}".format(len(trainingSet), len(testingSet)))


# In[86]:


sub_trainingSet = trainingSet[:57600]
sub_testingSet = testingSet[:28800]
print("sub_trainingSet length: {}  subtestingSet length {}".format(len(sub_trainingSet), len(sub_testingSet)))


def compute_E_dist(x, xi, length):
    dist = 0.0
    for i in range(length):
        dist += pow(float(x[i]) - float(xi[i]), 2)
    return math.sqrt(dist)

def getNeighbor(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = compute_E_dist(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0]) # take a row, because they are tuple
    #print("getNeighbor: {}".format(neighbors))
    return neighbors
    

def getResponse(neighbors):
    targetVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1] # ground truth
        if response in targetVotes:
            targetVotes[response] += 1
        else:
            targetVotes[response] = 1
    # sort based on votes
    sortedVotes = sorted(targetVotes.items(), key=operator.itemgetter(1), reverse=True)
    #print("getResponse: {}".format(sortedVotes[0][0]))
    return sortedVotes[0][0]

def getAccuracy(testingSet, predictions):
    correct = 0
    for x in range(len(testingSet)):
        if testingSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(tesingtSet))) * 100.0



######## train the model with different K values
predictions = []
k = 1 # change to 3 later
start_prediction = timer()
for x in range(len(sub_testingSet)):
    #print("predicting...now x is {}...".format(x))
    neighbors = getNeighbor(sub_trainingSet, sub_testingSet[x], k)
    result = getResponse(neighbors)
    predictions.append(result)
print("Prediction time: {}".format(timer()-start_prediction))
print("Finish prediction.")
accuracy = getAccuracy(sub_testingSet, predictions)
print("Accuracy: {}".format(accuracy))

