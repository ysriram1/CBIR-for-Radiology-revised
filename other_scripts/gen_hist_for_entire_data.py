# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:07:46 2018

@author: syarlag1
"""
# this script generates a histogram of the cosine distances between the 

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from scipy import stats
import pickle

from CBIR_functions import * # reads in all the necessary functions for CBIR

os.chdir('C:/Users/syarlag1.DPU/Desktop/paper/CBIR-for-Radiology/functions_for_database_CBIR')

# change these params as needed
images_folder = 'C:/Users/syarlag1.DPU/Desktop/CBIR-for-Radiology/functions_for_database_CBIR/all_images_sample'
images_percent_for_kmeans = 0.08
cluster_count = 50
save_files = False
ellipse = False
threshold_ = 0.3 # what I chosen based on visual inspection of results
img_count = 1000 # number of images to randomly pick

# Read in csv files ONLY if JUST running the online part (no offline part)
cluster_centers = pd.read_csv('cluster_centers.csv', index_col=0)
database_BoW_temp = pd.read_csv('database_BoW.csv')
database_BoW_temp.index = database_BoW_temp.iloc[:,0]; del database_BoW_temp['Unnamed: 0']
database_BoW_dict = database_BoW_temp.T.to_dict('list'); del database_BoW_temp

# pick images from folder randomly
all_imgs = os.listdir(images_folder); random.shuffle(all_imgs)
imgs_sub = all_imgs[:img_count]

dist_store = []

for img in imgs_sub:
    
    try:

        query_image_arr = plt.imread(images_folder+'/'+ img)
        
        # PART G
        query_SIFT_feats = sift(query_image_arr, ellipse=False)
        
        # PART H
        query_BoW_arr =  bag_of_words(query_SIFT_feats, cluster_centers, query=True)
        
        # PART I (using cosine distance NOT similarity)
        dist_dict = calc_dist_sim(query_BoW_arr, database_BoW_dict, method='cosine')
        
        dist_store += list(dist_dict.values())
        
    except: 
        continue

os.chdir('C:/Users/syarlag1.DPU/Desktop/paper/histogram_info')

with open('dist_store.p','wb') as s:
    pickle.dump(dist_store, s)

res = stats.relfreq(dist_store, numbins=100)
x = res.lowerlimit + np.linspace(0, res.binsize*res.frequency.size, res.frequency.size)

fig = plt.figure(figsize=(20, 16))
ax = fig.add_subplot(1, 1, 1)
ax.bar(x, res.frequency, width=res.binsize)
ax.set_title('Distribution of Cosine Distances')
ax.set_xlim([x.min(), x.max()])

plt.hist(dist_store, bins = 100)