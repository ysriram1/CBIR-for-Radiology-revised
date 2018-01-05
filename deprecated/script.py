from __future__ import print_function
# uses py 2.7

if __name__ == '__main__':

	import cv2
	import os
 	#os.chdir('/Users/Sriram/Desktop/DePaul/CBIR-for-Radiology/images_sample')
	os.chdir('C:/Users/syarlag1.DPU/Desktop/CBIR-for-Radiology')
	from calc_image_association import *
	from dicom_image_download_extraction import extract_pixels_and_modality
	from read_images_gen_feats import *
	import matplotlib.pyplot as plt
	seed = 99

	image_dict = read_images_from_folder('./images_sample/') # make sure '/' is included at end!
	#_,image_dict, image_mod_dict = extract_pixels_and_modality('./images_store/') # comment out if not dicom

	# first get the keypoints (local features)
	image_local_feats_dict = add_image_features(image_dict, kind = 'sift', ellipse=False)

	# next get the global features
	image_global_feats_dict = add_image_features(image_dict, kind = 'global', ellipse=False)

	# read in a query image
	query_image_arr = cv2.imread('./images_sample/3_11') # change as needed

	## Display image:
	plt.figure()

	plt.title('3_11') # change if query image has changed

	plt.imshow(query_image_arr, cmap="gray")

	plt.show()

	# calculate local and global feats for the query image
	query_image_local_feats = image_descriptors.sift(query_image_arr, ellipse=False)

	query_image_global_feats = image_descriptors.globalFeats(query_image_arr, ellipse=False)

	image_dist_dict_local = calc_dist_sim(query_image_local_feats, image_local_feats_dict,
										method='bag_of_words', k=50, dist_measure='Cosine')

	image_dist_dict_global = calc_dist_sim(query_image_global_feats, image_global_feats_dict,
										method = 'global', dist_measure='Cosine')

	# combine the two distance/similarities
	image_dist_dict = combine_measures(image_dist_dict_local, image_dist_dict_global, weighting=0.5)

	result_image_id_list = return_images(image_dist_dict, image_dict, k=10, distance=True, show=False)

	# iterate through the result images and diplay them
	for image_id in result_image_id_list:

		image_location = './images_sample/'+image_id

		# display images
		img = cv2.imread(image_location)

		plt.figure()

		plt.imshow(img, cmap='gray')

		plt.title(image_location)

		plt.show()


### Visualize the images in a mds projection plot of the images ###
import numpy as np
from numpy.matlib import repmat

# from the calc image association function:

# apply k-means to find the centroids
k = 10

train_feats = np.concatenate(image_feats_dict.values())

# TODO: Kmeans calculation takes the largest amount of time, everything else is fast
k_means = KMeans(n_clusters=k, random_state=seed).fit(train_feats)

cluster_centers = k_means.cluster_centers_

# TODO: loops are slow --> replace with numpy matrix magic
# find closest center to each image keypoint and generate histogram
image_hist_dict = {}

for image_id, each_image in image_feats_dict.items():

	image_hist_dict[image_id] = [0] * k

	for keypoint in each_image:

		diff = cluster_centers - repmat(keypoint,len(cluster_centers), 1)

		euclidean_dists = np.apply_along_axis(np.linalg.norm, 1, diff, ord=2)

		image_hist_dict[image_id][np.argmin(euclidean_dists)] += 1. # add to frequency of correponding center

X_to_project_unnormalized = np.array(image_hist_dict.values())

X_to_project = X_to_project_unnormalized / repmat(X_to_project_unnormalized.sum(1), m=k, n=1).T

Y_for_color = np.array([int(x.split('_')[0]) for x in image_hist_dict.keys()])

color_lookup_dict = {x[1]:x[0] for x in zip(range(len(set(Y_for_color))),set(Y_for_color))}

# perform MDS with 2 dimensions

seed = 99

from sklearn.manifold import MDS

mds = MDS(random_state=seed, dissimilarity='euclidean')

X_projected = mds.fit_transform(X_to_project)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(100, 50))

plt.xlim([-400,150])

for i,data in enumerate(X_projected):

	colormap = plt.cm.Dark2.colors

	Y = Y_for_color[i]

	Y_rank = color_lookup_dict[Y]; print(str(Y), str(Y_rank))

	if Y_rank >= 7: Y_rank = 7; print('change', str(Y_rank)); continue

	plt.scatter(data[0],data[1], color='white') # change to color=colormap[Y_rank]
	#ax.annotate(Y_for_color[i], xy=data)


for i,xy in enumerate(X_projected):

	ax.annotate(image_hist_dict.keys()[i], xy=xy, size=10)

fig.savefig('C:/Users/syarlag1.DPU/Desktop/CBIR-for-Radiology/mds_proj_full_image_names_normalized.png', dpi=100)
#plt.show()

#### number of keypoints #####
image_feats_count_dict = {}

for image_id in image_feats_dict.keys():

	image_feats_count_dict[image_id] = image_feats_dict[image_id].shape[0]



############### DEALING WITH THE DICOM IMAGES ################
# lots of code repeated from top
# find all the image features instances with None and delete them
# None correspond to dicom pixel arrays that arent in the right format (should be only 3 of them)

_,image_dict, image_mod_dict = extract_pixels_and_modality('./images_store/')

image_feats_dict = add_image_features(image_dict, kind = 'sift', ellipse=False)

for image_id in image_feats_dict.keys():
	# remove images from dictionary where we didnt extract image features
    if image_feats_dict[image_id] is None:

        del image_feats_dict[image_id]
        del image_mod_dict[image_id]

# find the average number of keypoints for each modality
mod_avg_kps_dict = {}
mod_count_dict = {}
mod_total_kps_dict = {}

for image_id in image_feats_dict.keys():

    if image_mod_dict[image_id] not in mod_total_kps_dict.keys():

		mod_total_kps_dict[image_mod_dict[image_id]] = 0.
		mod_count_dict[image_mod_dict[image_id]] = 0.

    mod_total_kps_dict[image_mod_dict[image_id]] += len(image_feats_dict[image_id])
    mod_count_dict[image_mod_dict[image_id]] += 1

for mod in mod_count_dict.keys():

	mod_avg_kps_dict[mod] = mod_total_kps_dict[mod]/mod_count_dict[mod]

# use the keypoints in the image_feats dict to perform Bag of Words
k = 10

train_feats = np.concatenate(image_feats_dict.values())

k_means = KMeans(n_clusters=k, random_state=seed).fit(train_feats)

cluster_centers = k_means.cluster_centers_

# find closest center to each image keypoint and generate histogram
image_hist_dict = {}

for image_id, each_image in image_feats_dict.items():

	image_hist_dict[image_id] = [0] * k

	for keypoint in each_image:

		diff = cluster_centers - repmat(keypoint,len(cluster_centers), 1)

		euclidean_dists = np.apply_along_axis(np.linalg.norm, 1, diff, ord=2)

		image_hist_dict[image_id][np.argmin(euclidean_dists)] += 1. # add to frequency of correponding center

X_to_project_unnormalized = np.array(image_hist_dict.values())

X_to_project = X_to_project_unnormalized / repmat(X_to_project_unnormalized.sum(1), m=k, n=1).T

Y_for_color = image_mod_dict.values() # these are the colors (the modality)

# perform MDS with 2 dimensions

seed = 99

from sklearn.manifold import MDS

mds = MDS(random_state=seed, dissimilarity='euclidean')

X_projected = mds.fit_transform(X_to_project)


#### Plotting ####
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(100, 50))

for i,data in enumerate(X_projected):

    colormap_1 = plt.cm.tab20b.colors
    colormap_2 = plt.cm.Set1.colors

    # TODO: This needs to be fixed
    if image_mod_dict.values()[i] in ['CT','MR','CR','DX','NM']: # since one colormap cant hold too many colors

        index = list(set(image_mod_dict.values())).index(image_mod_dict.values()[i])
        plt.scatter(data[0],data[1], color=colormap_1[index])

    else:

        index = list(set(image_mod_dict.values())).index(image_mod_dict.values()[i])
        plt.scatter(data[0],data[1], color=colormap_2[index]) # change to color=colormap[Y_rank]
        #ax.annotate(Y_for_color[i], xy=data)


for i,xy in enumerate(X_projected):

    ax.annotate(Y_for_color[i], xy=xy, size=10)

fig.savefig('C:/Users/syarlag1.DPU/Desktop/CBIR-for-Radiology/mds_proj_for_dicom_color.png', dpi=100)

######################## CLASSIFICATION #############################################
_,image_dict, image_mod_dict = extract_pixels_and_modality('./images_store/')

image_feats_dict = add_image_features(image_dict, kind = 'sift', ellipse=False)

for image_id in image_feats_dict.keys():
	# remove images from dictionary where we didnt extract image features
    if image_feats_dict[image_id] is None:

        del image_feats_dict[image_id]
        del image_mod_dict[image_id]

# use the keypoints in the image_feats dict to perform Bag of Words
k = 10

train_feats = np.concatenate(image_feats_dict.values())

k_means = KMeans(n_clusters=k, random_state=seed).fit(train_feats)

cluster_centers = k_means.cluster_centers_

# find closest center to each image keypoint and generate histogram
image_hist_dict = {}

for image_id, each_image in image_feats_dict.items():

	image_hist_dict[image_id] = [0] * k

	for keypoint in each_image:

		diff = cluster_centers - repmat(keypoint,len(cluster_centers), 1)

		euclidean_dists = np.apply_along_axis(np.linalg.norm, 1, diff, ord=2)

		image_hist_dict[image_id][np.argmin(euclidean_dists)] += 1. # add to frequency of correponding center

X_unnormalized = np.array(image_hist_dict.values())

X = X_unnormalized / repmat(X_unnormalized.sum(1), m=k, n=1).T

Y = image_mod_dict.values() # these are the colors (the modality)

# Split into test and train (stratified)
from sklearn.model_selection import train_test_split

# stratify=None means stratification is done based on class
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
													random_state=seed, stratify=None)
# SVM
from sklearn.svm import SVC
clf = SVC()
clf.fit(X_train, Y_train)

SVM_score = clf.score(X_test, Y_test)

# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, Y_train)

LDA_score = clf.score(X_test, Y_test)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(c=0.5)
clf.fit(X_train, Y_train)

logistic_score = clf.score(X_test, Y_test)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split=5)
clf.fit(X_train, Y_train)

tree_score = clf.score(X_test, Y_test)
