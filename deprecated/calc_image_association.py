from __future__ import print_function

import cv2
import numpy as np
from numpy.matlib import repmat# for repmat
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine

seed = 99 # for randomized computations


# returns a vector of distance to each of the training arrays from query array
# using jensen shannon divergence
def jensen_shannon_div(query_arr,train_mat):

	# normalize arrays so that they become probability distributions
	query_arr = query_arr/float(np.sum(query_arr))

	train_mat = np.divide(train_mat.T, np.sum(train_mat,1)).T

	query_mat = repmat(query_arr, len(train_mat), 1)

	mat_sum = 0.5*(query_mat + train_mat)

	D1 = query_mat * np.log2(np.divide(query_mat, mat_sum))

	D2 = train_mat * np.log2(np.divide(train_mat, mat_sum))

	# convert all nans to 0

	D1[np.isnan(D1)] = 0

	D2[np.isnan(D2)] = 0

	JS_mat = 0.5 * (np.sum(D1,1) + np.sum(D2,1))

	return JS_mat

# returns a vector of distance to each of the training arrays from query array
# using Euclidean Distance
# TODO: remove loop and vectorize the calculation
def euclidean_dist(query_arr, train_mat):

    query_arr = np.array(query_arr)

    train_mat = np.array(train_mat)

    eu_dist = [0]*train_mat.shape[0]

    for i, train_arr in enumerate(train_mat):

        eu_dist[i] = np.sqrt(np.sum(np.square(query_arr - train_arr))) # calc eu dist

    return eu_dist

# returns a vector of distnace to each of the training arrays from query array
# using cosine distance
# TODO: remove loop and vectorize the calculation
def cosine_dist(query_arr, train_mat):

    query_arr = np.array(query_arr)

    train_mat = np.array(train_mat)

    cosine_dist = [0]*train_mat.shape[0]

    for i, train_arr in enumerate(train_mat):

        cosine_dist[i] =  cosine(query_arr, train_arr) # calc cosine dist

    return cosine_dist

# returns keys of the top-k images
# euclidean and cosine used for histogram and sift has its own method using bag of words approach
# k used if the bag_of_words approach is used
# TODO: if cluster_kmeans is false and method='bag of words', then hierarchical clustering will be applied

def calc_dist_sim(query_feats,image_feats_dict, method='bag_of_words', cluster_kmeans=True, k=10, dist_measure='Euclidean'):

	image_sim_dist_dict = {}

	if method == 'cosine':

		# get the histograms of the database images
		mat = image_feats_dict.values()

		# L-2 norms of the train image featuers
		row_norms = np.apply_along_axis(np.linalg.norm, 1, mat)

		query_norm = np.linalg.norm(query_feats)

		# quick matrix multiplication
		sim_mat = np.einsum('ji,i->j', mat, query_feats.T)

		# first divide by query norm
		sim_mat = sim_mat/query_norm

        # then by individual row norms
		sim_mat = sim_mat/row_norms[:,None]

        # add to the dictionary
		image_sim_dist_dict = dict((key, val) for key,val in zip(image_feats_dict.keys(),sim_mat))


	if method == 'euclidean':

		# get the histograms of the database images
		mat = image_feats_dict.values()

		diff = mat - repmat(query_feats,len(mat), 1)

		# L-2 norms of the train image featuers
		euclidean_dists = np.apply_along_axis(np.linalg.norm, 1, diff, ord=2)

		# add to the dictionary
		image_sim_dist_dict = dict((key, val) for key,val in zip(image_feats_dict.keys(),euclidean_dists))

	# uses hamming distance as metric to find the closest keypoints
	if method == 'orb':

		# init the matching method
		matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

		# find the match between query and each training image
		for image in image_feats_dict:

			matches = matcher.match(query_feats, image_feats_dict[image])

			# calculate the matching distance
			distances = [x.distance for x in matches]

			# assign query, image dist as the average
			image_sim_dist_dict[image] = sum(distances)/(len(distances)+1)


	# currently works for sift
	if method == 'match_count':

		# initilizalize the matcher
		bf = cv2.BFMatcher() # L2-norm is default

		for image_id, each_image in image_feats_dict.items():

			# Match descriptors using knn matching -- tune k as needed
			matches = bf.knnMatch(query_feats, each_image, k=5)

			# count number of matches whose ratio is > 0.75
			match_count = 0

			for m,n in matches:

				if m.distance < 0.75*n.distance:

					match_count += 1# add to sim dict

			image_sim_dist_dict[image_id] = match_count

	# for feature vectors obtained form the global feature extraction
	if method == 'global':

		if dist_measure == 'JS':

			# use jesen-shannon divergence to find distance to each image from query
			distances = jensen_shannon_div(np.array(query_feats), np.array(image_feats_dict.values()))

		if dist_measure == 'Euclidean':

			distances = euclidean_dist(np.array(query_feats), np.array(image_feats_dict.values()))

		if dist_measure == 'Cosine': # note that it is cosine distance NOT similarity

			distances = cosine_dist(np.array(query_feats), np.array(image_feats_dict.values()))

		return dict((key, val) for key,val in zip(image_feats_dict.keys(),distances))

	# bag of visual words approach to retreive images
	if method == 'bag_of_words':

		# apply k-means to find the centroids
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

		query_hist = np.array([0] * k)

		# convert query_feats into the histogram like above
		for keypoint in query_feats:

			diff = cluster_centers - repmat(keypoint,len(cluster_centers), 1)

			euclidean_dists = np.apply_along_axis(np.linalg.norm, 1, diff)

			query_hist[np.argmin(euclidean_dists)] += 1.

		if dist_measure == 'JS':

			# use jesen-shannon divergence to find distance to each image from query
			distances = jensen_shannon_div(np.array(query_hist), np.array(image_hist_dict.values()))


		if dist_measure == 'Euclidean':

			distances = euclidean_dist(np.array(query_hist), np.array(image_hist_dict.values()))


		if dist_measure == 'Cosine': # note that it is cosine distance NOT similarity

			distances = cosine_dist(np.array(query_hist), np.array(image_hist_dict.values()))


		image_sim_dist_dict = dict((key, val) for key,val in zip(image_feats_dict.keys(),distances))

	return image_sim_dist_dict


# return the overall distance or similarity between the query and image
# this applies to images that have a seperate global and local features
def combine_measures(local_dict, global_dict, weighting=0.5):

	combined_dict = {}

	for image_id in local_dict.keys():

		combined_dict[image_id] = weighting * local_dict[image_id] + (1-weighting) * global_dict[image_id]

	return combined_dict


# display retrieved images
def return_images(image_sim_dist_dict, image_dict, k=5, distance=True, show=True):

	result_image_id_list = []

	# sort based on whether sim or dist measure
	sorted_list = sorted(image_sim_dist_dict.items(), key=lambda x: x[1], reverse= not distance)

	for i in range(k):

		image_id = sorted_list[i][0]

		image_name = 'result ' + str(i) + ': ' + image_id

		if show: cv2.imshow(image_name, image_dict[image_id])

		result_image_id_list.append(image_id)

	return result_image_id_list
