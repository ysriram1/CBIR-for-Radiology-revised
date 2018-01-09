# A list of functions for CBIR for Radiology images
import cv2
import numpy as np
from numpy.matlib import repmat# for repmat
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
import dicom
import os
import mahotas as mh
from ftplib import FTP
import dicom
import urllib
import os
from bs4 import BeautifulSoup
from PIL import Image
from numpy import uint8, double
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

seed = 99 # for randomized computations

################## FUNCTIONS BELOW TO DOWNLOAD IMAGES FROM SERVER ##############
# iterates through the images in the remote server and downloads all the image
# files or only the dicom files based on the only_dicom parameter. Images are
# saved to a folder named images_store
def download_images(only_dicom=True,
					images_loc='add url for images',
					folder_name = 'images_store'):

	# connect to the website
	conn = urllib.urlopen(images_loc)
	url_html = conn.read()
	soup = BeautifulSoup(url_html, 'html.parser')

	# make a folder to store images
	os.mkdir(folder_name)
	os.chdir('./'+folder_name)

	# iterate through the images in the directory and save the right files
	for a_tag in soup.find_all('a', href=True): # iterate through all the links
		image_id = a_tag['href'] # this is unique for each image
		image_loc = images_loc + '/' + image_id # image is a bs4 element obj
		# TODO: current saving to disk and then reading is tedious, try a simpler way
		urllib.urlretrieve(image_loc, 'temp') # save the file as temp
		try:
			temp = dicom.read_file('temp') # to check it dicom... files have no extensions, so not sure of better way
			os.rename('temp',image_id)
			print('dicom: '+image_id)
		except:
			if only_dicom: continue # skip image if not dicom
			if not only_dicom:
				try:
					temp = plt.imread('temp') # check if file is an image
					os.rename('temp',image_id)
					print('non-dicom: ', image_id)
				except: continue

	os.chdir('./..') # go back to parent dir

################## FUNCTIONS BELOW FOR READING IN IMAGES #######################

# for dicom only images
# extracts the modality and pixel array for each dicom image and returns this data
# in the form of a dictionary: {image_id:(pixel_arr, modality)}
def extract_pixels_and_attributes(dicom_images_loc, normalize=True): # normalizes pixel intensities to between 0 and 255
	# change to directory with images
	os.chdir(dicom_images_loc)
	image_ids_list = os.listdir('./')

	image_pixel_dict = {} # to store pixel arr of each image
	image_modality_dict = {} # to store modality of each image
	image_body_part_dict = {} # to store body part that each image represents
	fail_count = 0 # count of number of images without required attributes

	# iterate through each image in the folder
	for image_id in image_ids_list:
		try:
			dicom_image_temp = dicom.read_file(image_id) # read as dicom

		except: # most likely due to non-dicom hidden files
			print(image_id)
			continue

		try:
			pixel_array_ = dicom_image_temp.pixel_array

			if normalize:
				temp_arr = double(pixel_array_)
				pixel_array_ = uint8(255*((temp_arr - temp_arr.min()) / (temp_arr.max() - temp_arr.min())))

			image_pixel_dict[image_id] = pixel_array_
			image_modality_dict[image_id] = dicom_image_temp.Modality
			image_body_part_dict[image_id] = dicom_image_temp.BodyPartExamined

		except: # if either or both attributes not found in dicom
			fail_count += 1
			print('Dicom Success, but attribute(s) not found. Image ID: ' + image_id)
			continue

		return fail_count, image_pixel_dict, image_modality_dict, image_body_part_dict



# returns a dictionary of images extracted from folder
def read_images_from_folder(images_loc):

	os.chdir(images_loc)
	image_ids_list = os.listdir('./')

	image_pixel_dict = {} # to store pixel arr of each image
	fail_count = 0 # count of number of images without required attributes

	# iterate through each image in the folder
	for image_id in image_ids_list:
		try: # check if image is dicom
			image_temp = dicom.read_file(image_id) # read as dicom
			pixel_array_ = image_temp.pixel_array
			temp_arr = double(pixel_array_)
			# normalize
			pixel_array_ = uint8(255*((temp_arr - temp_arr.min()) / (temp_arr.max() - temp_arr.min())))
			image_pixel_dict[image_id] = pixel_array_

		except: # most likely due to non-dicom
			try:
				pixel_array_ = plt.imread(image_id)
				image_pixel_dict[image_id] = pixel_array_

			except: continue

	os.chdir('./..') # go back to parent dir

	return image_pixel_dict

############### FUNCTIONS BELOW FOR EXTRACTING IMAGE DESCRIPTORS ###############
# returns a vector of global features (Haralick + average intensity)
# ellipse=True applies an elliptical mask and then extracts features
def globalFeats(pixel_array, ellipse=False):

	# convert to grayscale only if color - i.e. more than 1 channels
	# image will have a color channel dim -> hence 3
	if len(pixel_array.shape) == 3:

		print(str(3))
		# TODO: replace with something better
		# hacky way of dealing with images that have color channels first
		if pixel_array.shape[0] == 3: pixel_array = pixel_array.T

		try:
			gray= cv2.cvtColor(pixel_array,cv2.COLOR_BGR2GRAY)

		except: # some pixel arrays have more than 3 color channels, we skip them

			return None

	elif len(pixel_array.shape) == 2:

		print(str(2))

		gray = pixel_array

	print(True)

	if ellipse:

		# get image dimensions
		x,y = gray.shape[:2]

		dx, dy = int(x*0.5), int(y*0.5)

		# major and minor axis
		ex, ey = int(x*0.75)/2, int(y*0.75)/2

		ellipse_mask = np.zeros_like(gray)

		# NOTE: for some reason in cv2 the x and y are opposite!
		cv2.ellipse(ellipse_mask, center=(dy, dx), axes=(ey, ex),
					angle=0, startAngle=0, endAngle=360, color=255,thickness=-1)

		# use binary AND operator to gen required image
		gray = np.bitwise_and(gray, ellipse_mask)


	# Haralick
	har = mh.features.haralick(gray)
	har_mean = np.mean(har, 0) # col mean -- will have 13 cols

	global_feats = list(har_mean)

	# mean  + std intensity
	global_feats.append(np.mean(gray))
	global_feats.append(np.std(gray))

	return np.array(global_feats)

# returns feature array obtained using sift algorithm
def orb(pixel_array):

	# convert to grayscale only if color - i.e. more than 1 channels
	# image will have a color channel dim -> hence 3
	if len(pixel_array.shape) == 3:

		print(str(3))
		# TODO: replace with something better
		# hacky way of dealing with images that have color channels first
		if pixel_array.shape[0] == 3: pixel_array = pixel_array.T

		try:
			gray= cv2.cvtColor(pixel_array,cv2.COLOR_BGR2GRAY)

		except: # some pixel arrays have more than 3 color channels, we skip them

			return None

	elif len(pixel_array.shape) == 2:

		print(str(2))

		gray = pixel_array

	print(True)

	if ellipse:

		# get image dimensions
		x,y = gray.shape[:2]

		dx, dy = int(x*0.5), int(y*0.5)

		# major and minor axis
		ex, ey = int(x*0.75)/2, int(y*0.75)/2

		ellipse_mask = np.zeros_like(gray)

		# NOTE: for some reason in cv2 the x and y are opposite!
		cv2.ellipse(ellipse_mask, center=(dy, dx), axes=(ey, ex),
					angle=0, startAngle=0, endAngle=360, color=255,thickness=-1)

		# use binary AND operator to gen required image
		gray = np.bitwise_and(gray, ellipse_mask)
	# use ORD. similar to SIFT + SURF - and is free to use (unlike the other 2)
	orb = cv2.ORB()
	_, feats = orb.detectAndCompute(gray,None)

	return feats

# returns sift keypoints. Typically need to run through bag of words
# if ellipse=True, then only ellipse of image keypoints extracted
# ellipse=True applies an elliptical mask and then extracts features
def sift(pixel_array, ellipse=False):

	# convert to grayscale only if color - i.e. more than 1 channels
	# image will have a color channel dim -> hence 3
	if len(pixel_array.shape) == 3:

		print(str(3))
		# TODO: replace with something better
		# hacky way of dealing with images that have color channels first
		if pixel_array.shape[0] == 3: pixel_array = pixel_array.T

		try:
			gray= cv2.cvtColor(pixel_array,cv2.COLOR_BGR2GRAY)

		except: # some pixel arrays have more than 3 color channels, we skip them

			return None

	elif len(pixel_array.shape) == 2:

		print(str(2))

		gray = pixel_array

	print(True)

	if ellipse:

		# get image dimensions
		x,y = gray.shape[:2]

		dx, dy = int(x*0.5), int(y*0.5)

		# major and minor axis
		ex, ey = int(x*0.75)/2, int(y*0.75)/2

		ellipse_mask = np.zeros_like(gray)

		# NOTE: for some reason in cv2 the x and y are opposite!
		cv2.ellipse(ellipse_mask, center=(dy, dx), axes=(ey, ex),
					angle=0, startAngle=0, endAngle=360, color=255,thickness=-1)

		# use binary AND operator to gen required image
		gray = np.bitwise_and(gray, ellipse_mask)

	# use sift
	# TODO: why isnt SIFT computing for image_id = 1990_2?
	sift = cv2.xfeatures2d.SIFT_create()

	_, feats = sift.detectAndCompute(gray,None)

	return feats


# returns a relative histogram for each HSV channel
def hist(pixel_array, bins = 10):
 # with thanks to
 # http://www.pyimagesearch.com/2014/12/01/complete-guide-building-image-search-engine-python-opencv/
 # and https://github.com/danuzclaudes/image-retrieval-OpenCV/

	hsv = cv2.cvtColor(pixel_array, cv2.COLOR_BGR2HSV)

	feats = []

	# get image dimensions
	x,y = hsv.shape[:2]

	dx, dy = int(x*0.5), int(y*0.5) # get halfway point to divde into segements

	# create 4 segements (top-left, bottom-left, top-right, bottom-right)
	regions = [(0,dx,0,dy), (0,dx,dy,y), (dx,x,0,dy), (dx,x,dy,y)]

	# elliptical mask for the center of the image
	# major and minor axis
	ex, ey = int(x*0.75)/2, int(y*0.75)/2

	ellipse_mask = np.zeros([x,y], dtype = "uint8")

	# make mask into an ellipse
	cv2.ellipse(ellipse_mask, (dx, dy), (ex, ey), 0, 0, 360, 255, -1)

	# gen histogram
	hist = cv2.calcHist([hsv],[0,1,2],ellipse_mask,[bins],[0,180,0,256,0,256],True)

	# normalize
	hist = cv2.normalize(hist).flatten()

	feats.extend(hist)


	# loop through the segements and extract the histograms
	for area in regions:

		# a second mask is needed for each of the corners (everything other than the ellipse)
		corner_mask = np.zeros([x,y], dtype = "uint8")

		# draw rectangle mask on corner_mask object
		corner_mask[area[0]:area[1], area[2]:area[3]] = 255
		corner_mask = cv2.subtract(corner_mask, ellipse_mask)

		# gen histogram like before
		hist = cv2.calcHist([hsv],[0,1,2],corner_mask,bins,[0,180,0,256,0,256])
		hist = cv2.normalize(hist).flatten()

		feats.extend(hist)

		return feats

# TODO
def geometric(pixel_array):
	pass

# TODO
def mixed(pixel_array):
	pass

# for each image in dict, extract image features and add to new dict
# this function will updated as more feature extraction techniques
# are introduced
def add_image_features(image_dict, kind = 'sift', ellipse=False):

	image_feats_dict = {}

	for image in image_dict.keys():

		if image[0] == '.': continue # ignore non-image hidden files

		print(str(image))

		if kind == 'global':

			image_feats_dict[image] = globalFeats(image_dict[image], ellipse)

		if kind == 'orb':

			image_feats_dict[image] = orb(image_dict[image])

		if kind == 'sift':

			image_feats_dict[image] = sift(image_dict[image], ellipse)

		if kind == 'hist':

			image_feats_dict[image] = hist(image_dict[image])

		if kind == 'geometric':

			image_feats_dict[image] = geometric(image_dict[image])

		if kind == 'mixed':

			image_feats_dict[image] = mixed(image_dict[image])

	return image_feats_dict

######################### FUNCTIONS BELOW FOR BAG OF WORDS ######################

# return cluster centers matrix using KMeans
# k is the number of clusters
# use_subset randomly selects a subset_quantity % of datapoints to generate the
# cluster centers. Needed when we have a huge number of images
def kmeans_centers(image_feats_dict, k = 10, use_subset=False, subset_quantity=None):

		# apply k-means to find the centroids
		train_feats = np.vstack(image_feats_dict.values())

		if use_subset:
			row_count = train_feats.shape[0]
			random_idx = np.random.randint(row_count, size=int(subset_quantity*row_count))
			train_feats = train_feats[random_idx,:]

		#TODO: Kmeans calculation takes the largest amount of time, everything else is fast
		k_means = KMeans(n_clusters=k, random_state=seed).fit(train_feats)

		return k_means.cluster_centers_

# returns a Bag of Words vectors. used for keypoint features
# set query=False when a large number of images with image_ids
# this is typical when generating BoWs vectors for training images
# Use query=True for single query image matrix
def bag_of_words(image_feats_dict, cluster_centers, query=True):

	k = cluster_centers.shape[0] # number of clusters

	if not query:
		# TODO: loops are slow --> replace with numpy matrix magic
		# find closest center to each image keypoint and generate histogram
		image_hist_dict = {}

		for image_id, each_image in image_feats_dict.items():

			image_hist_dict[image_id] = np.array([0] * k)

			for keypoint in each_image:

				diff = cluster_centers - repmat(keypoint,len(cluster_centers), 1)
				euclidean_dists = np.apply_along_axis(np.linalg.norm, 1, diff, ord=2)
				image_hist_dict[image_id][np.argmin(euclidean_dists)] += 1. # add to frequency of correponding center

		return image_hist_dict

	if query:

		query_hist = np.array([0] * k)
		# convert query_feats into the histogram like above
		for keypoint in image_feats_dict:

			diff = cluster_centers - repmat(keypoint,len(cluster_centers), 1)
			euclidean_dists = np.apply_along_axis(np.linalg.norm, 1, diff)
			query_hist[np.argmin(euclidean_dists)] += 1.

		return query_hist



######################### BELOW ARE DISCTANCE/SIMILARITY FUNCTIONS #############
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
def euclidean_dist(query_arr, train_mat):

	query_mat = np.matlib.repmat(query_arr, train_mat.shape[0], 1)

	train_mat = np.array(train_mat)

	eu_dist = cdist(query_mat, train_mat, 'euclidean')[0,:]

	return eu_dist

# returns a vector of distnace to each of the training arrays from query array
# using cosine distance
def cosine_dist(query_arr, train_mat):

	query_mat = np.matlib.repmat(query_arr, train_mat.shape[0], 1)

	train_mat = np.array(train_mat)

	cosine_dist = cdist(query_mat, train_mat, 'cosine')[0,:]

	return cosine_dist


# returns a dictionary with the distnace or similarity between query and each of
# train images
def calc_dist_sim(query_feats, image_feats_dict, method='cosine'):

	image_sim_dist_dict = {}

	# cosine distance NOT similarity
	if method == 'cosine':

		# get a matrix of the distances
		distances = cosine_dist(np.array(query_feats), np.array(list(image_feats_dict.values())))
		# add to the dictionary
		image_sim_dist_dict = dict((key, val) for key,val in zip(list(image_feats_dict.keys()),distances))


	if method == 'euclidean':

		# get a matrix of the distances
		distances = euclidean_dist(np.array(query_feats),  np.array(list(image_feats_dict.values())))
		# add to the dictionary
		image_sim_dist_dict = dict((key, val) for key,val in zip(list(image_feats_dict.keys()),distances))

	if method == 'JS':

		# use jesen-shannon divergence to find distance to each image from query
		distances = jensen_shannon_div(np.array(query_feats), np.array(list(image_feats_dict.values())))
		# add to the dictionary
		image_sim_dist_dict = dict((key, val) for key,val in zip(list(image_feats_dict.keys()),distances))

	# uses hamming distance as metric to find the closest keypoints
	# NOTE: mainly used for ORB
	if method == 'force_matching':

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

	return image_sim_dist_dict


# return the overall distance or similarity between the query and image
# this applies to images that have a seperate global and local features
def combine_measures(local_dict, global_dict, weighting=0.5):

	combined_dict = {}

	for image_id in local_dict.keys():

		combined_dict[image_id] = weighting * local_dict[image_id] + (1-weighting) * global_dict[image_id]

	return combined_dict


# display retrieved images
def return_images(image_sim_dist_dict, image_dict, use_threshold=True, threshold=0.3, k=5, distance=True):

	result_image_id_list = []
	result_image_dist_list = []

	# sort based on whether sim or dist measure
	sorted_list = sorted(image_sim_dist_dict.items(), key=lambda x: x[1], reverse= not distance)

	for i in range(k):

		image_id = sorted_list[i][0]
		metric = image_sim_dist_dict[image_id]

		if use_threshold:
			if metric > threshold: # assumes > is worse (less similar)
				continue

		result_image_id_list.append(image_id)
		result_image_dist_list.append(image_sim_dist_dict[image_id])


	return zip(result_image_id_list, result_image_dist_list)

# displays the images from the image list
def display_images(image_results, image_folder):

		img_lst = list(image_results)

		for image_id, image_dist in img_lst:

			image_dist_total = str(round(image_dist, 4))

			# read the image
			try:
				image_temp = dicom.read_file(image_folder + '/' + image_id) # read as dicom
				pixel_array_ = image_temp.pixel_array
				temp_arr = double(pixel_array_)
				# normalize
				img = uint8(255*((temp_arr - temp_arr.min()) / (temp_arr.max() - temp_arr.min())))

			except:
				image_temp = plt.imread(image_folder + '/' + image_id)
				img = uint8(255*((image_temp - image_temp.min()) / (image_temp.max() - image_temp.min())))

			# display the image
			plt.figure()
			plt.imshow(img, cmap='gray')
			plt.title(image_id + ' metric: ' + image_dist_total)
			plt.show()

###################################################################################
