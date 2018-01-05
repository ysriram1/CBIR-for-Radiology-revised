from __future__ import print_function

import cv2
import numpy as np
import dicom
import os
import mahotas as mh

# returns the pixel array of dicom image
def read_dicom_image(image_path):

	return dicom.read_file(image_path)


# returns a dictionary of images extracted from folder
def read_images_from_folder(location):

	image_dict = {}
	image_path_list = os.listdir(location)

	for image_path in image_path_list:

		if image_path[-3:] == 'dcm': # if dicom image

			image_dict[image_path]=read_dicom_image(location+image_path).pixel_array

		else:

			image_dict[image_path]=cv2.imread(location+image_path)

	return image_dict


# for each image in dict, exi
class image_descriptors():

	# returns a vector of global features (Haralick + average intensity)
	@staticmethod
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
	@staticmethod
	def orb(pixel_array):

		# convert to grayscale
		gray= cv2.cvtColor(pixel_array,cv2.COLOR_BGR2GRAY)

		# use ORD. similar to SIFT + SURF - and is free to use (unlike the other 2)
		orb = cv2.ORB()

		_, feats = orb.detectAndCompute(gray,None)

		return feats

	# if ellipse=True, then only ellipse of image keypoints extracted
	@staticmethod
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
		sift = cv2.SIFT()

		_, feats = sift.detectAndCompute(gray,None)

		return feats


	# returns a relative histogram for each HSV channel
	@staticmethod
	def hist(pixel_array, bins = 10):
     # with thanks to http://www.pyimagesearch.com/2014/12/01/complete-guide-building-image-search-engine-python-opencv/
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
	@staticmethod
	def geometric(pixel_array):
		pass
	# TODO
	@staticmethod
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

			image_feats_dict[image] = image_descriptors.globalFeats(image_dict[image], ellipse)

		if kind == 'orb':

			image_feats_dict[image] = image_descriptors.orb(image_dict[image])

		if kind == 'sift':

			image_feats_dict[image] = image_descriptors.sift(image_dict[image], ellipse)

		if kind == 'hist':

			image_feats_dict[image] = image_descriptors.hist(image_dict[image])

		if kind == 'geometric':

			image_feats_dict[image] = image_descriptors.geometric(image_dict[image])

		if kind == 'mixed':

			image_feats_dict[image] = image_descriptors.mixed(image_dict[image])

	return image_feats_dict
