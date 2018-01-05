# script to download the files from the server @ http://rasinsrv04.cstcis.cti.depaul.edu/all_images/
# and extract the dicom image pixels and the modality from each image

from ftplib import FTP
import dicom
import urllib
import os
from bs4 import BeautifulSoup
from PIL import Image
from numpy import uint8, double


# iterates through the images in the remote server and downloads all the image
# files or only the dicom files based on the only_dicom parameter. Images are 
# saved to a folder named images_store 
def download_images(only_dicom=True, 
                    images_loc='http://rasinsrv04.cstcis.cti.depaul.edu/all_images/all_tf/'):
    
    # connect to the website
    conn = urllib.urlopen('http://rasinsrv04.cstcis.cti.depaul.edu/all_images/all_tf/')
    url_html = conn.read()
    soup = BeautifulSoup(url_html, 'html.parser')
    
    # make a folder to store images
    os.mkdir('images_store')
    os.chdir('./images_store')
    
    # iterate through the images in the directory and save the right files
    for a_tag in soup.find_all('a', href=True): # iterate through all the links
        image_id = a_tag['href'] # this is unique for each image
        image_loc = images_loc + '/' + image_id# image is a bs4 element obj
        # TODO: current saving to disk and then reading is tedious, try a simpler way
        urllib.urlretrieve(image_loc, 'temp') # save the file as temp 
        try: 
            temp = dicom.read_file('temp') # to check it dicom... files have no extensions, so not sure of better way
            os.rename('temp',image_id) 
            print(image_id)
        except:
            if only_dicom: continue # skip image if not dicom
            else:
                try:
                    temp = Image.open('temp') # check if file is an image
                    os.rename('temp',image_id)
                    print(image_id)
                except: continue

# extracts the modality and pixel array for each dicom image and returns this data
# in the form of a dictionary: {image_id:(pixel_arr, modality)}
def extract_pixels_and_modality(dicom_images_loc = 'C:\Users\syarlag1.DPU\Desktop\CBIR-for-Radiology\images_store',
                                normalize=True # normalizes pixel intensities to between 0 and 255
                                ):
    # change to directory with images
    os.chdir(dicom_images_loc)
    image_ids_list = os.listdir('./')
    
    image_pixel_dict = {} # to store pixel arr of each image
    image_modality_dict = {} # to store modality of each image
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
        except: # if either or both attributes not found in dicom
            fail_count += 1
            print('Dicom Success, but attribute(s) not found. Image ID: ' + image_id)
            continue
        
    return fail_count, image_pixel_dict, image_modality_dict
        
    