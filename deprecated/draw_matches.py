# -*- coding: utf-8 -*-
"""
Created on Thu May 04 11:57:45 2017

@author: syarlag1 (drawMatches function mainly obtained from stackoverflow -- see below)
"""
import numpy as np
import cv2

def drawMatches(img1, kp1, img2, kp2, matches, kind='orb', max_display=10):
    """
    with thanks to http://stackoverflow.com/questions/20259025/
                            module-object-has-no-attribute-drawmatches-opencv-python    
    
    slightly modified to include distances on lines    
    
    cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    display = 0
    for mat_ in matches:
        
        if display == max_display: break
        
        if kind=='sift': mat = mat_[0]
        if kind=='orb': mat = mat_

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        dist_str = str(round(mat.distance,2))
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
        cv2.putText(out,dist_str, (int(x1),int(y1)), cv2.FONT_ITALIC, 0.4, (0,255,0), thickness=1)

        display += 1
    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')
    #plt.imshow(out)
        
    # Also return the image if you'd like a copy
    return out

import numpy as np

def make_ellipse(img):
    # get image dimensions
    x,y = img.shape[:2]

    # center
    dx, dy = int(x*0.5), int(y*0.5)

    # major and minor axis
    ex, ey = int(x*0.75)/2, int(y*0.75)/2

    # make into a mask
    ellipse_mask = np.zeros_like(img)

    cv2.ellipse(ellipse_mask, (dy, dx), (ey, ex), 0, 0, 360, 255, -1)

    # use binary AND operator to gen required image
    img = np.bitwise_and(img, ellipse_mask)
    
    return img


if __name__ == '__main__':
    
    import os; os.chdir('C:/Users/syarlag1.DPU/Desktop/CBIR-for-Radiology/images_sample')

    
    import matplotlib.pyplot as plt
    
    img_test = np.array([[255]*100]*25)
    ellipse = make_ellipse(img_test)
    
    plt.imshow(img_test,'gray')
    plt.imshow(ellipse,'gray')



    img1 = cv2.imread('171_5',0)
    img2 = cv2.imread('170_10',0)

    img1 = make_ellipse(cv2.imread('171_5',0)) # queryImage
    img2 = make_ellipse(cv2.imread('170_10',0))# trainImage

    plt.imshow(img2,cmap='gray')
    
    
    # Initiate ORB detector
    sift = cv2.SIFT()
    # find the keypoints and descriptors with ORB
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher()

    # Match descriptors.
    matches = bf.knnMatch(des1,des2, k=2)
    
    # Sort them in the order of their distance --> only for orb
    #matches = sorted(matches, key = lambda x:x.distance)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    out = drawMatches(img1, kp1, img2, kp2, good, 'sift',100)
    #cv2.imwrite('out',out)
    plt.figure()
    plt.imshow(out,cmap='gray')
    #plt.savefig('out')
# Draw first 10 matches.
#cv2.imshow('with lines', img3)#,plt.show()
