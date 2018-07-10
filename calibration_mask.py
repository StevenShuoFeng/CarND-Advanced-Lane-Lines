import numpy as np
import cv2
import glob

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# ------------------------------------
# return 3D coordinates for board corners of given size
def get3DCoor(sizeBoardCorners):
    r, c = sizeBoardCorners
    tar = np.zeros((r*c, 3), np.float32)
    tar[:,:2] = np.mgrid[0:r, 0:c].T.reshape(-1, 2)
    return tar

# ------------------------------------
# find corners of all images from a directory with given corner shape, then caliculate matrix for correcting distortion
def calDistortMatrix(fileDir, sizeBoardCorners, DEBUG=False):
    calFiles = glob.glob("camera_cal/calibration*.jpg")

    corners_img = []
    corners_tar = []
    
    tarCoord = get3DCoor(sizeBoardCorners)
        
    i = 0
    for fname in calFiles:            
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        
        if not ret: 
            print('failled finding board from: ', fname)
            continue
        
        corners_img.append(corners)
        corners_tar.append(tarCoord)
        
        if DEBUG:
            # draw the board and the found corners to make sure all are found correctly
            cv2.drawChessboardCorners(img, sizeBoardCorners, corners, ret)
            plt.subplot(5, 5, i+1)
            plt.imshow(img)
            i += 1
        
    if DEBUG:
        plt.show()
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(corners_tar, corners_img, gray.shape, None, None)
    
    return ret, mtx, dist

# ------------------------------------
def getGrad(img, kernelSize):    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    gradx = cv2.Sobel(img_gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize = kernelSize)
    grady = cv2.Sobel(img_gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize = kernelSize)    
    return gradx, grady


# ------------------------------------
# masking based on direction of the gradient
# img BGR color image, default threshhold: 0~pi/2
def gradientDirectionMask(gradx, grady, thresh=[0, np.pi/2]):
    angle = np.arctan2(np.absolute(grady), np.absolute(gradx))
    angle = np.absolute(angle)
    
    binary_output = np.zeros_like(angle)
    binary_output[(angle >= thresh[0]) & (angle < thresh[1])] = 1    
    return binary_output


# ------------------------------------
# masking based on magnitude of the gradient, threshhold
# img BGR color image, default threshhold:0~255
def gradientMagnitudeMask(gradx, grady, thresh=[0, 255], axis='x'):
    if axis == 'x':
        gradm = np.absolute(gradx)
    else:
        gradm = np.absolute(grady)
    gradm = gradm/gradm.max()*255
    
    binary_output = np.zeros_like(gradm)
    binary_output[(gradm >= thresh[0]) & (gradm < thresh[1])] = 1    
    return binary_output
    
    
# ------------------------------------
# masking based on the selected channel of the HLS representation of the image
# img BGR color image, default threshhold: 0~255
def colorChannelMask(img, thresh=[0, 255], channel='S'):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    
    channelMap = {'H':0, 'L':1, 'S':2}
    ch = channelMap[channel]
    singleChannel = hls[:, :, ch]
    
    binary_output = np.zeros_like(singleChannel)
    binary_output[(singleChannel >= thresh[0]) & (singleChannel <= thresh[1])] = 1
    return binary_output

def regionOfInterestMask(img):
    sizy, sizx, dummy = img.shape
    roi = np.array([[sizx*0.4, sizy*0.45], [sizx*0.05, sizy*0.95], [sizx*0.95, sizy*0.95], [sizx*0.6, sizy*0.45]]).astype(int)
    
    binary_mask = np.zeros((sizy, sizx))
    cv2.fillPoly(binary_mask, [roi], 1)
    return binary_mask

    
# ------------------------------------
# Input BGR color image
# Ouput Binary map 
# This function controls all threshholds for each individual type of masking
def findAllMasks(img):
    # Yellow channel
    rg_mask = (img[:,:,1] > 150) & (img[:,:,2] > 150)
    
    # Gradient related
    gradx, grady = getGrad(img, kernelSize = 3)    
    dir_mask = gradientDirectionMask(gradx, grady, thresh=(np.pi/6, np.pi/2))
    mag_mask = gradientMagnitudeMask(gradx, grady, thresh=(10, 200),  axis='x')
    gradient_mask = cv2.bitwise_and(dir_mask, mag_mask)
    
    # S channel
    s_mask = colorChannelMask(img, thresh=[100, 255], channel='S')
    
    # Region of Interest
    roi_mask = regionOfInterestMask(img)
    
    allMasks = {
        'rg': rg_mask, 
        'grad':gradient_mask,
        's': s_mask,
        'roi': roi_mask,
    }
    
    return allMasks


# ------------------------------------
# Stack 3 2D images/masks into RGB color images for viewing
def mergeMasks2RGB(masks):
    return np.stack(masks, axis=2)*255


# ------------------------------------
# Combine all individual masks 
def combineMasks(img):
    allMasks = findAllMasks(img)
    
    rg_mask = allMasks['rg']
    gradient_mask = allMasks['grad']
    s_mask = allMasks['s']
    roi_mask = allMasks['roi']

    # keep strong yellow channel and strong gradient
    combinedMask = (rg_mask > 0) & (gradient_mask > 0)
    
    # add the s-channel
    combinedMask = (combinedMask > 0) | (s_mask > 0)
    
    # cut off outer region and only keep region of interest in the lower middle part of the view
    combinedMask = (combinedMask > 0) & (roi_mask > 0)
    
    return combinedMask