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
def calDistortMatrix(fileDir, sizeBoardCorners, ifPlot=False):
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
        
        if ifPlot:
            # draw the board and the found corners to make sure all are found correctly
            cv2.drawChessboardCorners(img, sizeBoardCorners, corners, ret)
            plt.subplot(5, 5, i+1)
            plt.imshow(img)
            i += 1
        
    if ifPlot:
        plt.show()
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(corners_tar, corners_img, gray.shape, None, None)
    
    return ret, mtx, dist


# ------------------------------------
# Perspective transform of a rectangular
# Selected points are hard-coded
def computePerspective(ifPlot=False):
    
    # make a copy of image for drawing
    testFn = 'test_images/straight_lines1.jpg'
    img_orig = cv2.imread(testFn)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    
    # Four corner points of the rectangular used to map perspective
    points_orig = np.array([[312,650], [1007, 650], [720, 470], [568, 470]])
    points_targ = np.array([[360,650], [850, 650], [850, 100], [360, 100]])

    # Compute the forward and reverse persepective transform matrices
    M         = cv2.getPerspectiveTransform(np.float32(points_orig), np.float32(points_targ))
    M_inv = cv2.getPerspectiveTransform(np.float32(points_targ), np.float32(points_orig))
    
    # Apply the forward transform to get bird view image
    img_bird = cv2.warpPerspective(img_orig, M, (img_orig.shape[1], img_orig.shape[0]) , flags=cv2.INTER_LINEAR)

    if ifPlot:
        # Draw boundaries of the rectangular in both views
        cv2.polylines(img_orig, np.int32([points_orig]), isClosed=True, color=[255, 0 ,0], thickness=3)
        cv2.polylines(img_bird, np.int32([points_targ]), isClosed=True, color=[0, 0, 255], thickness=3)
        
        # Display
        f, axhandles = plt.subplots(1, 2, figsize=(20,10))

        axhandles[0].imshow(img_orig)
        axhandles[0].set_title('Original Image with 4 Source points')

        axhandles[1].imshow(img_bird)
        axhandles[1].set_title('Bird-view Image with 4 Target points')
        
        plt.show()
        
    return M, M_inv

