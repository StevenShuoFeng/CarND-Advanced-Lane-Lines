import numpy as np
import cv2

from calibration_mask import *
from laneFinder import LaneFinder

import matplotlib.pyplot as plt

class PipeLine:
    def __init__(self):
        self.isInitialized = False
            
    # Run len calibration, estimate view transform matrix
    def initialize(self):
        
        # Run lens calibration with all calibration images, store distortion matrix
        print('Pipeline Initialization: Calculating lens distortion ...')
        sizeBoardCorners = (9, 6)
        ret, mtx, dist = calDistortMatrix('camera_cal', sizeBoardCorners, ifPlot=False)
        self.distortionMtx = mtx
        self.distortionDist = dist
    
        # Calculate view transform matrices (forward and backward)
        print('Pipeline Initialization: Calculating bird view transform matrix ...')
        self.M, self.M_inv = computePerspective()
        
        # Initialize a LaneFinder object
        print('Pipeline Initialization: Initializing LaneFinder instance ...')
        self.laneFinderObj = LaneFinder(self.M, self.M_inv)
        
        self.isInitialized = True
         
        
    def run_Pipeline(self, img):
        if not self.isInitialized:
            self.initialize()
            
        # Fix lens distortion
        img_undist = cv2.undistort(img, self.distortionMtx, self.distortionDist, None, self.distortionMtx)
    
        # Compute the  final combined mask
        mask = combineMasks(img_undist)
        tmp = mask
        mask_bird = cv2.warpPerspective(np.float64(tmp), self.M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR)
        
        # Run the lane finder process with includes:
        # 1. Find lane lines with sliding window;
        # 2. Fit the lane line coefficients;
        # 3. Compute curvature;
        # 4. Mark image with lane lines/regions and display info as text;
        res = self.laneFinderObj.run(img, mask_bird)
        
        return res
        