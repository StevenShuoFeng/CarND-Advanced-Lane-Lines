import numpy as np
import cv2
import math

class LaneFinder:
    '''
    Initialize the window size and allocate size for window coordinates etc
    '''
    def __init__(self, num_window=10, width_window = 200):
        self.isInit = False
        self.threshold_minCountToUpdateCenter = 90
        
        self.num_win = num_window
        self.WW = int(width_window/2)
        self.HW = 0
        
        self.x_center_llane = -1*np.ones((num_window, 1), np.int32)
        self.x_center_rlane = -1*np.ones((num_window, 1), np.int32)
        
        self.y_range = np.zeros((num_window, 2), np.int32)
        

    ''' 
    Run for the first time, check the center of each left and right lane and assign it to the first (lowest) window
    Initialize window height according to image height
    '''
    def init(self, img):
        self.sizy, self.sizx = img.shape
        self.HW = math.floor(self.sizy/self.num_win) # height of window
        
        for step in range(self.num_win):
            self.y_range[step] = [self.sizy - (step+1)*self.HW, self.sizy - (step)*self.HW]
        
        self.init_proj = np.sum(img[int(self.sizy/2):, :], axis=0) # project the image into x-axis
        self.x_center_llane[0] = np.argmax(self.init_proj[:int(self.sizx/2)])
        self.x_center_rlane[0] = np.argmax(self.init_proj[int(self.sizx/2):]) + int(self.sizx/2)
        print('Initial Lane Centers: left = {}, right = {}'.format(self.x_center_llane.item(0),  self.x_center_rlane.item(0)))

        self.isInit = True

    
    '''
    '''
    def fitWindowCenter(self, img):        
        # new 3-channel mask for drawing
        self.out_img = np.dstack((img, img, img))*255
        
        # current non-zero coordinates
        self.nonzero_x = np.where(img>0)[1]
        self.nonzero_y = np.where(img>0)[0] 

        allPointsIndex_l = []
        allPointsIndex_r = []
        
        # ------------------------------------------------------------
        # Loop through each window and update window center, keep all points within each window
        for step in range(self.num_win):
            # If the center of the window is -1, use previous window center
            if self.x_center_llane[step] == -1:
                self.x_center_llane[step] = self.x_center_llane[step-1]                
            if self.x_center_rlane[step] == -1:
                self.x_center_rlane[step] = self.x_center_rlane[step-1]
            
            # Set window ranges
            center_l = self.x_center_llane.item(step)
            center_r = self.x_center_rlane.item(step)
            xwin_l = [center_l-self.WW, center_l+self.WW]
            xwin_r = [center_r-self.WW, center_r+self.WW]
            ywin = self.y_range[step]
            
            # Find points within current window
            pointsIndex_l = self.findPointsIndex(xwin_l, ywin)
            pointsIndex_r = self.findPointsIndex(xwin_r, ywin)            
            allPointsIndex_l.append(pointsIndex_l)
            allPointsIndex_r.append(pointsIndex_r)
            
            # Update window center
            if pointsIndex_l.shape[0] > self.threshold_minCountToUpdateCenter:
                self.x_center_llane[step] = np.mean(self.nonzero_x[pointsIndex_l][0])
                center_l = self.x_center_llane.item(step)
                xwin_l = [center_l-self.WW, center_l+self.WW]
            
            if pointsIndex_r.shape[0] > self.threshold_minCountToUpdateCenter:
                self.x_center_rlane[step] = np.mean(self.nonzero_x[pointsIndex_r][0])
                center_r = self.x_center_rlane.item(step)
                xwin_r = [center_r-self.WW, center_r+self.WW]
                
            # Draw the window boundary
            cv2.rectangle(self.out_img, (xwin_l[0], ywin[0]), (xwin_l[1], ywin[1]),\
                          color=(255,0,0), thickness=3)
            cv2.rectangle(self.out_img, (xwin_r[0], ywin[0]), (xwin_r[1], ywin[1]),\
                          color=(0,0,255), thickness=3)
        
        # ------------------------------------------------------------
        # Fit a second order polynomial to each side of the lane, get fitted line center
        allPointsIndex_l = np.squeeze(np.concatenate(allPointsIndex_l))
        allPointsIndex_r = np.squeeze(np.concatenate(allPointsIndex_r))
                
        fitcoeff_l = np.polyfit(self.nonzero_y[allPointsIndex_l], self.nonzero_x[allPointsIndex_l], 2)
        fitcoeff_r = np.polyfit(self.nonzero_y[allPointsIndex_r], self.nonzero_x[allPointsIndex_r], 2)        
        
        # ------------------------------------------------------------
        # Draw the fitted lines
        fit_yl = np.array(range(self.sizy), np.int32)
        fit_xl = fitcoeff_l[0]*fit_yl**2 + fitcoeff_l[1]*fit_yl**1 + fitcoeff_l[2]
        points_l = np.stack((fit_xl, fit_yl), axis=1)
        
        fit_yr = np.array(range(self.sizy), np.int32)
        fit_xr = fitcoeff_r[0]*fit_yr**2 + fitcoeff_r[1]*fit_yr**1 + fitcoeff_r[2]
        points_r = np.stack((fit_xr, fit_yr), axis=1)
        
        cv2.polylines(self.out_img, np.int32([points_l]),\
                      isClosed=False, color=(255,0,0), thickness=5)
        cv2.polylines(self.out_img, np.int32([points_r]),\
                      isClosed=False, color=(0,0,255), thickness=5)
        
        # ------------------------------------------------------------
        # Compute lane curvature
        print(fitcoeff_l, fitcoeff_r)
        curv_l = self.computeCurvature(fitcoeff_l, self.sizy)
        curv_r = self.computeCurvature(fitcoeff_r, self.sizy)        
        print('Left curvature {}, Right curvature {}'.format(curv_l, curv_r))
                         
    
    # Input coef is in unit of pixel
    def computeCurvature(self, coef, y_location):
        
        x_meter_per_pixel = 3.7/477 # meter/pixel US lane width 3.7 meter, 394~871 in the image
        y_meter_per_pixel = 30/400 # 40 feet, 12.19meters between two dash line, 400 pixel in image
        
        A = x_meter_per_pixel / (y_meter_per_pixel**2) * coef[0]
        B = x_meter_per_pixel / y_meter_per_pixel * coef[1]
        Y = y_location*y_meter_per_pixel
        
        curvature = (1+(2*A*Y+B)**2)**1.5 / np.absolute(2*A)
        return curvature
    
    
    def findPointsIndex(self, xrange, yrange):
        pntIndex = np.where((self.nonzero_x > xrange[0]) & (self.nonzero_x < xrange[1]) & (self.nonzero_y > yrange[0]) & (self.nonzero_y < yrange[1]))
        return np.squeeze(np.array(pntIndex))
        
        
    # INPUT: 
    # img should be 2D grayscale, bird-view image of lanes from combined mask
    # num_window and width_window define the sliding window
    def run(self, mask):
        # Initialize 
        if not self.isInit:
            self.init(mask)
            
        # Find window centers
        self.fitWindowCenter(mask)
        