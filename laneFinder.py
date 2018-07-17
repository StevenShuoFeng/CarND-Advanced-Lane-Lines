import numpy as np
import cv2
import math
from calibration_mask import *

class LaneFinder:
    '''
    Initialize the window size and allocate size for window coordinates etc
    '''
    def __init__(self,  M, M_inv, num_window=10, width_window = 200):
        self.isInit = False
        self.threshold_minCountToUpdateCenter = 90
        
        self.num_win = num_window
        self.WW = int(width_window/2)
        self.HW = 0
        
        self.x_center_llane = -1*np.ones((num_window, 1), np.int32)
        self.x_center_rlane = -1*np.ones((num_window, 1), np.int32)
        
        self.y_range = np.zeros((num_window, 2), np.int32)
        
        self.M = M
        self.M_inv = M_inv

    ''' 
    Run for the first time, check the center of each left and right lane and assign it to the first (lowest) window
    Initialize window height according to image height
    '''
    def init(self, img):
        self.sizy, self.sizx = img.shape
        self.HW = math.floor(self.sizy/self.num_win) # height of window
        
        for step in range(self.num_win):
            self.y_range[step] = [self.sizy - (step+1)*self.HW, self.sizy - (step)*self.HW - 1]
        
        self.init_proj = np.sum(img[int(self.sizy/2):, :], axis=0) # project the image into x-axis
        self.x_center_llane[0] = np.argmax(self.init_proj[:int(self.sizx/2)])
        self.x_center_rlane[0] = np.argmax(self.init_proj[int(self.sizx/2):]) + int(self.sizx/2)
        print('Initial Lane Centers: left = {}, right = {}'.format(self.x_center_llane.item(0),  self.x_center_rlane.item(0)))

        self.isInit = True
    
    '''
    '''
    def fitWindowCenter(self, img):        
        # new 3-channel mask for drawing
        self.img_laneAndWindow = np.dstack((img, img, img))*255
        
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
            cv2.rectangle(self.img_laneAndWindow, (xwin_l[0], ywin[0]), (xwin_l[1], ywin[1]), color=(255,0,0), thickness=5)
            cv2.rectangle(self.img_laneAndWindow, (xwin_r[0], ywin[0]), (xwin_r[1], ywin[1]), color=(0,0,255), thickness=5)
        
        # ------------------------------------------------------------
        # Fit a second order polynomial to each side of the lane, get fitted line center
        allPointsIndex_l = np.squeeze(np.concatenate(allPointsIndex_l))
        allPointsIndex_r = np.squeeze(np.concatenate(allPointsIndex_r))
                
        fitcoeff_l = np.polyfit(self.nonzero_y[allPointsIndex_l], self.nonzero_x[allPointsIndex_l], 2)
        fitcoeff_r = np.polyfit(self.nonzero_y[allPointsIndex_r], self.nonzero_x[allPointsIndex_r], 2)        
        
        self.fitcoeff_l = fitcoeff_l
        self.fitcoeff_r = fitcoeff_r
        
        # ------------------------------------------------------------
        # Draw the fitted lines
        fit_y = np.array(range(self.sizy), np.int32)
        
        fit_xl = fitcoeff_l[0]*fit_y**2 + fitcoeff_l[1]*fit_y**1 + fitcoeff_l[2]
        points_l = np.stack((fit_xl, fit_y), axis=1)
        
        fit_xr = fitcoeff_r[0]*fit_y**2 + fitcoeff_r[1]*fit_y**1 + fitcoeff_r[2]
        points_r = np.stack((fit_xr, fit_y), axis=1)
        
        cv2.polylines(self.img_laneAndWindow, np.int32([points_l]), isClosed=False, color=(255,0,0), thickness=5)
        cv2.polylines(self.img_laneAndWindow, np.int32([points_r]), isClosed=False, color=(0,0,255), thickness=5)
        
        # ------------------------------------------------------------
        # Compute lane curvature
        self.computeCurvature()
        
        
    def computeCurvature(self, ifPrintInfo=False):
        # Compute curvature at the bottom of the view
        curv_l = self.computeSingleCurvature(self.fitcoeff_l, self.sizy)
        curv_r = self.computeSingleCurvature(self.fitcoeff_r, self.sizy)
        if ifPrintInfo:
            print('Left curvature radius {:.1f} meter, Right curvature radius {:.1f} meter'.format(curv_l, curv_r))

        self.curv = (curv_l + curv_r)/2
    
    # Input coef is in unit of pixel
    def computeSingleCurvature(self, coef, y_location):
        
        x_meter_per_pixel = 3.7/477 # meter/pixel US lane width 3.7 meter, 394~871 in the image
        y_meter_per_pixel = 30/400 # 40 feet, 12.19meters between two dash line, 400 pixel in image
        
        # Map unit of coefficients and locaiton from pixel to meter
        A = x_meter_per_pixel / (y_meter_per_pixel**2) * coef[0]
        B = x_meter_per_pixel / y_meter_per_pixel * coef[1]
        Y = y_location*y_meter_per_pixel
        
        curvature = (1+(2*A*Y+B)**2)**1.5 / np.absolute(2*A)
        return curvature 
    
    
    def findPointsIndex(self, xrange, yrange):
        pntIndex = np.where((self.nonzero_x > xrange[0]) & (self.nonzero_x < xrange[1]) & (self.nonzero_y > yrange[0]) & (self.nonzero_y < yrange[1]))
        return np.squeeze(np.array(pntIndex))
    
    
    # 
    def findCarPosition(self, ifPrintInfo=False):
        m = np.zeros((self.sizy, self.sizx), np.int32)
        y = self.sizy - 10
        x_l = self.fitcoeff_l[0]*y**2 + self.fitcoeff_l[1]*y**1 + self.fitcoeff_l[2]
        x_l = np.int32(x_l)
        x_r = self.fitcoeff_r[0]*y**2 + self.fitcoeff_r[1]*y**1 + self.fitcoeff_r[2]
        x_r =  np.int32(x_r)
        
        # Set the lane line start points in the bird view
        m[y, x_l] = 1
        m[y, x_r] = 1
        
        # map this mask to orignal view
        m_orig = cv2.warpPerspective(np.float64(m), self.M_inv, (self.sizx, self.sizy), flags=cv2.INTER_LINEAR)

        # find line start points in original view
        nonzero_x = np.where(m_orig>0)[1]
        center_l = np.median(nonzero_x[nonzero_x < self.sizx/2])
        center_r = np.median(nonzero_x[nonzero_x > self.sizx/2])
        
        # calculate car location
        ratio = 3.7/(center_r-center_l) # meter/pixel (3.7 meter lane width)
        c_shift_pixel = (center_l+center_r)/2 - self.sizx/2
        self.c_shift_meter = c_shift_pixel*ratio
        
        if ifPrintInfo:
            print('line start pixel index : l {}, r {}'.format(center_l, center_r))
            print('lane center shifts:  {:.1f} pixels, or {:.3f} meters'.format(c_shift_pixel, self.c_shift_meter))
        
    # origImg is RGB orignal image
    def drawLaneBoundaryInOrigView(self, origImg):
        m = np.zeros((self.sizy, self.sizx, 3), np.float64)
        
        # Draw the fitted lines and regions in between
        fit_y = np.array(range(self.sizy), np.int32)        
        fit_xl = self.fitcoeff_l[0]*fit_y**2 + self.fitcoeff_l[1]*fit_y**1 + self.fitcoeff_l[2]
        points_l = np.stack((fit_xl, fit_y), axis=1)        
        fit_xr = self.fitcoeff_r[0]*fit_y**2 + self.fitcoeff_r[1]*fit_y**1 + self.fitcoeff_r[2]
        points_r = np.stack((fit_xr, fit_y), axis=1)
        
        cv2.polylines(m, np.int32([points_l]), isClosed=False, color=(255,0,0), thickness= 25)
        cv2.polylines(m, np.int32([points_r]), isClosed=False, color=(0,0,255), thickness= 25)
        
        corners = np.zeros((4,2))
        corners[0:2, :] = points_l[[1, -1], :]
        corners[0:2, 0] += 5
        corners[2:4, :] = points_r[[-1, 1], :]
        corners[2:4, 0] -= 5
        cv2.fillPoly(m, np.int32([corners]), color=(0, 230, 0))
        
        # Wrap it into orignal view
        fills_origView = cv2.warpPerspective(m, self.M_inv, (self.sizx, self.sizy), flags=cv2.INTER_LINEAR)
    
        # Add two images together
        raw = np.float64(cv2.cvtColor(origImg, cv2.COLOR_BGR2RGB)/255)
        outweighted = cv2.addWeighted(raw, 1,  fills_origView, 0.5, 0)
        
        return outweighted
        
    # Display info in output image
    def write_Info(self):
        font = cv2.FONT_HERSHEY_DUPLEX
        
        textCurv = 'Radius (curvature): {:.2f} m'.format(self.curv)
        
        if self.c_shift_meter > 0:
            side = 'right'
        else:
            side = 'left'
        textLoca = 'Location: {:.3} m {} of center'.format(abs(self.c_shift_meter), side)

        # cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) → None¶

        cv2.putText(self.final, text=textCurv, org=(40,70), fontFace=font, fontScale=1.5, color=(255,255,0), thickness=2)
        cv2.putText(self.final, text=textLoca, org=(40,120), fontFace=font, fontScale=1.5, color=(255,255,0), thickness=2)
        
    # INPUT: 
    # img should be 2D grayscale, bird-view image of lanes from combined mask
    # num_window and width_window define the sliding window
    def run(self, origImg, mask):
        # Initialize 
        if not self.isInit:
            self.init(mask)
            
        # Find window centers
        self.fitWindowCenter(mask)
        
        # Compute lane curvature
        self.computeCurvature()
        
        # Find the location of  the car within lane
        self.findCarPosition()
        
        # Draw lane boundary and regions in original view
        self.final = self.drawLaneBoundaryInOrigView(origImg)
        
        # Print information on final image
        self.write_Info()
        return self.final