import numpy as np
import cv2
import math

# The LaneFinder class is responsible for :
# 1) detecting lane edges after masking using a sliding window
# 2) estimate the line coefficients in term of a 2nd order polynomial
# 3) compute curvature and car locations

class LaneFinder:
    '''
    Initialize the window size and allocate size for window coordinates etc
    '''
    def __init__(self,  M, M_inv, num_window=10, width_window = 150):
        self.isInit = False
        self.threshold_minCountToUpdateCenter = 100
        
        self.num_win = num_window
        self.WW = int(width_window/2)
        self.HW = 0
        
        self.x_center_llane = -1*np.ones((num_window), np.int32)
        self.x_center_rlane = -1*np.ones((num_window), np.int32)
        
        self.y_range = np.zeros((num_window, 2), np.int32)
        
        self.M = M
        self.M_inv = M_inv
        
        self.laneWidth = 490 # num of pixels between lane lines in bird-view

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
        print('Initial Lane Centers: left = {}, right = {}'.format(self.x_center_llane[0],  self.x_center_rlane[0]))

        self.isInit = True
    
    '''
    '''
    def fitWindowCenter(self, mask):        
        # new 3-channel mask for drawing
        self.mask_laneAndWindow = np.dstack((mask, mask, mask))*255
        
        # current non-zero coordinates
        self.nonzero_x = np.where(mask>0)[1]
        self.nonzero_y = np.where(mask>0)[0] 

        allPointsIndex_l = []
        allPointsIndex_r = []
                
        self.init_proj = np.sum(mask[int(self.sizy*0.6):, :], axis=0) # project the image into x-axis
        self.x_center_llane[0] = np.argmax(self.init_proj[:int(self.sizx/2)])
        self.x_center_rlane[0] = np.argmax(self.init_proj[int(self.sizx/2):]) + int(self.sizx/2)
        
        # ------------------------------------------------------------
        # Loop through each window (from bottom to up) and update window center, keep all points within each window
        for step in range(int(1*self.num_win)):
            # If the center of the window is -1, use previous window center
            if step > 0 and self.x_center_llane[step] == -1:
                self.x_center_llane[step] = self.x_center_llane[step-1]                
            if step > 0 and self.x_center_rlane[step] == -1:
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
                        
            if pointsIndex_l.size > 0:
                allPointsIndex_l.append(pointsIndex_l)
            if pointsIndex_r.size > 0:
                allPointsIndex_r.append(pointsIndex_r)
            
            # Update window center if there're enough points within, otherwise use same location as the lower window
            if pointsIndex_l.size > self.threshold_minCountToUpdateCenter:
                self.x_center_llane[step] = np.mean(self.nonzero_x[pointsIndex_l])
                # print(self.nonzero_x[pointsIndex_l][0].shape)
            elif step > 0:
                self.x_center_rlane[step] = self.x_center_rlane[step-1]           
        
            if pointsIndex_r.size > self.threshold_minCountToUpdateCenter:
                self.x_center_rlane[step] = np.mean(self.nonzero_x[pointsIndex_r])
            elif step > 0:
                self.x_center_rlane[step] = self.x_center_rlane[step-1]
                
            # Draw the window boundary
            center_l = self.x_center_llane.item(step)
            xwin_l = [center_l-self.WW, center_l+self.WW]
            
            center_r = self.x_center_rlane.item(step)
            xwin_r = [center_r-self.WW, center_r+self.WW]
            
            cv2.rectangle(self.mask_laneAndWindow, (xwin_l[0], ywin[0]), (xwin_l[1], ywin[1]), color=(255,0,0), thickness=5)
            cv2.rectangle(self.mask_laneAndWindow, (xwin_r[0], ywin[0]), (xwin_r[1], ywin[1]), color=(0,0,255), thickness=5)
        
        # ------------------------------------------------------------
        # Fit a second order polynomial to each side of the lane, get fitted line center

        allPointsIndex_l = np.squeeze(np.concatenate(allPointsIndex_l))
        allPointsIndex_r = np.squeeze(np.concatenate(allPointsIndex_r))
        
        self.fitFromPointsInTwoLines(allPointsIndex_l, allPointsIndex_r)
        
        # ------------------------------------------------------------
        # Draw the fitted lines
        y_draw = np.array(range(int(0.7*self.sizy)), np.int32)

        fit_xl = self.getXcoordFromYCoord(y_draw, 'l')
        fit_xr = self.getXcoordFromYCoord(y_draw, 'r')

        points_l = np.stack((fit_xl, y_draw), axis=1).astype(np.int32)
        points_r = np.stack((fit_xr, y_draw), axis=1).astype(np.int32)
        
        cv2.polylines(self.mask_laneAndWindow, [points_l], isClosed=False, color=(255,0,0), thickness=4)
        cv2.polylines(self.mask_laneAndWindow, [points_r], isClosed=False, color=(0,0,255), thickness=4)
        
        
    # Fit the curve coefficients together for one set of coefficients, this enforce the fixed width fact
    # Shift the points from the left line to the right by half of the lane width, and shift points from right line to the left
    def fitFromPointsInTwoLines(self, indexOfNonZero_l, indexOfNonZero_r):
        x_l = self.nonzero_x[indexOfNonZero_l] + int(self.laneWidth/2)
        y_l = self.nonzero_y[indexOfNonZero_l]

        x_r = self.nonzero_x[indexOfNonZero_r] - int(self.laneWidth/2)
        y_r = self.nonzero_y[indexOfNonZero_r]

        x_coord = np.concatenate((x_l, x_r), axis=0)
        y_coord = np.concatenate((y_l, y_r), axis=0)
                
        self.fitcoeff = np.polyfit(y_coord, x_coord, 2)


    def getXcoordFromYCoord(self, y, side='l'):
        fit_x = self.fitcoeff[0]*y**2 + self.fitcoeff[1]*y**1 + self.fitcoeff[2]

        if side == 'l':
            fit_x = fit_x - (self.laneWidth)/2
        else:
            fit_x = fit_x + (self.laneWidth)/2

        return fit_x

        
    def computeCurvature(self, ifPrintInfo=False):
        # Compute curvature at the bottom of the view
        self.curv = self.computeSingleCurvature(self.fitcoeff, self.sizy)

        if ifPrintInfo:
            print('Curvature radius {:.1f} meter'.format(self.curv))
    
    # Input coef is in unit of pixel
    def computeSingleCurvature(self, coef, y_location):
        
        x_meter_per_pixel = 3.7/477 # meter/pixel US lane width 3.7 meter, 394~871 in the image
        y_meter_per_pixel = 20/720 # 40 feet, 12.19meters between two dash line, 400 pixel in image
        
        # Map unit of coefficients and locaiton from pixel to meter
        A = x_meter_per_pixel / (y_meter_per_pixel**2) * coef[0]
        B = x_meter_per_pixel / y_meter_per_pixel * coef[1]
        Y = y_location*y_meter_per_pixel
        
        curvature = (1+(2*A*Y+B)**2)**1.5 / np.absolute(2*A)
        return curvature 
    
    
    def findPointsIndex(self, xrange, yrange):
        pntIndex = np.where((self.nonzero_x > xrange[0]) & (self.nonzero_x < xrange[1]) & (self.nonzero_y > yrange[0]) & (self.nonzero_y < yrange[1]))
        return np.array(pntIndex).flatten()
    
    # Mark the starting point (bottom) of each lane line, mask the points and map back to original view
    # Then measure the lane center location relative to the center of image as shift in position
    def findCarPosition(self, ifPrintInfo=False):
        m = np.zeros((self.sizy, self.sizx), np.int32)
        y = self.sizy - 40
        x_l = self.getXcoordFromYCoord(y, 'l').astype(np.int32)
        x_r = self.getXcoordFromYCoord(y, 'r').astype(np.int32)

        # Set the lane line start points in the bird view
        m[y, x_l] = 255
        m[y, x_r] = 255
        
        # map this mask to orignal view
        m_orig = cv2.warpPerspective(np.float64(m), self.M_inv, (self.sizx, self.sizy), flags=cv2.INTER_LINEAR)

        # find line start points in original view
        nonzero_x = np.where(m_orig>0)[1]
        center_l = np.median(nonzero_x[nonzero_x < self.sizx/2])
        center_r = np.median(nonzero_x[nonzero_x > self.sizx/2])
        
        # calculate car location
        ratio = 370/(center_r-center_l) # meter/pixel (3.7 meter lane width)
        c_shift_pixel = (center_l+center_r)/2 - self.sizx/2
        self.c_shift_cm = c_shift_pixel*ratio
        
        if ifPrintInfo:
            print('line start pixel index : l {}, r {}'.format(center_l, center_r))
            print('lane center shifts:  {:.1f} pixels, or {:.3f} meters'.format(c_shift_pixel, self.c_shift_cm))
        
    # origImg is RGB orignal image
    def drawLaneBoundaryInOrigView(self, origImg):
        raw =  np.int32(origImg)
        
        m = np.zeros((self.sizy, self.sizx, 3), np.float64)
        
        # Draw the fitted lines and regions in between
        fit_y = np.array(range(int(self.sizy*0.3), int(self.sizy*0.95)), np.int32)        
        fit_xl = self.getXcoordFromYCoord(fit_y, 'l').astype(np.int32)
        fit_xr = self.getXcoordFromYCoord(fit_y, 'r').astype(np.int32)

        points_l = np.stack((fit_xl, fit_y), axis=1)
        points_r = np.stack((fit_xr, fit_y), axis=1)

        cv2.polylines(m, np.int32([points_l]), isClosed=False, color=(255, 0, 0), thickness= 25)
        cv2.polylines(m, np.int32([points_r]), isClosed=False, color=(0, 0, 255), thickness= 25)


        edges = np.concatenate((points_l[::-1, :], points_r), axis=0)

        cv2.fillPoly(m, np.int32([edges]), color=(149, 249, 166))
        
        # Wrap it into orignal view
        fills_origView = cv2.warpPerspective(m, self.M_inv, (self.sizx, self.sizy), flags=cv2.INTER_LINEAR).astype(np.int32)
    
        # Add two images together
        outweighted = cv2.addWeighted(raw, 1,  fills_origView, 0.5, 0).astype(np.int32)
        
        self.final = outweighted
        
    # Display info in output image
    def write_Info(self):        
        if self.c_shift_cm > 0:
            side = 'right'
        else:
            side = 'left'
        textCurv = 'Radius (curvature): {:.2f} m'.format(self.curv)        
        textLoca = 'Location: {:.3} cm {} of center'.format(abs(self.c_shift_cm), side)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(self.final, text=textCurv, org=(40,70), fontFace=font, fontScale=1.2, color=(255,255,0), thickness=2)
        cv2.putText(self.final, text=textLoca, org=(40,120), fontFace=font, fontScale=1.2, color=(255,255,0), thickness=2)
        self.final = np.int32(self.final)
        
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
        self.drawLaneBoundaryInOrigView(origImg)
        
        # Print information on final image
        self.write_Info()
        
        # return self.mask_laneAndWindow
        return self.final