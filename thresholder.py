import cv2
import numpy as np

# Reference from https://github.com/sumitbinnani/CarND-Advanced-Lane-Lines/blob/master/utils/thresholding.py

class Thresholder:
    def __init__(self):
        self.sobel_kernel = 15

        self.thresh_dir_min = 0.7
        self.thresh_dir_max = 1.2

        self.thresh_mag_min = 50
        self.thresh_mag_max = 255

    def dir_thresh(self, sobelx, sobely):
        abs_sobelx = np.abs(sobelx)
        abs_sobely = np.abs(sobely)
        scaled_sobel = np.arctan2(abs_sobely, abs_sobelx)
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= self.thresh_dir_min) & (scaled_sobel <= self.thresh_dir_max)] = 1

        return sxbinary

    def mag_thresh(self, sobelx, sobely):
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= self.thresh_mag_min) & (gradmag <= self.thresh_mag_max)] = 1

        return binary_output

    def color_thresh(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        yellow_min = np.array([15, 100, 120], np.uint8)
        yellow_max = np.array([80, 255, 255], np.uint8)
        yellow_mask = cv2.inRange(img, yellow_min, yellow_max)

        white_min = np.array([0, 0, 200], np.uint8)
        white_max = np.array([255, 30, 255], np.uint8)
        white_mask = cv2.inRange(img, white_min, white_max)

        binary_output = np.zeros_like(img[:, :, 0])
        binary_output[((yellow_mask != 0) | (white_mask != 0))] = 1

        filtered = img
        filtered[((yellow_mask == 0) & (white_mask == 0))] = 0

        return binary_output

    def roi_thresh(self, img):
        sizy, sizx, dummy = img.shape
        roi = np.array([[sizx*0.45, sizy*0.6], [sizx*0.05, sizy*0.95], [sizx*0.95, sizy*0.95], [sizx*0.55, sizy*0.6]]).astype(int)
        
        roi_mask = np.zeros((sizy, sizx))
        cv2.fillPoly(roi_mask, [roi], 1)
        return roi_mask

    def get_gradient(self, img):
        sobelx = cv2.Sobel(img[:, :, 2], cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(img[:, :, 2], cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)

        return sobelx, sobely


    def threshold(self, img):
        sobelx, sobely = self.get_gradient(img)

        direc = self.dir_thresh(sobelx, sobely)
        mag = self.mag_thresh(sobelx, sobely)
        color = self.color_thresh(img)
        roi = self.roi_thresh(img)

        combined = np.zeros_like(direc)
        combined[((roi == 1) & (color == 1) & ((mag == 1) | (direc == 1)))] = 1

        return combined
