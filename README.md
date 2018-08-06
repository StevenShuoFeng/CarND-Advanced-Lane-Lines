## Advanced Lane Detection

### Shuo Feng, Aug.05 2018

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of t he lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[img1]: ./output_images/writeup/chessboard.png 
[img2]: ./output_images/writeup/distortion.png 
[img3]: ./output_images/writeup/masks.png 
[img4]: ./output_images/writeup/finalMask.png 
[img5]: ./output_images/writeup/projectAndWindow.png 
[img6]: ./output_images/writeup/perspective.png 
[img7]: ./output_images/writeup/laneInOrigView.png 
[img8]: ./output_images/writeup/final.png 
[video]: ./videos/project_video_output.mp4 

---


### Here's the code structure. All source codes are under the root directory.
|-- P4_playground.ipynb

|-- calibration.py

|-- laneFinder.py

|-- pipeline.py

|-- thresholder.py

 [P4_playground.ipynb](https://github.com/StevenShuoFeng/CarND-Advanced-Lane-Lines/blob/master/P4_playground.ipynb) is the playground of the whole pipeline and more.

 [calibration.py](https://github.com/StevenShuoFeng/CarND-Advanced-Lane-Lines/blob/master/calibration) contains functions for calculate lens distortion and correction matrix, and also the function to perform perspective transforms.

[thresholder.py](https://github.com/StevenShuoFeng/CarND-Advanced-Lane-Lines/blob/master/thresholder.py) contains the Thresholder class with a few individual masking methods.

[laneFinder.py](https://github.com/StevenShuoFeng/CarND-Advanced-Lane-Lines/blob/master/laneFinder.py) contains the LaneFinder class and is responsible for 
1) detecting lane edges after masking using a sliding window
2) estimate the line coefficients in term of a 2nd order polynomial
3) compute curvature and car locations

[pipeline.py](https://github.com/StevenShuoFeng/CarND-Advanced-Lane-Lines/blob/master/pipeline.py) contains the PipeLine class, it initialize once to calculate the lens distortion, then perform the masking and lane detection modules. It holds a Thresholder object and a LaneFinder object.

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for computing the distortion matrix is performed by function 'calDistortMatrix' in 'calibration.py'
It scans through all the chessboard images provided and perform chessboard corner detection for each. All those images with 9x6 corners are then used for computing the distortion matrix. The distortion is calculated using 'cv2.calibrateCamera' function which takes points coordinates in the orignal images and the desired images.

The points in the original images are show here 
![alt text][img1]

And the desired coordinates are just grid conners on a 2D 9x6 rectangular.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

As the first step of the pipleline, the input image (frame) is corrected from distortion using the calibration matrices above. To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one, and as can be seen that the front edge of the car in the bottom of this image is corrected into a round shape.
![alt text][img2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Masking is critical to find pixels of lane edges and the following procedures in the pipeline. The masking is performed by 3 types of individual transform: a) gradient base b) color base and c) region of interest. Then, these individual masks are combined together into a final mask.

The gradient of the gray scaled image is computed using sobel transform to get the dx and dy components. Then, the magnitude of both and the directional angle are thresholded. The results of these are shown in the first and second image in the first row below.

The color thresholding makes use of the fact that lane lines are either yellow or white. The HSV transform is performed first, so that for the yellow color can be filtered out using a small hue range that represents yellow, and a larger value and saturation range for different brightness of yellow. The color mask result is show in the first image in the second row below. (this color mask is based on [umitbinnani's repo](https://github.com/sumitbinnani/CarND-Advanced-Lane-Lines/blob/master/utils/thresholding.py))

Last mask is done by a region of interest representing the view in the front bottom of the car, as shown in the 3rd image in the first row.

![alt text][img3]

All three types of the 4 masks are combined as line 10 below,
```python   
 def threshold(self, img):
    sobelx, sobely = self.get_gradient(img)

    direc = self.dir_thresh(sobelx, sobely)
    mag = self.mag_thresh(sobelx, sobely)
    color = self.color_thresh(img)
    roi = self.roi_thresh(img)

    combined = np.zeros_like(direc)
    combined[((roi == 1) & (color == 1) & ((mag == 1) | (direc == 1)))] = 1

    return combined
```
The original image and the final combined mask are given as the last two figures in the second row. In the final mask, only white and yellow pixels on the lane edges are kept.


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform is performed by function 'computePerspective' in calibration.py.
It is hard coded to use the 'test_images/straight_lines1.jpg' image as a reference and display. The four corner points of a rectangular is used to represent the view transform. 


```python
	# Four corner points of the rectangular used to map perspective
    points_orig = np.array([[312,650], [1007, 650], [720, 470], [568, 470]])
    points_targ = np.array([[360,650], [850, 650], [850, 100], [360, 100]])

    # Compute the forward and reverse persepective transform matrices
    M 	  = cv2.getPerspectiveTransform(np.float32(points_orig), np.float32(points_targ))
    M_inv = cv2.getPerspectiveTransform(np.float32(points_targ), np.float32(points_orig))
```
The source points are marked in red in the original image and in blue in the bird view image as below for reference. They're both shown over the 'straight_lines1.jpg'

![alt text][img6]

The forward and backward transform matrix are also used to transfer the mask results for demostration. 
![alt text][img4]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The 'fitWindowCenter' method of the LaneFinder class performs a sliding window procedure to update window center and find pixels belonging to each edge for the fitting purpose. For each frame, the projection along the y-axis is performed first to find the center of each lane edge as in the first figure below. Then, window centers are updated from bottom to top, if there're enough points within the window, as in the 2nd figure below.
![alt text][img5]

As one of the lane edge is dashed sometimes with much less points comparing to a solid line, the estimated polynomials may be unstable, and the two polynomials can differ quite a lot. My approach for fitting the polynomial is modified a bit from the original way such that the points from both sides of the lane edges are used at the same time to fit one set of polynomials instead of two. More specifically, the pixel coordinates from the left edge are shift toward right by half of the lane width, and likewise, the right ones towards left. Then, these pixels from both lane edges are used together to fit the polynomials. This is done by the 'fitFromPointsInTwoLines' method in LaneFinder as below. 

```python
def fitFromPointsInTwoLines(self, indexOfNonZero_l, indexOfNonZero_r):
    x_l = self.nonzero_x[indexOfNonZero_l] + int(self.laneWidth/2)
    y_l = self.nonzero_y[indexOfNonZero_l]

    x_r = self.nonzero_x[indexOfNonZero_r] - int(self.laneWidth/2)
    y_r = self.nonzero_y[indexOfNonZero_r]

    x_coord = np.concatenate((x_l, x_r), axis=0)
    y_coord = np.concatenate((y_l, y_r), axis=0)
            
    self.fitcoeff = np.polyfit(y_coord, x_coord, 2)
```


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature is calculated by the 'computeCurvature' method in LaneFinder. It maps the coefficients and the y coordinates from pixel to meter. 

```python
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
```


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The lane region is mapped back to the original view with the 'drawLaneBoundaryInOrigView' method. (Note that while using the polynomials to get coordinates, the half-lane-width shift need to compensated, as in the 'getXcoordFromYCoord' method)

The entire process from step 4 to step 6 is performed by the 'run' method of LaneFinder. which also calcualtes the car location relative to the center, and print information on the output image.

```python
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
```

![alt text][img7]

Finally, the entire pipeline is integrated together within the PipeLine class. 
```python
def run_Pipeline(self, img):
    if not self.isInitialized:
        self.initialize()
        
    # Fix lens distortion
    img_undist = cv2.undistort(img, self.distortionMtx, self.distortionDist, None, self.distortionMtx)

        # Generate mask for lane edges and transform to bird view
    thObj = Thresholder()
    mask = thObj.threshold(img)
    mask_bird = cv2.warpPerspective(np.float64(mask), self.M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR)
        
    # Run the lane finder process with includes:
	    # 1. Find lane lines with sliding window;
	    # 2. Fit the lane line coefficients;
	    # 3. Compute curvature;
	    # 4. Mark image with lane lines/regions and display info as text;
    res = self.laneFinderObj.run(img, mask_bird)
    
    # Add additional plot
    out = self.overlay(res, mask, self.laneFinderObj.mask_laneAndWindow)

    return out
```

As additional reference, the mask result and the lane detection windows are add in the upper-right corner of the image as reference. 

![alt text][img8]
---

### Pipeline (video)

Here's the final processed video [file](https://github.com/StevenShuoFeng/CarND-Advanced-Lane-Lines/blob/master/videos/project_video_output.mp4) and [youtube link](https://www.youtube.com/watch?v=RMWPVAgkINY).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

-- I've spent a lot of time to find a masking procedure that works well for all conditions, and it turns out to be very challenging. The final pick is based on other's idea of picking the white and yellow color rigidly.

-- During the lane edge detection and the polynomial fitting, more than frequent, the two edges are not parallel. This is mainly because that one of the edges have much less points in it and the polynomials can go wide. I finally decided to combine the points from two edges together fitting and this generates much more stable results.

