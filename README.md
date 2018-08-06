## Advanced Lane Detection

### Shuo Feng

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
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

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


### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

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

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][img3]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][img4]
![alt text][img6]

A final combined mask is given in the last figure. Only white and yellow pixels on the lane edges are left.


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][img5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:


![alt text][img7]

Finally, the mask result and the lane detection windows are add in the upper-right corner of the image as reference. 

![alt text][img8]
---

### Pipeline (video)
![alt text][video]

Here's the final processed video [file](https://github.com/StevenShuoFeng/CarND-Advanced-Lane-Lines/blob/master/videos/project_video_output.mp4) and [youtube](https://www.youtube.com/watch?v=VmKGCWJCwug)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
