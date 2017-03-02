# Project 4 - Advanced Lane Finding

## Goals of this project:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The camera calibration is performed in calibrate_camera.py. I start by preparing the object points for a 9x6 chessboard. Object points are the 3D coordinants of the corners of the chessboard in the real world, with the chessboard flat in the z plane. The object point will be the same for all calibration images, since it is assumed that they represent the same chessboard.

I then find image points using cv2.findChessboardCorners(). The image points are the x,y coordinates of the chessboard corners in the distorted image. The image points are then used to draw the corner points on the distorted image. This is done for each calibration image, with the object points and image points being collected for each image. 

The collected object points and image points are fed into cv2.calibrateCamera to return the camera matrix, distortion coefficients, rotation vector, and translation vector. See below for example of undistortion:

![alt text][./examples/original_chessboard.png "Original"]
![alt text][./examples/draw_corners.png "Draw Corners"]
![alt text][./examples/undistorted_chessboard.png "Undistorted"]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.

Individual images are undistorted by passing the image, camera matrix, and distortion coefficents to cv2.undistort(). See below for undistorted test image:

![alt text][./examples/undistorted_test.png "Undistorted"]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

My thresholding functions are found in video_gen.py (lines 12-72). The application and  combination of thresholding techniques is found in lines 155-164. I used a combination of gradient thresholds in the x orientation, gradient thresholds in the y orientation, and color thresholds using the S channel in HLS colorspace and the V channel in HSV colorspace. Through a fine-tuning process of trial and error, I was able to genetate a binary image like the one below:

![alt text][./examples/binary.png "Binary Image"]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Source and destination points were chose by manually identfying four points on an image with straight lanes (video_gen.py lines 168-169):


| Source        | Destination   | 
|:-------------:|:-------------:| 
| 594, 450      | 320, 0        | 
| 223, 720      | 320, 720      |
| 1106, 720     | 960, 720      |
| 688, 450      | 960, 0        |

I verified that my perspective transform was working as expected by verfiying that the top and bottom of the lanes are approximately the same distance apart in the transformed images.

![alt text][./examples/warped.png "Transformed Image"]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I first found window centers using find__window_centroids() in lines 86- 141. This function convolved the windows on vertical slices of the images. Sanity checks are done on the window centers, which are then averaged with with the last 15 frames. If window centroids are found, I draw windows around the centroids and add them to the transformed image (lines 185-220). I then fit a second order polynomial to the window centers on both the left and right lanes, and draw the lanes onto an unwarped image (lines 222-255).

![alt text][./examples/result.png "Resulting Image"]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of the curvature was calcuated in lines 260-268 using the equation Udacity provided, after converting pixels to meters in both the x and y orientation. The position of the vehicle was calculated in lines 270-275, assuming the camera is in the center of the vehicle. 

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][./examples/result.png "Resulting Image"]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The below picture is a link to the youtube video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/KtwqLI2NxCE/0.jpg)](https://youtu.be/KtwqLI2NxCE)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I spent a lot of time fine-tuning the binary images, along with the moving window dimensions and margins. These things seemed to have the most effect in producing a satisfying result. Other things I found to be important were averaging the window centers over previous frames, and implementing some sort of sanity check. I only implemented a sanity check on the first layer of windows, but I think a sanity check on the curve could possibly give better results. My pipeline fails when there are more shadows or cracks in the road that look like lanes. With more time, more improvement could be done with thresholding and sanity checks. 
