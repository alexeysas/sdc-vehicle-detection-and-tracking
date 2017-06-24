## Vehicle Detection and Tracking


### Goal
The goal is to write a software pipeline to identify vehicles in a video from a front-facing camera on a car. 


### Steps
Following steps were applied:

* Different image features like: spatial color bins, color histogram and HOG(Histogram of Oriented Gradients) computed 
* Relevant set of features for the car detection selected
* Trained classifier on the dataset which contains both cars and not cars images using seleced features
* Implemented a sliding-window technique and use your trained classifier to search for vehicles in images
* Used heat map clustering teqnique to reject false detections  
* Implemented pipline above to detect and track vehicles on the video stream.


**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.
 
[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./output_images/hog.png
[image4]: ./output_images/windows_search.png
[image5]: ./output_images/multiple_windows_search.png
[image6]: ./output_images/multiple_detections.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4


### Features evaluations


### Histogram of Oriented Gradients (HOG) features

HOG algorithm are pretty good for detecting dinami obkject as explained in [a this presentation](https://www.youtube.com/watch?v=7S5qXET179I)

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image3]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

### Training Classifier 

Now we are ready to train classifier using  extracted features. I've tried a couple of classifiers and it apperared that SVM provides best accuracy with same features set

| Classiifier         | Accuracy      | Training time | Predction Time |
|:-------------------:|:-------------:|:-------------:| :--------------: 
| SVC                 | 98.73%        |  18.13s       | 0.03201s       |
| Logistic Regression | 98.87%        |  27.45s       | 0.0105s        |
| Decision Tree       | 87.97%        |  293.1s       | 0.02523s       |
| AdaBoost with LR    | 98.17%        |  190.36s      | 0.0185s        |

The final step to train classifier is to select features set which provides best accuracy. We can yune both classifier parameters and HOG parametre

| Parameter           | Value         | 
|:-------------------:|:-------------:|
| SVC C parameter     | 0.05          | 
| HOG Bins            | 12            | 
| HOG Block Norm      | L2            | 
| HOG pixel per cell  | 8             |
| HOG cell per block  | 2             |
| HSV channels        | S, V          |
| YUV channels        | U             |


I tried various combinations of parameters and...

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

### Sliding Window Search

I've used sliding-window technique to find car image. Here is example of 96x96 window without overlap. It makes sense to restrict sliding search to the region of interest to make search more efficient.  

![alt text][image4]

However, to detect vehicles of different scales 

I've used following windows sizes and regions to perform full search of cars of different sizes:

| Size          | y-region      |  Overlap |
|:-------------:|:-------------:|:--------:| 
| 72, 72        | 400, 600      |  50%     |
| 96, 96        | 400, 650      |  75%     |
| 128, 128      | 450, None     |  75%     |

Here is image of all windows searched:

![alt text][image5]

The resulting detection cane be found below, as we can see each car has multiple detection points with windoes of different sizes:

![alt text][image6]



####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4](width=48)
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

