## Vehicle Detection and Tracking


### Goal
The goal is to write a software pipeline to identify vehicles in a video from a front-facing camera on a car. 


### Steps
Following steps were applied:

* Different image features like: spatial color bins, color histogram and HOG(Histogram of Oriented Gradients) computed 
* Relevant set of features for the car detection selected
* Trained classifier on the dataset which contains both cars and not cars images using selected features
* Implemented a sliding-window technique and use your trained classifier to search for vehicles in images
* Used heat map clustering technique to reject false detections  
* Implemented pipeline above to detect and track vehicles on the video stream.
 
[//]: # (Image References)
[image1]: ./output_images/histogram.png
[image2]: ./output_images/heatmap.png
[image3]: ./output_images/hog.png
[image4]: ./output_images/windows_search.png
[image5]: ./output_images/multiple_windows_search.png
[image6]: ./output_images/multiple_detections.png
[image7]: ./output_images/single_detection.png
[image8]: ./output_images/video.png
[video1]: ./project_video.mp4


### Features evaluations

To detect and recognize cars on the video stream firstly we need to figure out relevant car features. Firstly, we explored dataset and extracted simple feature like spatial bins of colors: code can be found in In 4 of the [a project code](sdc-vehicle-detection-and-tracking.ipynb)

Additionally, we can use color histogram feature using different color spaces. Here is histogram feature visualization for car and non-car images:

![alt text][image1]

Code for histogram feature extraction can be found in In 4 of the [a project code](sdc-vehicle-detection-and-tracking.ipynb)

However, these features ignores shape information - so we need a way to include shape information as well. 

### Histogram of Oriented Gradients (HOG) features

HOG algorithm are pretty good for detecting dynamic object as explained in [a this presentation](https://www.youtube.com/watch?v=7S5qXET179I) as it contains both color(changes) and shape information.

I've used [a scikit-image hog() function](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html) to calculate HOG features. I've explored different color spaces to find better representation of car images.

Here is an example using S channel image of `HSV` color space and HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(3, 3)` with `block_norm=L2)`:

![alt text][image3]

Code can be found in In 7-8 of the [a project code](sdc-vehicle-detection-and-tracking.ipynb)

### Training Classifier 

Now we are ready to train classifier using some of the features provided. I've tried a couple of classifiers and it appeared that SVM provides best accuracy with same features set

| Classifier         | Accuracy      | Training time | Prediction Time |
|:-------------------:|:-------------:|:-------------:| :--------------: 
| SVC                 | 98.73%        |  18.13s       | 0.03201s       |
| Logistic Regression | 98.67%        |  27.45s       | 0.0105s        |
| Decision Tree       | 87.97%        |  293.1s       | 0.02523s       |
| AdaBoost with LR    | 98.17%        |  190.36s      | 0.0185s        |

The final step to train classifier is to select features set which provides best accuracy. We can tune both classifier parameters and HOG parameters. After some experiments, I've come up with following parameters set which work best for me:


| Parameter           | Value         | 
|:-------------------:|:-------------:|
| SVC C parameter     | 1             | 
| HOG Bins            | 9             | 
| HOG Block Norm      | L2            | 
| HOG pixel per cell  | 8             |
| HOG cell per block  | 2             |
| HSV channels        | S, V          |
| YUV channels        | U             |

Also, interesting fact that by adding HOG features for the H channel for HLS reduces testing accuracy. Also 12 bins leads to more accuracy but reduces speed so I used 9 bins.

The resulting classifier accuracy for the big cars and non-cars images (17760) is 99.1%

Code for the data preparation and classifier training can be found in In 11-15 of the [a project code](sdc-vehicle-detection-and-tracking.ipynb)

### Sliding Window Search

I've used sliding-window technique to find car image. Here is example of 96x96 window without overlap. It makes sense to restrict sliding search to the region of interest to make search more efficient and reduce false positives count.

![alt text][image4]

However, to detect vehicles of different scales 

I've used following windows sizes and regions to perform full search of cars of different sizes:

| Size          | y-region      |  Pixels shift |
|:-------------:|:-------------:|:-------------:| 
| 80, 80        | 390, 500      |  10           |
| 100, 100      | 390, 610      |  20           |
| 120, 120      | 450, 630      |  30           |

Small windoe is required  for the centre of image to detect small vehicles and lager windoes required closer to the car to detect closer and larger vehicles.

Here is image with all windows searched:

![alt text][image5]

The resulting detections can be found below, as we can see each car has multiple detections points with windows of different sizes:

![alt text][image6]

To deal with this and create single bounding box we can use heatmap technique (add pixel intensity for each box which overlaps this pixel). Additionally, we can use this heatmap and pass it to [a scipy.labels]('https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.measurements.label.html) function to determine clusters of the boxes which is more likely are separate vehicles detected. To reduce amount false positives, we can apply heatmap threshold (select only pixels with intensity > threshold). I've used 1 as threshold as single detection has a great chance to be a false positive.

Here is image of applied techniques below:

![alt text][image7]

Also corresponding heatmap below:

![alt text][image2]

Code for the sliding windows search can be found in In 17-23 of the [a project code](sdc-vehicle-detection-and-tracking.ipynb)

### Video Implementation

We are ready to run vehicle pipeline on the video stream. One issue is that we might get not stable detections which may vary from frame to frame and false positives detections.

To deal with these issues I am collecting heatmaps of positive detection into the specified buffer (for last 3 frames) and calculating sliding sum of the heatmaps. Additionally, I increased threshold compared to individual image detection to remove false detections and make detection and make sure that resulting detection box is drown around solid part of the car. Rather than sliding averages for the heatmap - I've used exactly same technique as for the individual images.  

Here's an example result showing the heatmap and sliding average heatmap for two frames: first image is detection for the current frame, second image is moving average with threshold applied.

![alt text][image8]

Here is a [link to my video result](./project_video_updated.mp4)

Code for the vidio processing can be found in In 29 of the [a project code](sdc-vehicle-detection-and-tracking.ipynb)

### Discussion

The main issues with the current pipeline are following:

* It is slow and cannot be used for real-time detection. There are a couple of tricks can be used to increase pipeline speed - such as: reduce number of sliding windows, make HOG features calculated once per frame, rather than for single sliding window, probably try to reduce number of feature even if it makes classifier to perform worse - it still might work due to multiple detections ability.
 
* The training dataset for the images contains only rear views of the cars so pipeline will likely fail to detect front view cars and side views. Put it is easy to fix by training classifier on these class of images 

* Also, pipeline cannot distinguish two cars when they are close together. It looks like more advanced clustering techniques required rather than heatmap and labeling - so this question is opened for the future investigation.

* Additionaly, there are still two false detection on the right side of the road. The issues there that dataset for clasifier training does not   have similar images of grass. This can be fixed by applying extended dataset for training.    

Additional possible way to improve pipeline is to implement more reliable vehicle tracking for the video. One of the possible method is to provide small number of sliding windows in the beginning and as soon as detection is made, for each consequent frame provide more sliding windows for this specific area to make detection and tracking more solid of detected vehicle.   



