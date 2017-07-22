**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center_lane]: ./images/center_lane.png "Center lane driving"
[center_lane_flipped]: ./images/center_lane_flipped.png "Center lane driving. (Image flipped horizontally)"
[data_udacity]: ./images/udacity_data_distribution.png "Udacity data distrubution"
[data_original]: ./images/initial_data_distribution.png "Distribution of my own data"
[data_dropped]: ./images/dropped_data_distribution.png "Distribution of my own data where half of the samples for steering angle close to 0 is dropped"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depth of size 6 (model.py lines 84 and 86) 

After each convolutional layer there is the max pooling layer of 2x2 size.

The model includes RELU layers to introduce nonlinearity (code lines 84 and 86), and the data is normalized in the model using a Keras lambda layer (code line 83).

The confolutional layers are followed by two fully connected layers of size 120, 84 and the output layer of size 1. (lines 89, 91 and 93 respectievly).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers between the fully connected layers in order to reduce overfitting (model.py lines 90 and 92). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 95-98). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 95).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving the oposite direction.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a convolutional neural network.

My first step was to use a convolution neural network model similar to the one usd for the road signs recognition. I thought this model might be appropriate because we take an image as the input, process it and generate the output (one digit instead of many).

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I introduced dropout.

Then I realized that the fully connected network work well if it is not very large. That helped reducing overfitting as well. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track usually around the corners.

Initially I used the training data provided by Udacity. After I saw the car going straight most of the time, what resulted going to the sea. I generated the data distribution graph.

![data_udacity]

We can see that there is much more data for the car going stright rather than making turns. I thought that could cause the model to be highly biased towards going straight (to the sea). To fix that problem I decided to gather my own data set.

The initial distribution looked like in the chart below

![data_original]

The distribution looks like the standard distribution, but the there is about twice as much data for the car going stright. To battle that I decided to randomly drop 50% of the samples from the initial set (lines 24-25 in model.py). After this operation the charts looks like below and it is quite close to Guassian distribution.

![data_dropped]

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 81-93) consisted of a convolution neural network with the following layers and layer sizes.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   						| 
| Cropping              | Remove top 60 and bottom 25 pixels            |
| Lambda                | Normalize the data                            |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x6    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 				    |
| Fully connected		| Flattend 150 to 120      						|
| Dropout               | 50% during training                           |
| Fully connected		| 120 to 84        						    	|
| Dropout               | 50% during training                           |
| Output layer          | 84 to 1                                       |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![center_lane]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to how steer towards the center of the lane.

To augment the data sat, I flipped images and angles horizontally thinking that this would help avoiding left or right steering bias. For example, here is an image that has then been flipped:

![center_lane]
![center_lane_flipped]

After the collection process, I had 27978 number of data points. I then preprocessed this data by dropping randomly half or the images where the steering angle was close to zero. That resulted in the 23694 data points.

I finally randomly shuffled the data set and put 20% of the data into the validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The model was stabilizing quickly ususal about the third epoch, so I finished the training on the fifth epoch.
