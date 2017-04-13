# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[center]: ./data/IMG/center_2016_12_01_13_33_22_111.jpg "Center"
[left]: ./data/IMG/left_2016_12_01_13_36_04_931.jpg "Left"
[right]: ./data/IMG/right_2016_12_01_13_38_13_946.jpg "Right"
[cropped]: ./cropped.png "Cropped"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results
* agent.mp4 showing the video seen by the agent

A screen capture video of the vehicle driving around the track is viewable [here](https://youtu.be/KOkM3KufboY).

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolutional neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 filters (model.py lines 60-64) 

The model includes RELU layers to introduce nonlinearity (model.py lines 60-64), and the data is normalized in the model using a Keras Lambda layer (model.py line 56). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 68 and 70). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 89). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 74).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, along with data from the left and right vehicle cameras.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step in arriving at a working architecture was to use the convolution neural network model used to train NVIDIA's autonomous vehicle steering, published [here](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). I found this model to be appropriate because of their success with this similar problem.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my model had a low mean squared error on the training set but a higher mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to include dropout layers at the first two fully connected layers. I tuned the keep probability of these dropout layers and found that the training set error and validation set error had been reduced and converged closest with a 0.3 keep probability.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. The vehicle was turning through the sharp turns smoothly, but it would zig-zag through the gradual turns and eventually drive off the road. To improve the driving behavior in these cases, I recorded additional data on the slightly curved sections of the road, trying to maintain a steady, small steering angle through the turn. This improved the behavior of the vehicle through these sections.

At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road or crossing the lane lines.

#### 2. Final Model Architecture

The final model architecture (model.py lines 50-74) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		    | Description	        					                | 
|:---------------------:|:---------------------------------------------:| 
| Input         		    | 320x160x3 RGB image   							          | 
| Normalization      	  | (X / 255) - 0.5  	                            |
| Cropping					    |	225x160x3 RGB image											      |
| Convolution 5x5	      | 2x2 stride, valid padding, outputs 111x78x24  |
| RELU          		    |         									                    |
| Convolution 5x5				| 2x2 stride, valid padding, outputs 54x37x36   |
|	RELU					        |	                  											      |
| Convolution 5x5				| 2x2 stride, valid padding, outputs 25x17x48   |
|	RELU					        |	                  											      |
| Convolution 3x3				| 1x1 stride, valid padding, outputs 23x15x64   |
|	RELU					        |	                  											      |
| Convolution 3x3				| 1x1 stride, valid padding, outputs 21x13x64   |
|	RELU					        |	                  											      |
|	Flatten					      |	outputs 17472 params            						  |
|	Fully connected				|	outputs 100 params											      |
|	Dropout          			|	0.3 keep prob                  								|
|	Fully connected				|	outputs 50 params											        |
|	Dropout          			|	0.3 keep prob                  								|
|	Fully connected				|	outputs 10 params											        |
|	Fully connected				|	outputs 1 param   										        |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center][center]

I then included images from the left and right cameras so that the vehicle would learn to adjust its steering angle to correct itself towards the center of the road. The images were fed to the model with a steering angle correction value of ±0.05 to indicate that the steering should adjust towards center. Here are examples of the left and right camera images:

![Left][left]
![Right][right]

Notice the vehicle and road offset from the center of the images. I found the tuning of this value to have the most affect on the performance of the autonomous vehicle. The vehicle was able to stay within the center of the road with much less zig zag through experimentations of different correction values. 

Because the track had many more left turns than right turns, I recorded a lap travelling around the track backwards. To mitigate this left turn bias, I also flipped all the images and their corresponding angles to augment the data set.

The data set became too large for memory with all of the augmentation. Thus, I used a Python generator in order to generate the data as the model trained. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

As part of the training model, I normalized the image to bring the data to zero mean and equal variance. I also cropped the unnecessary portions of the image, such as the sky above the road and the hood of the car. Here is an example of the cropped image.

![Cropped][cropped]

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. Through experimentation, I found that the validation loss flattened out at 5 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
