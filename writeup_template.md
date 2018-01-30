# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/test_dataset.png "Test Dataset"
[image3]: ./examples/train_dataset.png "Train Dataset"
[image4]: ./examples/valid_dataset.png "Validation Dataset"
[image5]: ./examples/beforenormal.png "Image before normalization"
[image6]: ./examples/normalized.png "Normalized Image"
[image7]: ./examples/grayscale.png "Grayscale Image"
[image8]: ./examples/augmented.png "Augmented Image"
[image9]: ./web_images/trafficsign1.png "Traffic Sign 1"
[image10]: ./web_images/trafficsign2.png "Traffic Sign 2"
[image11]: ./web_images/trafficsign3.png "Traffic Sign 3"
[image12]: ./web_images/trafficsign4.png "Traffic Sign 4"
[image13]: ./web_images/trafficsign5.png "Traffic Sign 5"
[image14]: ./examples/prediction.png "Prediction"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  
Files submitted:
The project includes the notebook, the html file and the write up document. So all documents shall be included.

Dataset Exploration:
The size of the dataset is printed. A random traffic sign is visualized.
The distribution to the classes is shown in bar charts.

Design and Test a Model Architecture:
Some data preprocessing is used (normalization, convertion to grayscale and perturbation in position scale and rotation).
The Model Architecture includes the LeNet and the evaluation function.
The model was trained and get an accuracy higher than 0.93.

Test a Model on New Images:
Five new German Traffic signs were downloaded from web. The performance on the new images is shown and the softmax probabilities are printed.
---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)	
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

The code contained in the third cell of the IPython notebook.

Here is an exploratory visualization of the data set. It shows a random image from the dataset and shows the distribution of the train, test and validation dataset in bar charts.

![Visualization][image1]
![Test Dataset][image2]
![Train Dataset][image3]
![Valid Dataset][image4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to normalize the images.
Then, I converted the image data to grayscale because in Lesson 8.2 we learned that is it easier for the classifier to learn with grayscale images.

Here is an example of a traffic sign image before and after normalizing and in grayscale.

![Image before normalization][image5]
![Normalized Image][image6]
![Grayscale Image][image7]

I decided to generate additional data because with more input data the net could get a higher accuracy. So I perturbated the image in position, scale an angle like written in the linked paper.

Here is an example of an original image and an augmented image:

![Augmented Image][image8]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flatten				| intput 5x5x16, output 400 					|
| Fully connected		| intput 400, output 120						|
| RELU					| 	        									|
| Fully connected		| intput 120, output 84							|
| RELU					|												|
| Fully connected		| intput 84, output 43							|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the LeNet which is described in the classroom. I took the same code with the same parameters. The only parameter that I changed is the size of the epochs. It is set too 20 epochs which get me a better accuracy.
So here is a potential to get a better result if more parameters were changed or if an other net is used or if the same net with more layers is used.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.992
* validation set accuracy of 0.945 
* test set accuracy of 0.933

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
** First I tried the LeNet described in the classroom. I took the same code as in the classroom and got an accuracy of ~0.89.
* What were some problems with the initial architecture?
** I missed to set the output of the last layer from 10 too 43 and got an accuracy of ~0.04. After solving that issue I got an accuracy of ~0.89.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
** The only adjustment I have done is too add more input images by using image augmentation. I have nothing changed on the LeNet Architecture.
* Which parameters were tuned? How were they adjusted and why?
** I changed the epochs too 20 so that the net learned longer. That gave me a better accuracy. I changed even the learning rate but I recognized that a learning rate of 0.001 is already good to train this net.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Trafficsign 1][image9] ![Trafficsign 2][image10] ![Trafficsign 3][image11] 
![Trafficsign 4][image12] ![Trafficsign 5][image13]

I think that images 3 and 5 could be difficult because not the whole sign is on the image. But I dont think that the other images are difficult.
I made them all uniform in size.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![Prediction][image14]


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

See the question above.
