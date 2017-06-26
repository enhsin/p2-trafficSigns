# **Traffic Sign Recognition** 


### Dataset Exploration
#### 1. Dataset Summary 
* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Number of classes = 43
* Input image data shape = (32, 32, 3)

#### 2. Exploratory Visualization 
Here is an example of 43 traffic signes from the dataset. 
![alt text](./images/sign.png "traffic signs")

Here is the distribution of classes. Some signs have more examples, which could result a more accurate prediction on those signs. The distribution of the training set is similar to those of the validation and test sets. If the model is not overfit, the overall accuracy on the validation and test sets should not be too different from that of the training set.
![alt text](./images/class_hist.png "histogram")

### Design and Test a Model Architecture

#### 1. Preprocessing
The image has been normalized by subtracting the mean of the image (flatten array of all three channels) and dividing by the standard deviation of the image. This makes all the feature center around zero and have unit variance. Normalization helps the algorithm learn features from low-contrast images better. 

#### 2. Model Architecture 
I use [LeNet-5](https://github.com/udacity/CarND-LeNet-Lab) taught in the Convolutional Neural Network (CNN) lesson as my model. The model consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         	      	| 32x32x3 RGB image   							| 
| 1. Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| 1. RELU					            |												|
| 1. Max pooling	      	  | 2x2 stride,  outputs 14x14x24 				|
| 2. Convolution 5x5	     | 1x1 stride, valid padding, outputs 10x10x32 |
| 2. RELU					            |												|
| 2. Max pooling	      	  | 2x2 stride,  outputs 5x5x32 				|
| Flatten              | outputs 800 |
| 3. Fully connected		    | outputs 120 |
| 3. RELU					            |												|
| 4. Fully connected		    | outputs 80 |
| 4. RELU					            |												|
| 5. Fully connected		    | outputs 43 | 


#### 3. Model Training

To train the model, I used an ....

#### 4. Solution Approach

Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text](./images/new_images_5.png "new images")

The second image might be challenging because the sign is dirty and the colors have faded.  The last one is partially covered by the snow. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


