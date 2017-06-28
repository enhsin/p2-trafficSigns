# Adding dropout layer

Here is my network architecture with dropout added at layer 3.

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                     | 32x32x3 RGB image                             | 
| 1. Convolution 5x5        | 1x1 stride, valid padding, outputs 28x28x24   |
| 1. RELU                               |                                               |
| 1. Max pooling              | 2x2 stride,  outputs 14x14x24               |
| 2. Convolution 5x5         | 1x1 stride, valid padding, outputs 10x10x32 |
| 2. RELU                               |                                               |
| 2. Max pooling              | 2x2 stride,  outputs 5x5x32                 |
| 3. Fully connected            | outputs 120 |
| 3. RELU                               |                                               |
| 3. Dropout                    |                                               |
| 4. Fully connected            | outputs 80 |
| 4. RELU                               |                                               |
| 5. Fully connected            | outputs 43 | 

The accuracy on the validation and test set with dropout is listed below. The batch size is 128 and the learning rate is 0.001. The model is trained for 30 epochs using TensorFlow v1.2.0. It can be downloaded [here](./).

dropout   | validation accuracy    | valid acc/train acc   | test accuracy
------|-------| -------|-------
  0.5 |0.9773 | 0.9878 |0.9568
  0.6 |0.9673 | 0.9788 |0.9576
  0.7 |0.9748 | 0.9967 |0.9591 

The key thing is to set _keep_prob_ = 1 when evaluating validation and test accuracy. I forgot that in the beginning and found no benefit of adding a dropout layer.
```
train_accuracy = sess.run(accuracy_operation, feed_dict={x: X_train_scaled, y: y_train, keep_prob: 1.-dropout})
validation_accuracy = sess.run(accuracy_operation, feed_dict={x: X_valid_scaled, y: y_valid, keep_prob: 1.})
```

