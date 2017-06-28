# Tuning hyperparameters of the network architecture

Here is my network architecture. 

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                     | 32x32x3 RGB image                             | 
| 1. Convolution 5x5        | 1x1 stride, valid padding, outputs 28x28xd1   |
| 1. RELU                               |                                               |
| 1. Max pooling              | 2x2 stride,  outputs 14x14xd1               |
| 2. Convolution 5x5         | 1x1 stride, valid padding, outputs 10x10xd2 |
| 2. RELU                               |                                               |
| 2. Max pooling              | 2x2 stride,  outputs 5x5xd2                 |
| 3. Fully connected            | outputs d3 |
| 3. RELU                               |                                               |
| 4. Fully connected            | outputs d4 |
| 4. RELU                               |                                               |
| 5. Fully connected            | outputs 43 | 

I use grid search to find the best filter size of the convolution layers, d1 and d2, and the output size of the fully connected layers, d3 and d4. The accuracy of the validation set is listed below. The batch size is 128 and the learning rate is 0.001. The model keeps training until the absolute difference of the validation accuracy of the current and the previous epoch is smaller than 0.0005. 

va - validation accuracy

ta - train accuracy

d1   | d2 |  d3 | d4  | va     | va/ta
-----| ---| --- | ----| -------|-------
  6  |64  | 480 | 240 | 0.9193 | 0.9230 
  6  |16  | 120 |  80 | 0.9091 | 0.9236 
  6  |16  | 240 | 160 | 0.9143 | 0.9206 
  6  |64  | 480 | 160 | 0.9156 | 0.9183 
  6  |32  | 480 | 240 | 0.9252 | 0.9266 
  6  |32  | 120 |  80 | 0.9277 | 0.9291 
 12  |32  | 120 |  80 | 0.9268 | 0.9314 
  6  |64  | 120 |  80 | 0.9295 | 0.9339 
  6  |32  | 480 | 160 | 0.9304 | 0.9334 
 24  |64  | 480 | 160 | 0.9247 | 0.9304 
  6  |64  | 480 |  80 | 0.9435 | 0.9437 
 12  |32  | 240 | 160 | 0.9320 | 0.9379 
  6  |64  | 240 | 160 | 0.9528 | 0.9528 
  6  |32  | 480 |  80 | 0.9356 | 0.9375 
  6  |16  | 240 |  80 | 0.9061 | 0.9152 
 24  |32  | 480 | 240 | 0.9240 | 0.9269 
 12  |64  | 240 | 160 | 0.9490 | 0.9490 
 12  |16  | 120 |  80 | 0.9313 | 0.9342 
 12  |32  | 480 | 240 | 0.9213 | 0.9245 
  6  |32  | 240 |  80 | 0.9404 | 0.9404 
 12  |64  | 480 | 240 | 0.9429 | 0.9446 
  6  |64  | 240 |  80 | 0.9617 | 0.9617 
 24  |32  | 240 |  80 | 0.9512 | 0.9512 
 12  |32  | 480 |  80 | 0.9562 | 0.9562 
 12  |32  | 480 | 160 | 0.9476 | 0.9478 
 24  |32  | 480 |  80 | 0.9497 | 0.9497 
 12  |64  | 240 |  80 | 0.9531 | 0.9531 
 12  |16  | 240 |  80 | 0.9522 | 0.9522 
 24  |64  | 480 | 240 | 0.9370 | 0.9385 
 12  |16  | 240 | 160 | 0.9522 | 0.9522 
 24  |32  | 480 | 160 | 0.9535 | 0.9539 
  6  |32  | 240 | 160 | 0.9585 | 0.9585 
 24  |64  | 240 |  80 | 0.9385 | 0.9389 
 24  |64  | 120 |  80 | 0.9526 | 0.9538 
 12  |32  | 240 |  80 | 0.9558 | 0.9558 
 12  |64  | 480 |  80 | 0.9574 | 0.9574 
 24  |64  | 240 | 160 | 0.9424 | 0.9450 
 12  |64  | 120 |  80 | 0.9594 | 0.9594 
 24  |32  | 240 | 160 | 0.9567 | 0.9579 
 12  |64  | 480 | 160 | 0.9438 | 0.9454 
 24  |32  | 120 |  80 | 0.9655 | 0.9655 
 24  |64  | 480 |  80 | 0.9390 | 0.9391 

This work is done at Purdue RHEL6 (Santiago) clusters. There is no binary package of TensorFlow for RHEL systems, so I build it from the source with the latest release version 1.2.0. The results will be slightly different on version 0.12.1 that the course is currently using. 
