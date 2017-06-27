# Load pickled data
import pickle


training_file = 'train.p'
validation_file= 'valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


n_train = len(y_train)
n_validation = len(y_valid)
n_test = len(y_test)
image_shape = X_train[0].shape
n_classes = len(set(y_train))


from sklearn import preprocessing
import numpy as np

def scaleData(X):
    X_scaled=[]
    for x in X:
        t=np.zeros(x.shape,dtype=np.float32)
        m=np.mean(x)
        d=np.std(x)
        for i in range(x.shape[2]):
            t[:,:,i]=(x[:,:,i]-m)/d
        X_scaled.append(t)
    return X_scaled


X_train_scaled = scaleData(X_train)
X_valid_scaled = scaleData(X_valid)
X_test_scaled = scaleData(X_test)


import tensorflow as tf
from tensorflow.contrib.layers import flatten

def LeNet(x, n_classes, d1, d2, d3, d4, mu=0.0, sigma=0.1):
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28xd1.       
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, d1), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(d1))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28xd1. Output = 14x14xd1.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 
    
    # Layer 2: Convolutional. Output = 10x10xd2.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, d1, d2), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(d2))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    
    # Pooling. Input = 10x10xd2. Output = 5x5xd2.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 
    
    # Flatten. Input = 5x5xd2. Output = 5x5xd2.
    fc0   = flatten(conv2)

    # Layer 3: Fully Connected. Input = 5x5xd2. Output = d3.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(25*d2, d3), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(d3))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = d3. Output = d4.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(d3, d4), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(d4))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.relu(fc2)
    
    # Layer 5: Fully Connected. Input = d4. Output = n_classes.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(d4, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits


import sys
d1 = int(float(sys.argv[1]))
d2 = int(float(sys.argv[2]))
d3 = int(float(sys.argv[3]))
d4 = int(float(sys.argv[4]))

from sklearn.utils import shuffle

EPOCHS = 100
BATCH_SIZE = 128
rate = 0.001

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)


logits = LeNet(x, n_classes, d1, d2, d3, d4)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


output = 'model_%d_%d_%d_%d_b%d_r%g' % (d1,d2,d3,d4,BATCH_SIZE,rate)
ofile=open(output+'.log','w')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(y_train)
    va0 =  0.0
    mindiff = 0.0005
    for i in range(EPOCHS):
        X_train_scaled, y_train = shuffle(X_train_scaled, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = min(offset + BATCH_SIZE, num_examples)
            batch_x, batch_y = X_train_scaled[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = sess.run(accuracy_operation, feed_dict={x: X_valid_scaled, y: y_valid})
        train_accuracy = sess.run(accuracy_operation, feed_dict={x: X_train_scaled, y: y_train})
        ofile.write('%3d %.4f %.4f\n' % (i,validation_accuracy,validation_accuracy/train_accuracy))
        if abs(validation_accuracy - va0) < mindiff:
            break
        va0 = validation_accuracy

    saver.save(sess, output)
    pfile=open('para.txt','a')
    pfile.write('%3d %3d %3d %3d %.4f %.4f \n' % (d1,d2,d3,d4,validation_accuracy,validation_accuracy/train_accuracy))
    pfile.close()

    test_accuracy = sess.run(accuracy_operation, feed_dict={x: X_test_scaled, y: y_test})
    ofile.write('%3d %.4f %.4f\n' % (EPOCHS,test_accuracy,test_accuracy))
    pfile=open('para_test.txt','a')
    pfile.write('%3d %3d %3d %3d %.4f\n' % (d1,d2,d3,d4,test_accuracy))
    pfile.close()

ofile.close()

