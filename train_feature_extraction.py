import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet
import numpy as np

# TODO: Load traffic signs data.
training_file = r"C:\Users\Christoph\Documents\udacity\08_CNN\TrafficSignClassifier\CarND-Traffic-Sign-Classifier-Project\traffic-signs-data\train.p"
testing_file = r"C:\Users\Christoph\Documents\udacity\08_CNN\TrafficSignClassifier\CarND-Traffic-Sign-Classifier-Project\traffic-signs-data\test.p"

training_file = r".\train.p"
testing_file = r".\test.p"


with open(training_file, mode='rb') as f:
    train = pickle.load(f)
    print("loaded file: {}".format(training_file))
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    print("loaded file: {}".format(testing_file))
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
print("\n")

# TODO: Split data into training and validation sets.
### Replace each question mark with the appropriate value.

# TODO: Number of training examples

n_train =X_train.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = [X_train.shape[1:-1]]

# TODO: How many unique classes/labels there are in the dataset.
n_classes =y_train.max()-y_train.min()

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
X_train, y_train = shuffle(X_train, y_train)

#X_train = (X_train-128)/128
 
X_train = (X_train - X_train.mean()) / (np.max(X_train) - np.min(X_train))
X_test_org = X_test
X_test = (X_test - X_test.mean()) / (np.max(X_test) - np.min(X_test))    

X_train, X_validation, y_train, Y_validation = train_test_split(X_train, y_train,
                                                  test_size=0.2,
                                                  random_state=42)
# TODO: Define placeholders and resize operation.

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

shape = (fc7.get_shape().as_list()[-1], 43)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(43))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
probs = tf.nn.softmax(logits)

# TODO: Add the final layer for traffic sign classification.

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, depth=43, on_value=1., off_value=0., axis=-1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
rate = 0.001
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# TODO: Train and evaluate the feature extraction model.
import time
EPOCHS = 20
BATCH_SIZE = 100
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    t1  =time.clock()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        t2 = time.clock()   
        validation_accuracy = evaluate(X_validation, Y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Time: {}".format(t2-t1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
    t3 = time.clock()
    print("Time Total: {}".format(t3-t1))
    saver.save(sess, r'.\lenet3.chpt')
    print("Model saved")