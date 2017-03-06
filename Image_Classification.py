
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import problem_unittests as tests
import tarfile

cifar10_dataset_folder_path = 'cifar-10-batches-py'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile('cifar-10-python.tar.gz'):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            'cifar-10-python.tar.gz',
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open('cifar-10-python.tar.gz') as tar:
        tar.extractall()
        tar.close()


tests.test_folder_path(cifar10_dataset_folder_path)

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

import helper
import numpy as np

# Explore the dataset
batch_id = 1
sample_id = 5
helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)

# ### Normalize

def normalize(x):
    return x/255

tests.test_normalize(normalize)


# ### One-hot encode

one_hot_map = {0:[1,0,0,0,0,0,0,0,0,0],
               1:[0,1,0,0,0,0,0,0,0,0],
               2:[0,0,1,0,0,0,0,0,0,0],
               3:[0,0,0,1,0,0,0,0,0,0],
               4:[0,0,0,0,1,0,0,0,0,0],
               5:[0,0,0,0,0,1,0,0,0,0],
               6:[0,0,0,0,0,0,1,0,0,0],
               7:[0,0,0,0,0,0,0,1,0,0],
               8:[0,0,0,0,0,0,0,0,1,0],
               9:[0,0,0,0,0,0,0,0,0,1]}


def one_hot_encode(x):
    output = []
    for label in x:
        output.append(one_hot_map[label])
    return np.array(output)

tests.test_one_hot_encode(one_hot_encode)


# ### Randomize Data

# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)


# # Check Point
import pickle
import problem_unittests as tests
import helper

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))


# ## Build the network
import tensorflow as tf

def neural_net_image_input(image_shape):
    dim = [None]
    for d in image_shape:
        dim.append(d)
    return tf.placeholder(tf.float32,shape=dim,name="x")


def neural_net_label_input(n_classes):
    return tf.placeholder(tf.float32,shape=(None,n_classes),name="y")


def neural_net_keep_prob_input():
    return tf.placeholder(tf.float32,name="keep_prob")


tf.reset_default_graph()
tests.test_nn_image_inputs(neural_net_image_input)
tests.test_nn_label_inputs(neural_net_label_input)
tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)


# ### Convolution and Max Pooling Layer

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
 
    filter_size = []
    c_strides = [1]
    p_strides = [1]
    p_ksize=[1]
    
    for i,j,k,l in zip(pool_strides,pool_ksize,conv_strides,conv_ksize):
        p_strides.append(i)
        p_ksize.append(j)
        c_strides.append(k)
        filter_size.append(l)
    
    filter_size.append(int(x_tensor.shape[3]))
    filter_size.append(conv_num_outputs)
    p_strides.append(1)
    p_ksize.append(1) 
    c_strides.append(1)
    
    
    filter_weights = tf.Variable(tf.truncated_normal(filter_size,stddev=0.05))
    filter_biases = tf.Variable(tf.zeros(conv_num_outputs)) 
    x_tensor = tf.nn.conv2d(x_tensor, filter_weights, c_strides, padding='SAME')
    x_tensor = tf.nn.bias_add(x_tensor, filter_biases)
    x_tensor = tf.nn.relu(x_tensor)
    x_tensor = tf.nn.max_pool(x_tensor, p_ksize, p_strides, padding='SAME')
    return x_tensor

tests.test_con_pool(conv2d_maxpool)


# ### Flatten Layer

def flatten(x_tensor):
    return tf.reshape(x_tensor, [-1, int(x_tensor.shape[1] * x_tensor.shape[2] * x_tensor.shape[3])])

tests.test_flatten(flatten)


# ### Fully-Connected Layer

def fully_conn(x_tensor, num_outputs):
    weights = tf.Variable(tf.truncated_normal([int(x_tensor.shape[1]),num_outputs]))
    biases = tf.Variable(tf.zeros(num_outputs))
    return tf.nn.relu(tf.add(tf.matmul(x_tensor,weights),biases))

tests.test_fully_conn(fully_conn)


# ### Output Layer
def output(x_tensor, num_outputs):
    weights = tf.Variable(tf.truncated_normal([int(x_tensor.shape[1]),num_outputs]))
    biases = tf.Variable(tf.zeros(num_outputs))
    return tf.add(tf.matmul(x_tensor,weights),biases)

tests.test_output(output)


# ### Create Convolutional Model
def conv_net(x, keep_prob):
   conv1 = conv2d_maxpool(x_tensor=x, 
                           conv_num_outputs=40, 
                           conv_ksize=(5, 5), 
                           conv_strides=(1, 1), 
                           pool_ksize=(2, 2), 
                           pool_strides=(1, 1))
    
    conv2 = conv2d_maxpool(x_tensor=conv1,
                           conv_num_outputs=30,
                           conv_ksize=(4, 4),
                           conv_strides=(1, 1),
                           pool_ksize=(2, 2),
                           pool_strides=(1, 1))
    
    flat_layer = flatten(conv2)

    fc_layer = fully_conn(flat_layer, 15)
    fc_layer = tf.nn.dropout(fc_layer, keep_prob)
   
    output_layer = output(fc_layer, 10)
   
    return output_layer

##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

tests.test_conv_net(conv_net)


# ## Train the Neural Network
# ### Single Optimization
def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    session.run(optimizer, feed_dict = {x:feature_batch, y:label_batch, keep_prob:keep_probability})

tests.test_train_nn(train_neural_network)


# ### Show Stats
def print_stats(session, feature_batch, label_batch, cost, accuracy):
    loss = session.run(cost, feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.0})
    validation_accuracy = session.run(accuracy, feed_dict={x: valid_features, y: valid_labels, keep_prob: 1.0})
    
    print('Loss: {:.4f} Validation Accuracy: {:.4f}'.format(loss, valid_accuracy))


# ### Hyperparameters
epochs = 40
batch_size = 128
keep_probability = 0.7


# ### Train on a Single CIFAR-10 Batch
# Instead of training the neural network on all the CIFAR-10 batches of data, let's use a single batch. This should save time while you iterate on the model to get a better accuracy.  Once the final validation accuracy is 50% or greater, run the model on all the data in the next section.

print('Checking the Training on a Single Batch...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        batch_i = 1
        for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
        print_stats(sess, batch_features, batch_labels, cost, accuracy)


# ### Fully Train the Model

save_model_path = './image_classification'

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)
            
    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)


# # Checkpoint
# The model has been saved to disk.
# ## Test Model
# Test your model against the test dataset.  This will be your final accuracy.

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

import tensorflow as tf
import pickle
import helper
import random

# Set batch size if not already set
try:
    if batch_size:
        pass
except NameError:
    batch_size = 64

save_model_path = './image_classification'
n_samples = 4
top_n_predictions = 3

def test_model():
    """
    Test the saved model against the test dataset
    """

    test_features, test_labels = pickle.load(open('preprocess_training.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        
        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0
        
        for train_feature_batch, train_label_batch in helper.batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)


test_model()