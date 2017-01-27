import tensorflow as tf
from sklearn.datasets import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.prepocessing import LabelBinarizer

#load data
digits = load_digits()
x = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, \
        test_size=0.3)


def add_layer(inputs, in_size, out_size, layer_name, \
        activation_function=None, ):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    #here to dropout
    Wx_plus = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    tf.histogram_summary(layer_name + '/outputs', outputs)
    return outputs


# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, 'l2',\
        activation_function = tf.nn.softmax)

# the loss between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys* \
    tf.log(prediction), reduction_indices[1])) # loss


tf.scalar_summary('loss', cross_entropy)

    


