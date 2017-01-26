import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, n_layer), \
        activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = "layer%s" % n_layer
    with tf.name_scope("layer"):
        with tf.name_scope("weights"):
            Weights = tf.Variable(tf.random_normal([in_size, \
            out_size]), name='W')
            tf.histogram_summary(layer_name+"/weights", Weights)
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            tf.histogram_summary(layer_name+"/weights", Weights)
        with tf.name_scope("inputs"):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs



#Make up some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise


#plt.scatter(x_data, y_data)
#plt.show()

# define placeholder for inputs to network

with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_inputs')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_inputs')

#add hidden layers
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

#add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

#the error between prediction and real data

with tf.name_scope("loss"):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),\
        reduction_indices=[1]))
    tf.scalar_summary("loss",loss)
with tf.name_scope("train"):
    train_step = \
    tf.train.GradientDescentOptimizer(0.1).minimize(loss)


sess = tf.Session()
writer = tf.summary.FileWriter("logs/",sess.graph)

#the important step
init = tf.global_variables_initializer()
sess.run(init)

#plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

for i in range(5000):
    # training
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    pass
    if i % 100 == 0:
        # to see the step improvement
        pass
        #print(sess.run(loss,feed_dict={xs:x_data, ys:y_data}))
        #visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs:x_data})
        #plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=3)
        plt.pause(1)


plt.show(block=True)
        








#define 

#
