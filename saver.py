import tensorflow as tf

#save to file
#rememeber to define the same dtype and shape
#when restoring

#W = tf.Variable([[1, 2, 3], [1, 2, 3]], dtype= \
#        tf.float32, name='weights')
#b = tf.Variable([[1, 2, 3]], dtype=tf.float32, \
#        name='biases')
#
#
#init = tf.initialize_all_variables()
#saver = tf.train.Saver()
#with tf.Session() as sess:
#    sess.run(init)
#    save_path = saver.save(sess, "my_net/save_net.ckpt")
#    print("Save to path: ", save_path)
#

#resotre variables
# redefine the same shape and same dtype

W = tf.Variable(tf.zeros([2, 3]), dtype=tf.float32, \
        name='weights')

b = tf.Variable(tf.zeros([1, 3]), dtype=tf.float32, \
        name='biases')

# not need init step
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "my_net/save_net.ckpt")
    print('weights:', sess.run(W))
    print("biases", sess.run(b))


