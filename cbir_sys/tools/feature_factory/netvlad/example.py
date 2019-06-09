import cv2
import numpy as np
import tensorflow as tf

import netvlad_tf.net_from_mat as nfm
import netvlad_tf.nets as nets
import time

tf.reset_default_graph()

image_batch = tf.placeholder(
    dtype=tf.float32, shape=[None, None, None, 3])

net_out = nets.vgg16NetvladPca(image_batch)
saver = tf.train.Saver()

sess = tf.Session()
saver.restore(sess, nets.defaultCheckpoint())

inim = cv2.imread(nfm.exampleImgPath())
inim = cv2.cvtColor(inim, cv2.COLOR_BGR2RGB)

t1 = time.time()
batch = np.expand_dims(inim, axis=0)
result = sess.run(net_out, feed_dict={image_batch: batch})
t2 = time.time()

print('Time: ', round(t2 - t1, 3), '\nRes:', result)
