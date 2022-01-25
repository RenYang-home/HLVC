import tensorflow as tf
import numpy as np

def convnet(im1_warp, im2, flow, layer):

    with tf.variable_scope("flow_cnn_" + str(layer), reuse=tf.AUTO_REUSE):

        input = tf.concat([im1_warp, im2, flow], axis=-1)

        conv1 = tf.layers.conv2d(inputs=input, filters=32, kernel_size=[7, 7], padding="same",
                                 activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[7, 7], padding="same",
                                 activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(inputs=conv2, filters=32, kernel_size=[7, 7], padding="same",
                                 activation=tf.nn.relu)
        conv4 = tf.layers.conv2d(inputs=conv3, filters=16, kernel_size=[7, 7], padding="same",
                                 activation=tf.nn.relu)
        conv5 = tf.layers.conv2d(inputs=conv4, filters=2 , kernel_size=[7, 7], padding="same",
                                 activation=None)

    return conv5


def loss(flow_course, im1, im2, layer):

    flow = tf.image.resize_images(flow_course, [tf.shape(im1)[1], tf.shape(im2)[2]])
    im1_warped = tf.contrib.image.dense_image_warp(im1, flow)
    res = convnet(im1_warped, im2, flow, layer)
    flow_fine = res + flow

    im1_warped_fine = tf.contrib.image.dense_image_warp(im1, flow_fine)
    loss_layer = tf.reduce_mean(tf.squared_difference(im1_warped_fine, im2))

    return loss_layer, flow_fine


def optical_flow(im1_4, im2_4, batch, h, w):

    im1_3 = tf.layers.average_pooling2d(im1_4, pool_size=2, strides=2, padding='same')
    im1_2 = tf.layers.average_pooling2d(im1_3, pool_size=2, strides=2, padding='same')
    im1_1 = tf.layers.average_pooling2d(im1_2, pool_size=2, strides=2, padding='same')
    im1_0 = tf.layers.average_pooling2d(im1_1, pool_size=2, strides=2, padding='same')

    im2_3 = tf.layers.average_pooling2d(im2_4, pool_size=2, strides=2, padding='same')
    im2_2 = tf.layers.average_pooling2d(im2_3, pool_size=2, strides=2, padding='same')
    im2_1 = tf.layers.average_pooling2d(im2_2, pool_size=2, strides=2, padding='same')
    im2_0 = tf.layers.average_pooling2d(im2_1, pool_size=2, strides=2, padding='same')

    flow_zero = tf.zeros([batch, h//16, w//16, 2])

    loss_0, flow_0 = loss(flow_zero, im1_0, im2_0, 0)
    loss_1, flow_1 = loss(flow_0, im1_1, im2_1, 1)
    loss_2, flow_2 = loss(flow_1, im1_2, im2_2, 2)
    loss_3, flow_3 = loss(flow_2, im1_3, im2_3, 3)
    loss_4, flow_4 = loss(flow_3, im1_4, im2_4, 4)

    return flow_4, loss_0, loss_1, loss_2, loss_3, loss_4

def tf_inverse_flow(flow_input, b, h, w):

    # x = vertical (channel 0 in flow), y = horizontal (channel 1 in flow)
    flow_list = tf.unstack(flow_input)

    x, y = tf.meshgrid(tf.range(h), tf.range(w), indexing='ij')

    x = tf.expand_dims(x, -1)
    y = tf.expand_dims(y, -1)

    grid = tf.cast(tf.concat([x, y], axis=-1), tf.float32)

    for r in range(b):

        flow = flow_list[r]
        grid1 = grid - flow

        x1, y1 = tf.split(grid1, [1, 1], axis=-1)
        x1 = tf.clip_by_value(x1, 0, h - 1)
        y1 = tf.clip_by_value(y1, 0, w - 1)
        grid1 = tf.concat([x1, y1], axis=-1)
        grid1 = tf.cast(grid1, tf.int32)

        tf_zeros = tf.zeros([h, w, 1, 1], np.int32)
        indices = tf.expand_dims(grid1, 2)
        indices = tf.concat([indices, tf_zeros], axis=-1)

        flow_x, flow_y = tf.split(flow, [1, 1], axis=-1)

        ref_x = tf.Variable(np.zeros([h, w, 1], np.float32), trainable=False, dtype=tf.float32)
        ref_y = tf.Variable(np.zeros([h, w, 1], np.float32), trainable=False, dtype=tf.float32)
        inv_flow_x = tf.scatter_nd_update(ref_x, indices, -flow_x)
        inv_flow_y = tf.scatter_nd_update(ref_y, indices, -flow_y)
        inv_flow_batch = tf.expand_dims(tf.concat([inv_flow_x, inv_flow_y], axis=-1), axis=0)

        if r == 0:
            inv_flow = inv_flow_batch
        else:
            inv_flow = tf.concat([inv_flow, inv_flow_batch], axis=0)

    return inv_flow
