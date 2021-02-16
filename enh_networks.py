import tensorflow as tf
from cell import QGConvLSTMCell
import numpy as np

def dense(x, step):

    for i in range(step):

        if i == 0:
            reuse = False
        else:
            reuse = True

        x_1 = x[:,i,:]

        x_2 = tf.layers.dense(x_1, 1, kernel_initializer=tf.random_normal_initializer, reuse=reuse)

        if i == 0:
            x_o = tf.expand_dims(x_2, 1)
        else:
            x_o = tf.concat([x_o, tf.expand_dims(x_2, 1)], axis=1)

    return x_o

def resblock(input, IC, OC, kernel, name):

    l0 = tf.nn.relu(input)

    l1 = tf.layers.conv2d(inputs=l0, filters=np.minimum(IC, OC), kernel_size=kernel, strides=1, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                          name=name + 'l1', reuse=tf.AUTO_REUSE)

    l1 = tf.nn.relu(l1)

    l2 = tf.layers.conv2d(inputs=l1, filters=OC, kernel_size=kernel, strides=1, padding='same', reuse=tf.AUTO_REUSE,
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name=name + 'l2')

    if IC != OC:
        input = tf.layers.conv2d(inputs=input, filters=OC, kernel_size=1, strides=1, padding='same', reuse=tf.AUTO_REUSE,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name=name + 'map')

    output = input + l2

    return output

def CNN(x, step, filter_no, filter_no_last, kernel, relu, layer, scale, name):

    for i in range(step):

        if i == 0:
            reuse = False
        else:
            reuse = True

        x_i = x[:,i,:,:,:]

        if scale == True:

            x_i_1 = tf.layers.conv2d(x_i, filter_no // 3, [3, 3], padding='SAME', reuse=reuse,name='conv' + name + '_' + '01')
            x_i_2 = tf.layers.conv2d(x_i, filter_no // 3, [5, 5], padding='SAME', reuse=reuse,name='conv' + name + '_' + '02')
            x_i_3 = tf.layers.conv2d(x_i, filter_no // 3, [7, 7], padding='SAME', reuse=reuse,name='conv' + name + '_' + '03')
            x_i = tf.concat([x_i_1, x_i_2, x_i_3], axis=-1, name='conv' + name + '_' + str(0))

        elif scale == False:

            x_i = tf.layers.conv2d(x_i, filter_no, kernel, padding='SAME', reuse=reuse, name='conv' + name + '_' + str(0))

        if relu == 1:
            x_i = tf.nn.relu(x_i)

        for ii in range(1, layer - 1):

            x_i = tf.layers.conv2d(x_i, filter_no, kernel, padding='SAME', reuse=reuse, name='conv' + name + '_' + str(ii))

            if relu == 1:
                x_i = tf.nn.relu(x_i)

        x_i = tf.layers.conv2d(x_i, filter_no_last, kernel, padding='SAME', reuse=reuse, name='conv' + name + '_' + str(layer - 1))

        if relu == 1:
            if filter_no_last == filter_no:
                x_i = tf.nn.relu(x_i)

        if i == 0:
            x_o = tf.expand_dims(x_i, 1)
        else:
            x_o = tf.concat([x_o, tf.expand_dims(x_i, 1)], axis=1)

    return x_o

def CNN_res_1(x, step, filter_no, kernel, kernel_small):

    for i in range(step):

        x_0 = x[:,i,:,:,:]

        x_0 = tf.layers.conv2d(inputs=x_0, filters=filter_no, kernel_size=kernel, padding='same',
                               reuse=tf.AUTO_REUSE,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='res1_0')

        x_1 = resblock(x_0, filter_no, filter_no, kernel_small, name='res1_1')
        x_2 = resblock(x_1, filter_no, filter_no, kernel_small, name='res1_2')
        x_3 = resblock(x_2, filter_no, filter_no, kernel_small, name='res1_3')

        x_3 = x_3 + x_0

        x_4 = tf.layers.conv2d(inputs=x_3, filters=filter_no, kernel_size=kernel_small, strides=1, padding='same',
                               reuse=tf.AUTO_REUSE,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='res1_4')

        if i == 0:
            x_k1 = tf.expand_dims(x_1, 1)
            x_k2 = tf.expand_dims(x_2, 1)
            x_k3 = tf.expand_dims(x_3, 1)
            x_o = tf.expand_dims(x_4, 1)
        else:
            x_k1 = tf.concat([x_k1, tf.expand_dims(x_1, 1)], axis=1)
            x_k2 = tf.concat([x_k2, tf.expand_dims(x_2, 1)], axis=1)
            x_k3 = tf.concat([x_k3, tf.expand_dims(x_3, 1)], axis=1)
            x_o = tf.concat([x_o, tf.expand_dims(x_4, 1)], axis=1)

    return x_k1, x_k2, x_k3, x_o

def CNN_res_2(x, x_skip1, x_skip2, x_skip3, step, filter_no, kernel, kernel_small):

    for i in range(step):

        x_0 = x[:,i,:,:,:]
        x_k3_i = x_skip3[:, i, :, :, :]
        x_k2_i = x_skip2[:, i, :, :, :]
        x_k1_i = x_skip1[:, i, :, :, :]

        x_0 = tf.layers.conv2d(inputs=x_0, filters=filter_no, kernel_size=kernel_small, padding='same',
                               reuse=tf.AUTO_REUSE,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='res2_0')

        x_1 = resblock(x_0 + x_k3_i, filter_no, filter_no, kernel_small, name='res2_1')
        x_2 = resblock(x_1 + x_k2_i, filter_no, filter_no, kernel_small, name='res2_2')
        x_3 = resblock(x_2 + x_k1_i, filter_no, filter_no, kernel_small, name='res2_3')

        x_3 = x_3 + x_0

        x_4 = tf.layers.conv2d(inputs=x_3, filters=3, kernel_size=kernel, strides=1, padding='same',
                               reuse=tf.AUTO_REUSE,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='res2_4')

        if i == 0:
            x_o = tf.expand_dims(x_4, 1)
        else:
            x_o = tf.concat([x_o, tf.expand_dims(x_4, 1)], axis=1)

    return x_o

def net_bi_wcell(x, f, u, step, Height, Width, filter_num, kernel, relu, CNNlayer, peephole, scale):

    x1, x2, x3, xo = CNN_res_1(x, step, filter_num, kernel, kernel)

    inputs = tf.concat([xo, f, u], axis=-1)

    cell = QGConvLSTMCell(shape=[Height, Width], activation=tf.nn.relu, filters=filter_num, kernel=kernel, peephole=peephole)

    x_lstm, state = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs, dtype=inputs.dtype)

    x_lstm_output = tf.concat([x_lstm[0], x_lstm[1]], axis=4)

    x4 = CNN_res_2(x_lstm_output, x1, x2, x3, step, filter_num, kernel, kernel)

    return x4


def net_bi_wcell_ssim(x, f, u, step, Height, Width, filter_num, kernel, relu, CNNlayer, peephole, scale):

    x1 = CNN(x, step, filter_num, filter_num, kernel, relu, CNNlayer, scale=scale, name="1")

    inputs = tf.concat([x1, f, u], axis=-1)

    cell = QGConvLSTMCell(shape=[Height, Width], filters = filter_num, kernel = kernel, peephole = peephole)

    x2, state = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs, dtype = inputs.dtype)

    x22 = tf.concat([x2[0], x2[1]], axis=4)

    x3 = CNN(x22, step, filter_num, 3, kernel, relu, CNNlayer, scale=False, name="2")

    return x3




