import tensorflow as tf

class QGConvLSTMCell(tf.nn.rnn_cell.RNNCell):
  """A LSTM cell with convolutions instead of multiplications.
  Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
  """

  def __init__(self, shape, filters, kernel, forget_bias=1.0, activation=tf.tanh, normalize=False, peephole=False, data_format='channels_last', reuse=None):
    super(QGConvLSTMCell, self).__init__(_reuse=reuse, name='conv_lstm_cell')
    self._kernel = kernel
    self._filters = filters
    self._forget_bias = forget_bias
    self._activation = activation
    self._normalize = normalize
    self._peephole = peephole
    if data_format == 'channels_last':
        self._size = tf.TensorShape(shape + [self._filters])
        self._feature_axis = self._size.ndims
        self._data_format = None
    elif data_format == 'channels_first':
        self._size = tf.TensorShape([self._filters] + shape)
        self._feature_axis = 0
        self._data_format = 'NC'
    else:
        raise ValueError('Unknown data_format')

  @property
  def state_size(self):
    return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

  @property
  def output_size(self):
    return self._size

  def call(self, x, state):

    c, h = state

    x, f, u = tf.split(x, [self._filters, 1, 1], axis=-1)

    x = tf.concat([x, h], axis=self._feature_axis)
    n = x.shape[-1].value

    # m = 4 * self._filters if self._filters > 1 else 4
    m = 2 * self._filters if self._filters > 1 else 2

    W = tf.get_variable('kernel', self._kernel + [n, m])
    y = tf.nn.convolution(x, W, 'SAME', data_format=self._data_format)
    if not self._normalize:
      y += tf.get_variable('bias', [m], initializer=tf.zeros_initializer())

    # j, i, f, o = tf.split(y, 4, axis=self._feature_axis)

    j, o = tf.split(y, 2, axis=self._feature_axis)

    #if self._peephole:
      # i += tf.get_variable('W_ci', c.shape[1:]) * c
      # f += tf.get_variable('W_cf', c.shape[1:]) * c

    if self._normalize:
      j = tf.contrib.layers.layer_norm(j)
      # i = tf.contrib.layers.layer_norm(i)
      # f = tf.contrib.layers.layer_norm(f)

    # f = tf.sigmoid(f + self._forget_bias)
    # i = tf.sigmoid(i)
    # c = c * f + i * self._activation(j)

    c = c * f + u * self._activation(j)

    if self._peephole:
      o += tf.get_variable('W_co', c.shape[1:]) * c

    if self._normalize:
      o = tf.contrib.layers.layer_norm(o)
      c = tf.contrib.layers.layer_norm(c)

    o = tf.sigmoid(o)
    h = o * self._activation(c)

    state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

    return h, state


