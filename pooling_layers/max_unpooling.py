import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend


class MaxUnpooling2D(layers.Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with tf.compat.v1.variable_scope(self.name):
            mask = backend.cast(mask, 'int32')
            input_shape = tf.shape(updates, out_type='int32')
            #  calculation new shape
            if output_shape is None:
                output_shape = (
                    input_shape[0], input_shape[1] * self.size[0], input_shape[2] * self.size[1], input_shape[3])
                self.output_shape1 = output_shape

        # calculation indices for batch, height, width and feature maps
        one_like_mask = backend.ones_like(mask, dtype='int32')
        batch_shape = backend.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
        batch_range = backend.reshape(tf.range(output_shape[0], dtype='int32'), shape=batch_shape)
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2]
        feature_range = tf.range(output_shape[3], dtype='int32')
        f = one_like_mask * feature_range

        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(updates)
        indices = backend.transpose(backend.reshape(backend.stack([b, y, x, f]), [4, updates_size]))
        values = backend.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return mask_shape[0], mask_shape[1] * self.size[0], mask_shape[2] * self.size[1], mask_shape[3]
