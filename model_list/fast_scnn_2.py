import tensorflow as tf
from tensorflow.keras import optimizers
import sys

sys.path.append("../..")
from metrics.intersection_over_union import iou


def conv_block(inputs, conv_type, kernel, kernel_size, strides, padding='same', relu=True):
    if (conv_type == 'ds'):
        x = tf.keras.layers.SeparableConv2D(kernel, kernel_size, padding=padding, strides=strides)(inputs)
    else:
        x = tf.keras.layers.Conv2D(kernel, kernel_size, padding=padding, strides=strides)(inputs)

    x = tf.keras.layers.BatchNormalization()(x)

    if (relu):
        x = tf.keras.activations.relu(x)

    return x


def _res_bottleneck(inputs, filters, kernel, t, s, r=False):
    tchannel = tf.keras.backend.int_shape(inputs)[-1] * t

    x = conv_block(inputs, 'conv', tchannel, (1, 1), strides=(1, 1))

    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    x = conv_block(x, 'conv', filters, (1, 1), strides=(1, 1), padding='same', relu=False)

    if r:
        x = tf.keras.layers.add([x, inputs])
    return x


def bottleneck_block(inputs, filters, kernel, t, strides, n):
    x = _res_bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _res_bottleneck(x, filters, kernel, t, 1, True)

    return x


def pyramid_pooling_block(input_tensor, bin_sizes):
    concat_list = [input_tensor]
    #   w = 64
    #   h = 32
    # Change the width and height when the previous layer's output tensor is not 32 x 32 x filters
    w = 32
    h = 32

    for bin_size in bin_sizes:
        x = tf.keras.layers.AveragePooling2D(pool_size=(w // bin_size, h // bin_size),
                                             strides=(w // bin_size, h // bin_size))(input_tensor)
        x = tf.keras.layers.Conv2D(128, 3, 2, padding='same')(x)
        x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (w, h)))(x)
        concat_list.append(x)

    return tf.keras.layers.concatenate(concat_list)


def fast_scnn_v2(input_shape, batch_size, n_labels, model_summary=False):
    # Step 1: Learning to DownSample
    # Input Layer
    input_layer = tf.keras.layers.Input(shape=input_shape, name='input_layer', batch_size=batch_size)
    lds_layer = conv_block(input_layer, 'conv', 32, (3, 3), strides=(2, 2))
    lds_layer = conv_block(lds_layer, 'conv', 64, (3, 3), strides=(1, 1))
    lds_layer = conv_block(lds_layer, 'conv', 96, (3, 3), strides=(1, 1))
    # Step 2: Global Feature Extractor
    gfe_layer = bottleneck_block(lds_layer, 64, (3, 3), t=6, strides=2, n=3)
    gfe_layer = bottleneck_block(gfe_layer, 96, (3, 3), t=6, strides=2, n=3)
    gfe_layer = bottleneck_block(gfe_layer, 128, (3, 3), t=6, strides=1, n=3)
    gfe_layer = pyramid_pooling_block(gfe_layer, [2, 4, 6, 8])
    # Step 3: Feature Fusion
    ff_layer1 = conv_block(lds_layer, 'conv', 128, (1, 1), padding='same', strides=(1, 1), relu=False)

    ff_layer2 = tf.keras.layers.UpSampling2D((4, 4))(gfe_layer)
    ff_layer2 = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), depth_multiplier=1, padding='same')(ff_layer2)
    ff_layer2 = tf.keras.layers.BatchNormalization()(ff_layer2)
    ff_layer2 = tf.keras.activations.relu(ff_layer2)
    ff_layer2 = tf.keras.layers.Conv2D(128, 1, 1, padding='same', activation=None)(ff_layer2)

    ff_final = tf.keras.layers.add([ff_layer1, ff_layer2])
    ff_final = tf.keras.layers.BatchNormalization()(ff_final)
    ff_final = tf.keras.activations.relu(ff_final)
    # Step 4: Classifier
    classifier = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', strides=(1, 1),
                                                 name='DSConv1_classifier')(ff_final)
    classifier = tf.keras.layers.BatchNormalization()(classifier)
    classifier = tf.keras.activations.relu(classifier)

    classifier = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', strides=(1, 1),
                                                 name='DSConv2_classifier')(classifier)
    classifier = tf.keras.layers.BatchNormalization()(classifier)
    classifier = tf.keras.activations.relu(classifier)

    num_of_classes = n_labels
    classifier = conv_block(classifier, 'conv', num_of_classes, (1, 1), strides=(1, 1), padding='same', relu=False)

    classifier = tf.keras.layers.Dropout(0.3)(classifier)

    classifier = tf.keras.layers.UpSampling2D((2, 2))(classifier)
    classifier = tf.keras.activations.softmax(classifier)
    # Model Compilation
    fast_scnn = tf.keras.Model(inputs=input_layer, outputs=classifier, name='Fast_SCNN')
    fast_scnn.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=[iou, 'accuracy'])
    if model_summary is True:
        fast_scnn.summary()

    return fast_scnn
