from tensorflow.keras import Model, Input
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import backend

import sys

sys.path.append("../..")
from pooling_layers.max_pooling import MaxPoolingWithArgmax2D
from pooling_layers.max_unpooling import MaxUnpooling2D
from metrics.intersection_over_union import iou


def conv_block(input_tensor, kernel_size, filters, stage, block):
    """Conv block.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # assume all the filters are the same value
    if input_tensor.shape[3] == filters[0]:
        shortcut = input_tensor
    else:
        shortcut = layers.Conv2D(filters[0], (1, 1))(input_tensor)

    x = layers.Conv2D(filters[0], (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)
    for i in range(1, len(filters) - 1):
        x = layers.Conv2D(filters[i], kernel_size,
                          padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2b' + str(i))(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b' + str(i))(x)
        x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters[len(filters) - 1], (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block, next_layer_filter):
    """Identity block
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        next_layer_filter: in segnet previous mask will be added to unpool,
        need to make sure they have same number of filters
    # Returns
        Output tensor for the block.
    """
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    if input_tensor.shape[3] == filters[0]:
        shortcut = input_tensor
    else:
        shortcut = layers.Conv2D(filters[0], (1, 1))(input_tensor)

    x = layers.Conv2D(filters[0], (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    for i in range(1, len(filters) - 1):
        x = layers.Conv2D(filters[i], kernel_size,
                          padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2b' + str(i))(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b' + str(i))(x)
        x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters[len(filters) - 1], (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    if x.shape[3] != next_layer_filter:
        x = layers.Conv2D(next_layer_filter, (1, 1))(x)
    return x


def segnet_resnet_v2(input_shape, batch_size, n_labels=2, kernel=3, pool_size=(2, 2), output_mode="softmax",
                     model_summary=None):
    # encoder
    inputs = Input(shape=input_shape, batch_size=batch_size)
    conv_1 = conv_block(input_tensor=inputs, kernel_size=(kernel, kernel), filters=[64, 64, 64], stage=1, block='a')
    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_1)
    conv_2 = conv_block(input_tensor=pool_1, kernel_size=(kernel, kernel), filters=[128, 128, 128], stage=2, block='a')
    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_2)
    conv_3 = conv_block(input_tensor=pool_2, kernel_size=(kernel, kernel), filters=[128, 128, 128, 128], stage=3,
                        block='a')
    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_3)
    conv_4 = conv_block(input_tensor=pool_3, kernel_size=(kernel, kernel), filters=[256, 256, 256, 256, 256], stage=4,
                        block='a')
    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_4)
    conv_5 = conv_block(input_tensor=pool_4, kernel_size=(kernel, kernel), filters=[256, 256, 256, 256, 256], stage=5,
                        block='a')
    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_5)
    print("Build enceder done..")

    # decoder
    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])
    conv_6 = identity_block(input_tensor=unpool_1, kernel_size=(kernel, kernel), filters=[256, 256, 256, 256, 256],
                            stage=6, block='a', next_layer_filter=256)
    unpool_2 = MaxUnpooling2D(pool_size)([conv_6, mask_4])
    conv_7 = identity_block(input_tensor=unpool_2, kernel_size=(kernel, kernel), filters=[256, 256, 256, 256, 256],
                            stage=7, block='a', next_layer_filter=128)
    unpool_3 = MaxUnpooling2D(pool_size)([conv_7, mask_3])
    conv_8 = identity_block(input_tensor=unpool_3, kernel_size=(kernel, kernel), filters=[128, 128, 128, 128], stage=8,
                            block='a', next_layer_filter=128)
    unpool_4 = MaxUnpooling2D(pool_size)([conv_8, mask_2])
    conv_9 = identity_block(input_tensor=unpool_4, kernel_size=(kernel, kernel), filters=[128, 128, 128], stage=9,
                            block='a', next_layer_filter=64)
    unpool_5 = MaxUnpooling2D(pool_size)([conv_9, mask_1])
    conv_10 = identity_block(input_tensor=unpool_5, kernel_size=(kernel, kernel), filters=[64, 64, 64], stage=10,
                             block='a', next_layer_filter=64)

    out = layers.Conv2D(n_labels, 1, activation=output_mode)(conv_10)
    print("Build decoder done..")

    model = Model(inputs=inputs, outputs=out, name="SegNet")
    model.compile(optimizer=optimizers.Adam(lr=1e-3), loss='categorical_crossentropy', metrics=[iou, 'accuracy'])

    if model_summary is True:
        model.summary()

    return model
