from tensorflow.keras import Model, Input
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import sys

sys.path.append("..")
from pooling_layers.max_pooling import MaxPoolingWithArgmax2D
from pooling_layers.max_unpooling import MaxUnpooling2D
from metrics.intersection_over_union import iou


def segnet_original(input_shape, batch_size, n_labels=2, kernel=3, pool_size=(2, 2), output_mode="softmax",
                    model_summary=None):
    # encoder
    inputs = Input(shape=input_shape, batch_size=batch_size)

    conv_1 = layers.Conv2D(64, (kernel, kernel), padding="same", kernel_initializer='he_normal')(inputs)
    conv_1 = layers.BatchNormalization()(conv_1)
    conv_1 = layers.Activation("relu")(conv_1)
    conv_2 = layers.Conv2D(64, (kernel, kernel), padding="same", kernel_initializer='he_normal')(conv_1)
    conv_2 = layers.BatchNormalization()(conv_2)
    conv_2 = layers.Activation("relu")(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = layers.Conv2D(128, (kernel, kernel), padding="same", kernel_initializer='he_normal')(pool_1)
    conv_3 = layers.BatchNormalization()(conv_3)
    conv_3 = layers.Activation("relu")(conv_3)
    conv_4 = layers.Conv2D(128, (kernel, kernel), padding="same", kernel_initializer='he_normal')(conv_3)
    conv_4 = layers.BatchNormalization()(conv_4)
    conv_4 = layers.Activation("relu")(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = layers.Conv2D(128, (kernel, kernel), padding="same", kernel_initializer='he_normal')(pool_2)
    conv_5 = layers.BatchNormalization()(conv_5)
    conv_5 = layers.Activation("relu")(conv_5)
    conv_6 = layers.Conv2D(128, (kernel, kernel), padding="same", kernel_initializer='he_normal')(conv_5)
    conv_6 = layers.BatchNormalization()(conv_6)
    conv_6 = layers.Activation("relu")(conv_6)
    conv_7 = layers.Conv2D(128, (kernel, kernel), padding="same", kernel_initializer='he_normal')(conv_6)
    conv_7 = layers.BatchNormalization()(conv_7)
    conv_7 = layers.Activation("relu")(conv_7)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = layers.Conv2D(256, (kernel, kernel), padding="same", kernel_initializer='he_normal')(pool_3)
    conv_8 = layers.BatchNormalization()(conv_8)
    conv_8 = layers.Activation("relu")(conv_8)
    conv_9 = layers.Conv2D(256, (kernel, kernel), padding="same", kernel_initializer='he_normal')(conv_8)
    conv_9 = layers.BatchNormalization()(conv_9)
    conv_9 = layers.Activation("relu")(conv_9)
    conv_10 = layers.Conv2D(256, (kernel, kernel), padding="same", kernel_initializer='he_normal')(conv_9)
    conv_10 = layers.BatchNormalization()(conv_10)
    conv_10 = layers.Activation("relu")(conv_10)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = layers.Conv2D(256, (kernel, kernel), padding="same", kernel_initializer='he_normal')(pool_4)
    conv_11 = layers.BatchNormalization()(conv_11)
    conv_11 = layers.Activation("relu")(conv_11)
    conv_12 = layers.Conv2D(256, (kernel, kernel), padding="same", kernel_initializer='he_normal')(conv_11)
    conv_12 = layers.BatchNormalization()(conv_12)
    conv_12 = layers.Activation("relu")(conv_12)
    conv_13 = layers.Conv2D(256, (kernel, kernel), padding="same", kernel_initializer='he_normal')(conv_12)
    conv_13 = layers.BatchNormalization()(conv_13)
    conv_13 = layers.Activation("relu")(conv_13)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    print("Build enceder done..")

    # decoder
    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    conv_14 = layers.Conv2D(256, (kernel, kernel), padding="same", kernel_initializer='he_normal')(unpool_1)
    conv_14 = layers.BatchNormalization()(conv_14)
    conv_14 = layers.Activation("relu")(conv_14)
    conv_15 = layers.Conv2D(256, (kernel, kernel), padding="same", kernel_initializer='he_normal')(conv_14)
    conv_15 = layers.BatchNormalization()(conv_15)
    conv_15 = layers.Activation("relu")(conv_15)
    conv_16 = layers.Conv2D(256, (kernel, kernel), padding="same", kernel_initializer='he_normal')(conv_15)
    conv_16 = layers.BatchNormalization()(conv_16)
    conv_16 = layers.Activation("relu")(conv_16)

    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

    conv_17 = layers.Conv2D(256, (kernel, kernel), padding="same", kernel_initializer='he_normal')(unpool_2)
    conv_17 = layers.BatchNormalization()(conv_17)
    conv_17 = layers.Activation("relu")(conv_17)
    conv_18 = layers.Conv2D(256, (kernel, kernel), padding="same", kernel_initializer='he_normal')(conv_17)
    conv_18 = layers.BatchNormalization()(conv_18)
    conv_18 = layers.Activation("relu")(conv_18)
    conv_19 = layers.Conv2D(256, (kernel, kernel), padding="same", kernel_initializer='he_normal')(conv_18)
    conv_19 = layers.BatchNormalization()(conv_19)
    conv_19 = layers.Activation("relu")(conv_19)
    # reduce the number of feature maps to 128,
    # since mask_3 has 128 feature maps
    conv_19 = layers.Conv2D(128, (1, 1))(conv_19)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = layers.Conv2D(128, (kernel, kernel), padding="same", kernel_initializer='he_normal')(unpool_3)
    conv_20 = layers.BatchNormalization()(conv_20)
    conv_20 = layers.Activation("relu")(conv_20)
    conv_21 = layers.Conv2D(128, (kernel, kernel), padding="same", kernel_initializer='he_normal')(conv_20)
    conv_21 = layers.BatchNormalization()(conv_21)
    conv_21 = layers.Activation("relu")(conv_21)
    conv_22 = layers.Conv2D(128, (kernel, kernel), padding="same", kernel_initializer='he_normal')(conv_21)
    conv_22 = layers.BatchNormalization()(conv_22)
    conv_22 = layers.Activation("relu")(conv_22)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = layers.Conv2D(128, (kernel, kernel), padding="same", kernel_initializer='he_normal')(unpool_4)
    conv_23 = layers.BatchNormalization()(conv_23)
    conv_23 = layers.Activation("relu")(conv_23)
    conv_24 = layers.Conv2D(128, (kernel, kernel), padding="same", kernel_initializer='he_normal')(conv_23)
    conv_24 = layers.BatchNormalization()(conv_24)
    conv_24 = layers.Activation("relu")(conv_24)
    # reduce the number of feature maps to 64,
    # since mask_1 has 64 feature maps
    conv_24 = layers.Conv2D(64, (1, 1))(conv_24)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = layers.Conv2D(64, (kernel, kernel), padding="same", kernel_initializer='he_normal')(unpool_5)
    conv_25 = layers.BatchNormalization()(conv_25)
    conv_25 = layers.Activation("relu")(conv_25)
    conv_26 = layers.Conv2D(64, (kernel, kernel), padding="same", kernel_initializer='he_normal')(conv_25)
    conv_26 = layers.BatchNormalization()(conv_26)
    conv_26 = layers.Activation("relu")(conv_26)

    out = layers.Conv2D(n_labels, 1, activation=output_mode)(conv_26)
    print("Build decoder done..")

    model = Model(inputs=inputs, outputs=out, name="SegNet")
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=[iou, 'accuracy'])

    if model_summary is True:
        model.summary()

    return model
