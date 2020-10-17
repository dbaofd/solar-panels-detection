from PIL import Image
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAIN_IMAGE_PATH = 'data_set/training/image/'
TRAIN_LABEL_PATH = 'data_set/training/label/'
VALIDATION_IMAGE_PATH = 'data_set/validation/image/'
VALIDATION_LABEL_PATH = 'data_set/validation/label/'
TEST_IMAGE_PATH = 'data_set/images_without_label/image/'
TEST_LABEL_PATH = 'data_set/images_without_label/label/'
IMAGES_WITHOUT_LABEL_PATH = 'data_set/images_without_label/'
id2code = {0: (0, 0, 0), 1: (128, 0, 0)}


def rgb_to_onehot(rgb_image, colormap=id2code):
    """
    Function to one hot encode RGB mask labels.
    Args:
        rgb_image: image matrix (eg. 256 x 256 x 3 dimension numpy ndarray).
        colormap: dictionary of color to label id.
    Return:
        encoded_image: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap).
    Raises:
        None.
    """
    num_classes = len(colormap)
    shape = rgb_image.shape[:2] + (num_classes,)
    encoded_image = np.zeros(shape, dtype=np.int8)
    for i, cls in enumerate(colormap):
        encoded_image[:, :, i] = np.all(rgb_image.reshape((-1, 3)) == colormap[i], axis=1).reshape(shape[:2])
    return encoded_image


def onehot_to_rgb(onehot, colormap=id2code):
    """
    Function to decode encoded mask labels.
    Args:
        onehot: one hot encoded image matrix (height x width x num_classes).
        colormap: dictionary of color to label id.
    Return:
        Decoded RGB image (height x width x 3).
    Raises:
        None.
    """
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:2] + (3,))
    for k in colormap.keys():
        output[single_layer == k] = colormap[k]
    return np.uint8(output)


# Define the generator
# Normalizing only frame images, since masks contain label info
data_gen_args = dict(rescale=1. / 255)
mask_gen_args = dict()

train_frames_datagen = ImageDataGenerator(**data_gen_args)
train_masks_datagen = ImageDataGenerator(**mask_gen_args)
val_frames_datagen = ImageDataGenerator(**data_gen_args)
val_masks_datagen = ImageDataGenerator(**mask_gen_args)
test_frames_datagen = ImageDataGenerator(**data_gen_args)
test_masks_datagen = ImageDataGenerator(**mask_gen_args)
images_without_label_datagen = ImageDataGenerator(**data_gen_args)
# Seed defined for aligning images and their masks
seed = 1


def train_data_generator(seed=1, batch_size=8, target_size=(256, 256)):
    """
    training image data generator.
    Args:
        seed: seed provided to the flow_from_directory function to ensure aligned data flow.
        batch_size: number of images to import at a time.
        target_size: target image size.
    Return:
        Decoded RGB image (height x width x 3), mask.
    Raises:
        None.
    """
    train_image_generator = train_frames_datagen.flow_from_directory(
        TRAIN_IMAGE_PATH,
        batch_size=batch_size, seed=seed, target_size=target_size, color_mode='rgb')
    train_mask_generator = train_masks_datagen.flow_from_directory(
        TRAIN_LABEL_PATH,
        batch_size=batch_size, seed=seed, target_size=target_size, color_mode='rgb')
    while True:
        X1i = train_image_generator.next()
        X2i = train_mask_generator.next()
        # One hot encoding RGB images
        mask_encoded = [rgb_to_onehot(X2i[0][x, :, :, :], id2code) for x in range(X2i[0].shape[0])]

        yield X1i[0], np.asarray(mask_encoded)


def validation_data_generator(seed=1, batch_size=8, target_size=(256, 256)):
    """
    Validation image data generator.
    Args:
        seed: seed provided to the flow_from_directory function to ensure aligned data flow.
        batch_size: number of images to import at a time.
        target_size: target image size.
    Return:
        Decoded RGB image (height x width x 3), mask.
    Raises:
        None.
    """
    val_image_generator = val_frames_datagen.flow_from_directory(
        VALIDATION_IMAGE_PATH,
        batch_size=batch_size, seed=seed, target_size=target_size)

    val_mask_generator = val_masks_datagen.flow_from_directory(
        VALIDATION_LABEL_PATH,
        batch_size=batch_size, seed=seed, target_size=target_size)

    while True:
        X1i = val_image_generator.next()
        X2i = val_mask_generator.next()

        # One hot encoding RGB images
        mask_encoded = [rgb_to_onehot(X2i[0][x, :, :, :], id2code) for x in range(X2i[0].shape[0])]

        yield X1i[0], np.asarray(mask_encoded)


def test_data_generator(seed=1, batch_size=8, target_size=(256, 256)):
    """
    Test image data generator.
    Args:
        seed: seed provided to the flow_from_directory function to ensure aligned data flow.
        batch_size: number of images to import at a time.
        target_size: target image size.
    Return:
        Decoded RGB image (height x width x 3), mask.
    Raises:
        None.
    """
    test_image_generator = test_frames_datagen.flow_from_directory(
        TEST_IMAGE_PATH,
        batch_size=batch_size, seed=seed, target_size=target_size)

    test_mask_generator = test_masks_datagen.flow_from_directory(
        TEST_LABEL_PATH,
        batch_size=batch_size, seed=seed, target_size=target_size)

    while True:
        X1i = test_image_generator.next()
        X2i = test_mask_generator.next()

        # One hot encoding RGB images
        mask_encoded = [rgb_to_onehot(X2i[0][x, :, :, :], id2code) for x in range(X2i[0].shape[0])]

        yield X1i[0], np.asarray(mask_encoded)


def images_without_label_data_generator(seed=1, batch_size=8, target_size=(256, 256)):
    """
    Images without label data generator.
    Args:
        seed: seed provided to the flow_from_directory function to ensure aligned data flow.
        batch_size: number of images to import at a time.
        target_size: target image size.
    Return:
        Decoded RGB image (height x width x 3).
    Raises:
        None.
    """
    images_without_label_generator = images_without_label_datagen.flow_from_directory(
        IMAGES_WITHOUT_LABEL_PATH,
        batch_size=batch_size, seed=seed, target_size=target_size)

    while True:
        X1i = images_without_label_generator.next()
        yield X1i[0]
