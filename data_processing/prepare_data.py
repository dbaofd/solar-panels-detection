from PIL import Image
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def loadData():
    img=np.zeros((64, 512, 512, 3))
    label = np.zeros((64, 512, 512, 3))
    for index in range(1,65):
        pathImg=r"dataset/Training/Image/"+str(index)+".png"
        pathLabel = r"dataset/Training/Label/" + str(index) + ".png"
        im1 = Image.open(pathImg)
        imgArray = np.array(im1)
        imgArray=imgArray/255.0
        img[index-1]=imgArray

        im2 = Image.open(pathLabel)
        labelArray = np.array(im2)
        labelArray = labelArray / 255.0
        label[index - 1] = labelArray
    return img, label


def _read_to_tensor(fname, output_height=256, output_width=256, normalize_data=False):
    '''Function to read images from given image file path, and provide resized images as tensors
        Inputs:
            fname - image file path
            output_height - required output image height
            output_width - required output image width
            normalize_data - if True, normalize data to be centered around 0 (mean 0, range 0 to 1)
        Output: Processed image tensors
    '''

    # Read the image as a tensor
    img_strings = tf.io.read_file(fname)
    imgs_decoded = tf.image.decode_png(img_strings)

    # Resize the image
    output = tf.image.resize(imgs_decoded, [output_height, output_width])

    # Normalize if required
    if normalize_data:
        output = (output - 128) / 128
    return output


def read_images():
    '''Function to get all image directories, read images and masks in separate tensors
        Outputs
            frame_tensors, masks_tensors, frame files list, mask files list
    '''
    train_label_path = 'dataset/Training/Label/'
    train_frame_path = 'dataset/Training/Image/'
    frames_list = os.listdir(train_frame_path)
    masks_list = os.listdir(train_label_path)

    frames_list.sort()
    masks_list.sort()

    print('{} frame files found in the provided directory.'.format(len(frames_list)))
    print('{} mask files found in the provided directory.'.format(len(masks_list)))

    # Create file paths from file names
    frames_paths = [os.path.join(train_frame_path, fname) for fname in frames_list]
    masks_paths = [os.path.join(train_label_path, fname) for fname in masks_list]
    # Create dataset of tensors
    frame_data = tf.data.Dataset.from_tensor_slices(frames_paths)
    masks_data = tf.data.Dataset.from_tensor_slices(masks_paths)

    # Read images into the tensor dataset
    frame_tensors = frame_data.map(_read_to_tensor)
    masks_tensors = masks_data.map(_read_to_tensor)

    print('Completed importing {} frame images from the provided directory.'.format(len(frames_list)))
    print('Completed importing {} mask images from the provided directory.'.format(len(masks_list)))

    return frame_tensors, masks_tensors, frames_list, masks_list

id2code={0:(0, 0, 0), 1:(128, 0, 0)}

def rgb_to_onehot(rgb_image, colormap = id2code):
    '''Function to one hot encode RGB mask labels
        Inputs:
            rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]+(num_classes,)
    encoded_image = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(colormap):
        encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])
    return encoded_image

def onehot_to_rgb(onehot, colormap = id2code):
    '''Function to decode encoded mask labels
        Inputs:
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3)
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)

#Define the generator
#Normalizing only frame images, since masks contain label info
data_gen_args = dict(rescale=1./255)
mask_gen_args = dict()

train_frames_datagen = ImageDataGenerator(**data_gen_args)
train_masks_datagen = ImageDataGenerator(**mask_gen_args)
val_frames_datagen = ImageDataGenerator(**data_gen_args)
val_masks_datagen = ImageDataGenerator(**mask_gen_args)

# Seed defined for aligning images and their masks
seed = 1

def TrainAugmentGenerator(seed=1, batch_size=8):
    '''Test Image data generator
        Inputs:
            seed - seed provided to the flow_from_directory function to ensure aligned data flow
            batch_size - number of images to import at a time
        Output: Decoded RGB image (height x width x 3)
    '''
    train_image_generator = train_frames_datagen.flow_from_directory(
        'dataset/Training/Image/',
        batch_size=batch_size, seed=seed, target_size=(256, 256), color_mode='rgb')
    train_mask_generator = train_masks_datagen.flow_from_directory(
        'dataset/Training/Label/',
        batch_size=batch_size, seed=seed, target_size=(256, 256), color_mode='rgb')
    while True:
        X1i = train_image_generator.next()
        X2i = train_mask_generator.next()
        # One hot encoding RGB images
        mask_encoded = [rgb_to_onehot(X2i[0][x, :, :, :], id2code) for x in range(X2i[0].shape[0])]

        yield X1i[0], np.asarray(mask_encoded)


def ValAugmentGenerator(seed=1, batch_size=8):
    '''Validation Image data generator
        Inputs:
            seed - seed provided to the flow_from_directory function to ensure aligned data flow
            batch_size - number of images to import at a time
        Output: Decoded RGB image (height x width x 3)
    '''
    val_image_generator = val_frames_datagen.flow_from_directory(
        'dataset/Testing/Image/',
        batch_size=batch_size, seed=seed, target_size=(256, 256))

    val_mask_generator = val_masks_datagen.flow_from_directory(
        'dataset/Testing/Label/',
        batch_size=batch_size, seed=seed, target_size=(256, 256))

    while True:
        X1i = val_image_generator.next()
        X2i = val_mask_generator.next()

        # One hot encoding RGB images
        mask_encoded = [rgb_to_onehot(X2i[0][x, :, :, :], id2code) for x in range(X2i[0].shape[0])]
        yield X1i[0], np.asarray(mask_encoded)

