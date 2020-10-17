from PIL import Image
import cv2 as cv
import numpy as np

"""
Summary of the file.
This tool is used by segnet_application. It contains many functions which 
are used to process image. 
"""
MODEL_INPUT_SIZE = 256


def get_sub_images(image):
    """
    Get sub images of one giant image.
    Args:
        image: numpy array.
    Return:
        image_list: numpy array, sub images list.
        gray_rgb_img: PIL.image, gray scale rgb image of the giant image.
        width: padded image width.
        height: padded image height.
    Raises:
        None.
    """
    image, image_array = pad_image(image)
    gray_rgb_img = get_gray_rgb_img(image_array)
    width, height = image.size
    box_list = []
    for i in range(0, int(height / MODEL_INPUT_SIZE)):
        for j in range(0, int(width / MODEL_INPUT_SIZE)):
            box = (
                j * MODEL_INPUT_SIZE, i * MODEL_INPUT_SIZE, (j + 1) * MODEL_INPUT_SIZE,
                (i + 1) * MODEL_INPUT_SIZE)  # (left, top, right, bottom)
            box_list.append(box)

    image_list = [image.crop(box) for box in box_list]
    image_array = np.zeros((len(image_list), MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3))
    for i in range(0, len(image_array)):
        # convert image to array
        image_array[i] = normalize_image(np.array(image_list[i]))
    return image_array, gray_rgb_img, width, height


def pad_image(image_array):
    """
    Pad arbitrary size image. Make the height and width be times
    of model input size.
    Args:
        image_array: numpy array, input image.
    Returns:
        img: PIL.image, padded image.
        image_array: numpy array, padded image
    Raises:
        Can only convert float32 image into unit8 image.
    """
    height = image_array.shape[0]
    width = image_array.shape[1]
    if width % MODEL_INPUT_SIZE != 0:
        width_padding_number = MODEL_INPUT_SIZE - width % MODEL_INPUT_SIZE
    else:
        width_padding_number = 0

    if height % MODEL_INPUT_SIZE != 0:
        height_padding_number = MODEL_INPUT_SIZE - height % MODEL_INPUT_SIZE
    else:
        height_padding_number = 0

    if str(image_array.dtype) != 'uint8':
        # Convert float to uint(0-255)
        image_array = cv.copyMakeBorder((image_array * 255).astype(np.uint8), 0, height_padding_number, 0,
                                        width_padding_number, cv.BORDER_CONSTANT, value=0)
    else:
        image_array = cv.copyMakeBorder(image_array, 0, height_padding_number, 0, width_padding_number,
                                        cv.BORDER_CONSTANT, value=0)
    img = Image.fromarray(image_array)

    return img, image_array


def normalize_image(image_array):
    """
    Normalize image, then feed the model to process.
    Arg:
        image_array: numpy array, input image.
    Return:
        img: numpy array, normalized image.
    Raises:
        None.
    """
    # Convert from integers to floats, this is necessary,
    # because ImageDataGenerator does the same thing before feeding the image
    # to the model.
    img = image_array.astype('float32')
    img = img * (1. / 255)  # Manually normalize image.
    return img


def restore_image_size(image, original_width, original_height):
    """
    Restore padded image into its original size.
    Args:
        image: PIL.image, input image.
        original_width: integer, original width.
        original_height: integer, original height.
    Return:
        restored_image: PIL.image.
    Raises:
        None.
    """
    box_list = []
    box = (0, 0, original_width, original_height)  # (left, top, right, bottom)
    box_list.append(box)
    restored_image = image.crop(box_list[0])
    return restored_image


def get_gray_rgb_img(image):
    """
    Transform image into gray rgb image.
    Arg:
        image: numpy array, input image.
    Return:
        gray_rgb_img: PIL.image.
    Raises:
        None.
    """
    img = Image.fromarray(image).convert('L')  # convert img to grayscale
    img = np.array(img)
    gray_rgb_img = np.stack((img,) * 3, axis=-1)  # convert one channel to three channel RGB
    gray_rgb_img = Image.fromarray(gray_rgb_img)
    return gray_rgb_img


def get_full_predicted_label(padded_height, padded_width, sub_predicted_label_list):
    """
    Joint sub predicted labels to one giant predicted label.
    Args:
        padded_height: integer, height of giant predicted label.
        padded_width: integer, width of giant predicted label.
        sub_predicted_label_list: numpy array, list of sub predicted labels.
    Return:
        full_label: PIL.image, the giant predicted label.
    Raises:
        None.
    """
    row_sub_predicted_label_list = []
    row_num = int(padded_height / MODEL_INPUT_SIZE)
    column_num = int(padded_width / MODEL_INPUT_SIZE)
    for i in range(0, row_num):
        if column_num >= 2:
            row = np.concatenate(
                (sub_predicted_label_list[i * column_num], sub_predicted_label_list[i * column_num + 1]), axis=1)
        else:
            row = sub_predicted_label_list[0]
        for j in range(2, column_num):
            # merge sub predicted label by column
            row = np.concatenate((row, sub_predicted_label_list[i * column_num + j]), axis=1)
        row_sub_predicted_label_list.append(row)
    if len(row_sub_predicted_label_list) >= 2:
        predicted_label = np.concatenate((row_sub_predicted_label_list[0], row_sub_predicted_label_list[1]), axis=0)
    else:
        predicted_label = row_sub_predicted_label_list[0]

    for i in range(2, row_num):
        # merge by row
        predicted_label = np.concatenate((predicted_label, row_sub_predicted_label_list[i]), axis=0)

    full_label = Image.fromarray(predicted_label)
    return full_label


def add_transparent_mask(padded_img, full_label, original_width, original_height):
    """
    Cover full predicted label as transparent mask to original gray scale rgb image.
    Args:
        padded_img: PIL.image, padded gray scale rgb image.
        full_label: PIL.image, full predicted label.
        original_width: integer, original image width.
        original_height: integer, original image width.
    Return:
        img: PIL.image, original image with full predicted label mask.
    Raises:
        None.
    """
    # need to convert to RGBA, otherwise, error occurs
    transparency = 50  # 55%
    if full_label.mode != 'RGBA':
        alpha = Image.new('L', full_label.size, 255)
        full_label.putalpha(alpha)
    paste_mask = full_label.split()[3].point(lambda i: i * transparency / 100.)
    padded_img.paste(full_label, (0, 0), mask=paste_mask)
    img = restore_image_size(padded_img, original_width, original_height)
    return img
