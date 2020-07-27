import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

"""
Summary of this file here.
This tool is used to crop one big pictures into many small pictures.
For example, we have a 4000x4000 picture, crop it into many 256x256 pictures.
Save these images.
"""
IMAGE_PATH = '../dataset/original_image/test_4.jpg'
IMAGE_SAVING_PATH = '../dataset/new_training_image/'
LABEL_PATH = '../dataset/original_label/training_6_fairfield_2.png'
LABEL_SAVING_PATH = '../dataset/new_training_label/'
CROP_WIDTH = 256
CROP_HEIGHT = 256
SAVING_INDEX = 1


def crop_image(img):
    """
    Crop image into CROP_WIDTH by CROP_WIDTH picture.
    Args:
        img: numpy array.
    Return:
        cropped_image: cropped image.
    Raises:
        Wrong type exception: if img is not float32 format, bug may occur.
    """
    if str(img.dtype) != 'uint8':
        img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    width, height = img.size
    column_number = int(width / CROP_WIDTH)
    row_number = int(height / CROP_HEIGHT)
    box_list = []
    for i in range(0, row_number):
        for j in range(0, column_number):
            box = (
                j * CROP_WIDTH, i * CROP_HEIGHT, (j + 1) * CROP_WIDTH,
                (i + 1) * CROP_HEIGHT)  # (left, top, right, bottom)
            box_list.append(box)
    img_list = [img.crop(box) for box in box_list]
    return img_list


def save_images(img_list, saving_path):
    """
    Crop image into CROP_WIDTH by CROP_WIDTH picture.
    Args:
        img_list: PIL.Image.Image list.
        saving_path: String, saving path.
    Return:
        None.
    Raises:
        None.
    """
    index = SAVING_INDEX
    for img in img_list:
        img.save(saving_path + str(index) + '.png', 'PNG')
        index += 1


if __name__ == '__main__':
    image = plt.imread(IMAGE_PATH)
    image_list = crop_image(image)
    save_images(image_list, IMAGE_SAVING_PATH)
    label = plt.imread(LABEL_PATH)
    label_list = crop_image(label)
    save_images(label_list, LABEL_SAVING_PATH)
    print("Save successfully!")
