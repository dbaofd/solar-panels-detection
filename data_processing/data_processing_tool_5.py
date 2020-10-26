from PIL import Image
import matplotlib.pyplot as plt
from random import randrange
import numpy as np

IMAGE_PATH = '../dataset/original_image/test_1_Perth.png'
IMAGE_SAVING_PATH = '../dataset/new_training_image/'
LABEL_PATH = '../dataset/original_label/test_1_Perth.png'
LABEL_SAVING_PATH = '../dataset/new_training_label/'
SAVING_INDEX = 1
CROP_NUM=100


def random_crop(img, mask):
    """
    Randomly, crop image and mask
    Args:
        img: plt image list.
        mask: mask list.
    Return:
        img_list: cropped image list.
        label_list: cropped mask list.
    Raises:
        None.
    """
    if str(img.dtype) != 'uint8':
        img = (img * 255).astype(np.uint8)
    if str(mask.dtype) != 'uint8':
        mask = (mask * 255).astype(np.uint8)
    img = Image.fromarray(img)
    mask = Image.fromarray(mask)
    x, y = img.size
    matrix = 256
    img_list = []
    label_list = []
    for i in range(CROP_NUM):
        x1 = randrange(0, x - matrix)
        y1 = randrange(0, y - matrix)
        img_list.append(img.crop((x1, y1, x1 + matrix, y1 + matrix)))
        label_list.append(mask.crop((x1, y1, x1 + matrix, y1 + matrix)))

    return img_list, label_list


def save_images(img_list, img_saving_path, label_list, label_saving_path):
    """
    Save cropped images and masks.
    Args:
        img_list: list, cropped images.
        img_saving_path: string, image saving path.
        label_list: list, cropped masks.
        label_saving_path: string, mask saving path.
    Return:
        None.
    Raises:
        None.
    """
    img_index = SAVING_INDEX
    label_index=SAVING_INDEX
    for img in img_list:
        img.save(img_saving_path + str(img_index) + '.png', 'PNG')
        img_index+=1
    for label in label_list:
        label.save(label_saving_path + str(label_index) + '.png', 'PNG')
        label_index += 1


if __name__ == '__main__':
    image = plt.imread(IMAGE_PATH)
    label = plt.imread(LABEL_PATH)
    image_list, mask_list = random_crop(image, label)
    save_images(image_list, IMAGE_SAVING_PATH, mask_list, LABEL_SAVING_PATH)
    print("Save successfully!")
