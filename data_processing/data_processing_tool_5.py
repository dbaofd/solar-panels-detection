from PIL import Image
import matplotlib.pyplot as plt
from random import randrange
import numpy as np

IMAGE_PATH = '../dataset/original_image/test_1_Perth.png'
IMAGE_SAVING_PATH = '../dataset/new_training_image/'
LABEL_PATH = '../dataset/original_label/testing_1_Perth.png'
LABEL_SAVING_PATH = '../dataset/new_training_label/'
SAVING_INDEX = 1
CROP_NUM=1000


def random_crop(img, mask):
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
    Crop image into CROP_WIDTH by CROP_WIDTH picture.
    Args:
        img_list: PIL.image.image list.
        saving_path: String, saving path.
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