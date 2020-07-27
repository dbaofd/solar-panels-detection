from segnet import segnet
from data_processing import prepare_data
from data_processing import data_processing_tool_4
import matplotlib.pyplot as plt
import math

TEST_IMAGE_PATH = "299.jpg"
MODEL_PATH = "../segnet/models/vgg16_segnet_3.h5"
INITIAL_BATCH_SIZE = 8


def get_predicted_label_list(sub_imgs):
    model = segnet.SegNet(input_shape=sub_imgs[0].shape, batch=INITIAL_BATCH_SIZE, n_labels=2, model_summary=False)
    model.load_weights(MODEL_PATH)
    sub_predicted_label_list = []
    for i in range(0, math.ceil(sub_imgs.shape[0] / INITIAL_BATCH_SIZE)):
        # number of sub images is not the times of 8, the batch size will be reset
        # for the last batch
        if sub_imgs.shape[0] % INITIAL_BATCH_SIZE != 0:
            if i != math.ceil(sub_imgs.shape[0] / INITIAL_BATCH_SIZE) - 1:
                batch_size = INITIAL_BATCH_SIZE
            else:
                batch_size = sub_imgs.shape[0] % INITIAL_BATCH_SIZE
                model = segnet.SegNet(input_shape=sub_imgs[0].shape, batch=batch_size, n_labels=2, model_summary=False)
                model.load_weights(MODEL_PATH)
        else:
            # batch size is set to 8 all the time
            batch_size = INITIAL_BATCH_SIZE
        results = model.predict(sub_imgs[i * INITIAL_BATCH_SIZE:i * INITIAL_BATCH_SIZE + batch_size])
        for j in range(0, results.shape[0]):
            my_img = prepare_data.onehot_to_rgb(results[j], prepare_data.id2code)
            sub_predicted_label_list.append(my_img)
    return sub_predicted_label_list


if __name__ == '__main__':
    image = plt.imread(TEST_IMAGE_PATH)
    original_width = image.shape[1]
    original_height = image.shape[0]
    sub_imgs, padded_img, padded_width, padded_height = data_processing_tool_4.get_sub_images(image)
    sub_predicted_label_list = get_predicted_label_list(sub_imgs)
    full_label = data_processing_tool_4.get_full_predicted_label(padded_height, padded_width, sub_predicted_label_list)
    full_label_with_mask = data_processing_tool_4.add_transparent_mask(padded_img, full_label, original_width,
                                                                       original_height)
    full_label_with_mask.save('test.png', 'PNG')
