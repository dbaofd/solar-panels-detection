import matplotlib.pyplot as plt
import math
import sys
import datetime
import os

print(os.getcwd())
from model_list import segnet_1
from model_list import segnet_3
from model_list import fast_scnn_2
from model_list import segnet_0

sys.path.append("..")
from data_processing import prepare_data, data_processing_tool_4

IMAGE_LIST = ["EPSG3857_Date20170224_Lat-28.17291_Lon153.541585_Mpp0.149.jpg",
              "EPSG3857_Date20170228_Lat-31.936911_Lon115.814916_Mpp0.149.jpg",
              "EPSG3857_Date20200321_Lat-27.496309_Lon153.012468_Mpp0.149.jpg",
              "EPSG3857_Date20200714_Lat-27.367418_Lon153.054237_Mpp0.149.jpg",
              "47.png"]
TRAINED_MODELS = ["fast_scnn_2.h5",  # model type 1
                  "seg_resnet_2.h5",  # model type 2
                  "segnet_1.h5",  # model type 3
                  "segnet_2.h5",  # model type 3
                  "segnet_original.h5"]  # model type 4
TEST_IMAGE_PATH = "test_images/"
MODEL_PATH = "../trained_models/"
SAVING_PATH = "prediction_images/"
INITIAL_BATCH_SIZE = 8


def get_predicted_label_list(sub_imgs, model):
    """
    Get predicted sub labels.
    Args:
        sub_imgs: np.array.
        model: model.
    Return:
        sub_predicted_label_list: list, predicted sub labels.
    Raises:
        None.
    """
    sub_predicted_label_list = []
    for i in range(0, math.ceil(sub_imgs.shape[0] / INITIAL_BATCH_SIZE)):
        # number of sub images is not the times of 8, the batch size will be reset
        # for the last batch
        if sub_imgs.shape[0] % INITIAL_BATCH_SIZE != 0:
            if i != math.ceil(sub_imgs.shape[0] / INITIAL_BATCH_SIZE) - 1:
                batch_size = INITIAL_BATCH_SIZE
            else:
                batch_size = sub_imgs.shape[0] % INITIAL_BATCH_SIZE
        else:
            # batch size is set to 8 all the time
            batch_size = INITIAL_BATCH_SIZE
        results = model.predict(sub_imgs[i * INITIAL_BATCH_SIZE:i * INITIAL_BATCH_SIZE + batch_size])
        for j in range(0, results.shape[0]):
            my_img = prepare_data.onehot_to_rgb(results[j], prepare_data.id2code)
            sub_predicted_label_list.append(my_img)
    return sub_predicted_label_list


def generate_prediction_image(model_type, model_name, test_image_name, saving_image_name):
    """
    Generate prediction image.
    Args:
        model_type: Integer, indicate model type.
        model_name: h5 file name.
        test_image_name: string, test image name.
        saving_image_name: string, saving image name.
    Return:
        None.
    Raises:
        None.
    """
    image = plt.imread(TEST_IMAGE_PATH + test_image_name)
    original_width = image.shape[1]
    original_height = image.shape[0]
    sub_imgs, padded_img, padded_width, padded_height = data_processing_tool_4.get_sub_images(image)
    # Bulid model and load weight
    # The reason to set model's batch_size to 1 is that only this enables data generator to have a dynamic batch_size.
    # For example, if the model's batch_size is 16, then the batch_size of data generator must be 16.
    # But if the model's batch_size is 1, then the batch_size of data generator can be 1,2,8,16 whatever.
    if model_type == 1:  # fast scnn
        model = fast_scnn_2.fast_scnn_v2(input_shape=sub_imgs[0].shape, batch_size=1, n_labels=2, model_summary=False)
        model.load_weights(MODEL_PATH + model_name)
    elif model_type == 2:  # segnet with resnet
        model = segnet_3.segnet_resnet_v2(input_shape=sub_imgs[0].shape, batch_size=1, n_labels=2, model_summary=False)
        model.load_weights(MODEL_PATH + model_name)
    elif model_type == 3:  # segnet with 4 encoders and decoders
        model = segnet_1.segnet_4_encoder_decoder(input_shape=sub_imgs[0].shape, batch_size=1, n_labels=2,
                                                  model_summary=False)
        model.load_weights(MODEL_PATH + model_name)
    elif model_type == 4:#original segnet
        model = segnet_0.segnet_original(input_shape=sub_imgs[0].shape, batch_size=1, n_labels=2, model_summary=False)
        model.load_weights(MODEL_PATH+model_name)
    else:
        raise ModelTypeError
    sub_predicted_label_list = get_predicted_label_list(sub_imgs, model)
    full_label = data_processing_tool_4.get_full_predicted_label(padded_height, padded_width, sub_predicted_label_list)
    full_label_with_mask = data_processing_tool_4.add_transparent_mask(padded_img, full_label, original_width,
                                                                       original_height)
    full_label_with_mask.save(SAVING_PATH + saving_image_name + '.png', 'PNG')
    print("Save successfully!")


class ModelTypeError(Exception):
    """Raise this error when the model type is invalid."""
    pass


if __name__ == '__main__':
    # choose one of them.
    # model_type=1, model_name=TRAINED_MODELS[0]
    # model_type=2, model_name=TRAINED_MODELS[1]
    # model_type=3, model_name=TRAINED_MODELS[2]
    # model_type=3, model_name=TRAINED_MODELS[3]
    # model_type=4, model_name=TRAINED_MODELS[4]
    start_time = datetime.datetime.now()
    generate_prediction_image(model_type=1, model_name=TRAINED_MODELS[4],
                              test_image_name=IMAGE_LIST[4], saving_image_name="ab9000")
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).seconds
    print("Execution time: ", execution_time, "s")
