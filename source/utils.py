import tensorflow as tf
from tensorflow.python.platform import gfile


class Props:
    image_dir = '../flower_photos'
    model_dir = '../data/inception_dir'
    bottleneck_dir = '../data/tmp/bottleneck'
    final_tensor_name = 'final_result'
    summaries_dir = '../data/tmp/retrain_logs'
    architecture = 'inception_v3'
    how_many_training_steps = 4000
    train_batch_size = 100
    test_batch_size = -1  # A value of -1 causes the entire test set to be used, which leads to more stable results across runs.\
    eval_step_interval = 10  # How often to evaluate the training results.
    validation_batch_size = 100
    intermediate_store_frequency = 0  # How many steps to store intermediate graph. If "0" then will not store.\
    intermediate_output_graphs_dir = '../data/tmp/intermediate_graph/'
    print_misclassified_test_images = False  # Whether to print out a list of all misclassified test images
    output_graph = '../data/tmp/output_graph.pb'
    output_labels = '../data/tmp/output_labels.txt'
    learning_rate = 0.01
    flip_left_right = False  # Whether to randomly flip half of the training images horizontally
    random_crop = 0  # A percentage determining how much of a margin to randomly crop off the training images.\
    random_scale = 0  # A percentage determining how much to randomly scale up the size of the training images by.\
    random_brightness = 0  # A percentage determining how much to randomly multiply the training image input pixels up or down by.\
    random_scale = 0  # A percentage determining how much to randomly scale up the size of the training images by.\
    random_crop = 0  # A percentage determining how much of a margin to randomly crop off the training images.\

def get_model_info():
    is_quantized = False

    # pylint: disable=line-too-long
    data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    # pylint: enable=line-too-long
    bottleneck_tensor_name = 'pool_3/_reshape:0'
    bottleneck_tensor_size = 2048
    input_width = 299
    input_height = 299
    input_depth = 3
    resized_input_tensor_name = 'Mul:0'
    model_file_name = 'classify_image_graph_def.pb'
    input_mean = 128
    input_std = 128

    return {
        'data_url': data_url,
        'bottleneck_tensor_name': bottleneck_tensor_name,
        'bottleneck_tensor_size': bottleneck_tensor_size,
        'input_width': input_width,
        'input_height': input_height,
        'input_depth': input_depth,
        'resized_input_tensor_name': resized_input_tensor_name,
        'model_file_name': model_file_name,
        'input_mean': input_mean,
        'input_std': input_std,
        'quantize_layer': is_quantized,
    }


def add_jpeg_decoding(input_width, input_height, input_depth, input_mean,
                      input_std):
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                             resize_shape_as_int)
    offset_image = tf.subtract(resized_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    return jpeg_data, mul_image

def create_model_graph(model_info, model_path):
    with tf.Graph().as_default() as graph:
        print('Model path: ', model_path)
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
                graph_def,
                name='',
                return_elements=[
                    model_info['bottleneck_tensor_name'],
                    model_info['resized_input_tensor_name'],
                ]))
    return graph, bottleneck_tensor, resized_input_tensor