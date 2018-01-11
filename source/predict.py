import numpy as np
import tensorflow as tf

from old_retrain_for_video import VidRetrain
from show_image import Video
from utils import create_model_graph, get_model_info, add_jpeg_decoding, Props

CLASS_NUMBER = 2

model_path = '../data/tmp/output_graph.pb' #'../inception_dir/classify_image_graph_def.pb'
model_info = get_model_info()
props = Props()

graph, bottleneck_tensor, resized_image_tensor = create_model_graph(model_info, model_path)

accuracy = 0.0
category = 'NO_CAT'

with tf.Session(graph=graph) as sess:
    bottleneck_input = tf.get_default_graph().get_tensor_by_name('input/BottleneckInputPlaceholder:0')
    final_tensor = tf.get_default_graph().get_tensor_by_name('final_result:0')

    video = Video(interval=0.1)
    retrain = VidRetrain(learning_rate=props.learning_rate)

    jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
        model_info['input_width'], model_info['input_height'],
        model_info['input_depth'], model_info['input_mean'],
        model_info['input_std'])

    init = tf.global_variables_initializer()
    sess.run(init)

    # will show updated image after one cycle in the while loop
    i = 0
    while(True):
        try:
            imgs = video.get_images(1, is_show=True, category_name=category, accuracy=accuracy)
        except GeneratorExit:
            print('application stopped by user')
            break

        for img, key_pressed in imgs:
            tensor = retrain.run_bottleneck_on_image(sess, img, jpeg_data_tensor, decoded_image_tensor,
                                                     resized_image_tensor, bottleneck_tensor)

            # Predict result
            final_result = sess.run([final_tensor],
                                           feed_dict={bottleneck_input:[tensor]})

        index = np.argmax(final_result)
        prediction = np.max(final_result)

        if prediction > 0.5:
            category = 'apple' if index == 0 else 'paprica'
            accuracy = prediction
        else:
            category = 'NOTHING'
            accuracy = -1

        i += 1