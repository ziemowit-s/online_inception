from datetime import datetime

import numpy as np
import tensorflow as tf
from retrain_for_video import VidRetrain
from show_image import Video
from utils import create_model_graph, get_model_info, add_jpeg_decoding, Props

CLASS_NUMBER = 2

model_path = '../tmp/output_graph.pb' #'../inception_dir/classify_image_graph_def.pb'
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

    # Create all Summaries
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(props.summaries_dir + '/train',
                                         sess.graph)
    validation_writer = tf.summary.FileWriter(
        props.summaries_dir + '/validation')

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

        bottlenecks = []
        for img in imgs:
            tensor = retrain.run_bottleneck_on_image(sess, img, jpeg_data_tensor, decoded_image_tensor,
                                                     resized_image_tensor, bottleneck_tensor)
            bottlenecks.append(tensor)

        #TODO dodac kategorie przewciwne (jakies losowe obrazki inne
        categories_index = [i % 2 for i in range(len(bottlenecks))]

        # -----TRAIN------
        #train_summary, _ = sess.run([merged, train_step],
        #                            feed_dict={bottleneck_input: bottlenecks,ground_truth_input: categories_index})
        #train_writer.add_summary(train_summary, i)

        # -----PREDICT------
        final_result = sess.run([final_tensor],
                                       feed_dict={bottleneck_input: bottlenecks})

        ind = np.argmax(final_result)
        val = np.max(final_result)

        if val > 0.5:
            category = 'apple' if ind == 0 else 'paprica'
            accuracy = val
        else:
            category = 'NOTHING'
            accuracy = -1

        i += 1