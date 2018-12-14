import numpy as np
import tensorflow as tf

from model import Model
from show_image import Video
from utils import get_model_info, Props, create_model_graph, add_jpeg_decoding

CLASS_NUMBER = 2

model_path = '../data/inception_dir/classify_image_graph_def.pb'
model_info = get_model_info()
props = Props()

graph, bottleneck_tensor, resized_image_tensor = create_model_graph(model_info, model_path)

accuracy = 0.0
category = 'NO_CAT'

with tf.Session(graph=graph) as sess:

    video = Video(interval=0.1)
    retrain = Model(learning_rate=props.learning_rate)

    jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
        model_info['input_width'], model_info['input_height'],
        model_info['input_depth'], model_info['input_mean'],
        model_info['input_std'])

    # Add the new layer that we'll be training.
    (train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor) = retrain.add_final_training_ops(
        CLASS_NUMBER, props.final_tensor_name, bottleneck_tensor,
        model_info['bottleneck_tensor_size'], model_info['quantize_layer'])

    # Create the operations we need to evaluate the accuracy of our new layer.
    evaluation_step, prediction = retrain.add_evaluation_step(final_tensor, ground_truth_input)

    # Create all Summaries
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(props.summaries_dir + '/train',
                                         sess.graph)
    validation_writer = tf.summary.FileWriter(
        props.summaries_dir + '/validation')

    final_tensor = tf.get_default_graph().get_tensor_by_name('final_result:0')

    init = tf.global_variables_initializer()
    sess.run(init)

    train_bottlenecks = []
    train_ground_truth = []

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
            key_pressed = int(key_pressed)

            # === PREDICT ===
            if key_pressed == -1:
                i = 0
                train_bottlenecks = []
                train_ground_truth = []

                final_result = sess.run([final_tensor], feed_dict={bottleneck_input: [tensor]})
                index = np.argmax(final_result)
                prediction = np.max(final_result)

                if prediction > 0.5:
                    category = str(index)
                    accuracy = str(prediction)
                else:
                    category = 'NOTHING'
                    accuracy = -1

            # === TRAIN ===
            else:
                train_bottlenecks.append(tensor)
                train_ground_truth.append(key_pressed)

                train_summary, _ = sess.run([merged, train_step],
                                            feed_dict={bottleneck_input: [tensor], ground_truth_input: [key_pressed]})
                train_writer.add_summary(train_summary, i)

                # show results
                if (i % 50) == 0:
                    train_accuracy, cross_entropy_value = sess.run(
                        [evaluation_step, cross_entropy],
                        feed_dict={bottleneck_input: train_bottlenecks,
                                   ground_truth_input: train_ground_truth})
                    train_bottlenecks = []
                    train_ground_truth = []
                    category = '[TRAINING] %s' % str(key_pressed)
                    accuracy = str(train_accuracy)

                i += 1