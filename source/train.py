import tensorflow as tf

from old_retrain_for_video import VidRetrain
from show_image import Video
from utils import create_model_graph, get_model_info, add_jpeg_decoding, Props

CLASS_NUMBER = 2

model_path = '../inception_dir/classify_image_graph_def.pb'
model_info = get_model_info()
props = Props()

graph, bottleneck_tensor, resized_image_tensor = create_model_graph(model_info, model_path)

with tf.Session(graph=graph) as sess:
    video = Video(interval=0.1)
    retrain = VidRetrain(learning_rate=props.learning_rate)

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

    init = tf.global_variables_initializer()
    sess.run(init)

    # will show updated image after one cycle in the while loop
    i = 0
    while(True):
        try:
            imgs = video.get_images(10, is_show=True)
        except GeneratorExit:
            print('application stopped by user')
            break

        bottlenecks = []
        for img, key_pressed in imgs:
            tensor = retrain.run_bottleneck_on_image(sess, img, jpeg_data_tensor, decoded_image_tensor,
                                                     resized_image_tensor, bottleneck_tensor)
            bottlenecks.append(tensor)

        #TODO dodac kategorie przewciwne (jakies losowe obrazki inne
        categories_index = [i % 2 for i in range(len(bottlenecks))]

        train_summary, _ = sess.run([merged, train_step],
                                    feed_dict={bottleneck_input: bottlenecks,ground_truth_input: categories_index})
        train_writer.add_summary(train_summary, i)

        train_accuracy, cross_entropy_value = sess.run([evaluation_step, cross_entropy],
                                       feed_dict={bottleneck_input: bottlenecks, ground_truth_input: categories_index})
        accuracy = train_accuracy

        i += 1