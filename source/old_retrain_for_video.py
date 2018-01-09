import numpy as np
import tensorflow as tf
from tensorflow.contrib.quantize.python import quant_ops

class VidRetrain:

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def run_bottleneck_on_image(self, sess, image_data, image_data_tensor,
                                decoded_image_tensor, resized_input_tensor,
                                bottleneck_tensor):
        """Runs inference on an image to extract the 'bottleneck' summary layer.
        Args:
          sess: Current active TensorFlow Session.
          image_data: String of raw JPEG data.
          image_data_tensor: Input data layer in the graph.
          decoded_image_tensor: Output of initial image resizing and preprocessing.
          resized_input_tensor: The input node of the recognition graph.
          bottleneck_tensor: Layer before the final softmax.
        Returns:
          Numpy array of bottleneck values.
        """
        # First decode the JPEG image, resize it, and rescale the pixel values.
        resized_input_values = sess.run(decoded_image_tensor,
                                        {image_data_tensor: image_data})
        # Then run it through the recognition network.
        bottleneck_values = sess.run(bottleneck_tensor,
                                     {resized_input_tensor: resized_input_values})
        bottleneck_values = np.squeeze(bottleneck_values)
        return bottleneck_values

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def add_final_training_ops(self, class_count, final_tensor_name, bottleneck_tensor,
                               bottleneck_tensor_size, quantize_layer):
        """Adds a new softmax and fully-connected layer for training.
        We need to retrain the top layer to identify our new classes, so this function
        adds the right operations to the graph, along with some variables to hold the
        weights, and then sets up all the gradients for the backward pass.
        The set up for the softmax and fully-connected layers is based on:
        https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html
        Args:
          class_count: Integer of how many categories of things we're trying to
              recognize.
          final_tensor_name: Name string for the new final node that produces results.
          bottleneck_tensor: The output of the main CNN graph.
          bottleneck_tensor_size: How many entries in the bottleneck vector.
          quantize_layer: Boolean, specifying whether the newly added layer should be
              quantized.
        Returns:
          The tensors for the training and cross entropy results, and tensors for the
          bottleneck input and ground truth input.
        """
        with tf.name_scope('input'):
            bottleneck_input = tf.placeholder_with_default(
                bottleneck_tensor,
                shape=[None, bottleneck_tensor_size],
                name='BottleneckInputPlaceholder')

            ground_truth_input = tf.placeholder(
                tf.int64, [None], name='GroundTruthInput')

        # Organizing the following ops as `final_training_ops` so they're easier
        # to see in TensorBoard
        layer_name = 'final_training_ops'
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                initial_value = tf.truncated_normal(
                    [bottleneck_tensor_size, class_count], stddev=0.001)
                layer_weights = tf.Variable(initial_value, name='final_weights')
                if quantize_layer:
                    quantized_layer_weights = quant_ops.MovingAvgQuantize(
                        layer_weights, is_training=True)
                    self.variable_summaries(quantized_layer_weights)

                self.variable_summaries(layer_weights)
            with tf.name_scope('biases'):
                layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
                if quantize_layer:
                    quantized_layer_biases = quant_ops.MovingAvgQuantize(
                        layer_biases, is_training=True)
                    self.variable_summaries(quantized_layer_biases)

                self.variable_summaries(layer_biases)

            with tf.name_scope('Wx_plus_b'):
                if quantize_layer:
                    logits = tf.matmul(bottleneck_input,
                                       quantized_layer_weights) + quantized_layer_biases
                    logits = quant_ops.MovingAvgQuantize(
                        logits,
                        init_min=-32.0,
                        init_max=32.0,
                        is_training=True,
                        num_bits=8,
                        narrow_range=False,
                        ema_decay=0.5)
                    tf.summary.histogram('pre_activations', logits)
                else:
                    logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
                    tf.summary.histogram('pre_activations', logits)

        final_tensor = tf.nn.softmax(logits, name=final_tensor_name)

        tf.summary.histogram('activations', final_tensor)

        with tf.name_scope('cross_entropy'):
            cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
                labels=ground_truth_input, logits=logits)

        tf.summary.scalar('cross_entropy', cross_entropy_mean)

        with tf.name_scope('train'):
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            train_step = optimizer.minimize(cross_entropy_mean)

        return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
                final_tensor)

    def add_evaluation_step(self, result_tensor, ground_truth_tensor):
        """Inserts the operations we need to evaluate the accuracy of our results.
        Args:
          result_tensor: The new final node that produces results.
          ground_truth_tensor: The node we feed ground truth data
          into.
        Returns:
          Tuple of (evaluation step, prediction).
        """
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                prediction = tf.argmax(result_tensor, 1)
                correct_prediction = tf.equal(prediction, ground_truth_tensor)
            with tf.name_scope('accuracy'):
                evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', evaluation_step)
        return evaluation_step, prediction