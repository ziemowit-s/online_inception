
class Props:
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
