"""
Neural Network to recognize pictures of Piccolo and Gohan while
also differentiating between pictures with both or neither of them
also code for a support vector machine
"""
import numpy as np
import os  # file path related code
import glob  # helps find files of a certain type with path
from PIL import Image  # loads in and manipulates images
import matplotlib.pyplot as plt
from tqdm import tqdm  # for-loop progress bars
import pickle as pkl  # save transformed image data to file
from sklearn.model_selection import train_test_split  # Creates training and test data
import time
from sklearn.svm import SVC


def assign_image_categories(working_directory, search_term, test_code):
    """
    Searches a working directory for subdirectory names containing a search
    term, inputs the string of the found subdirectory into an empty variable.
    :param working_directory: String representing current working directory.
    :param search_term: String to be checked against the subdirectory names.
    :param test_code: If true, runs prints statements
    :return: If found returns the directory which contains the search term.
    If not found, returns an empty string.
    """
    # Initialize empty string for result
    result = ''

    # Create list of strings representing the subdirectories in working_directory
    directory_list = next(os.walk(working_directory))[1]  # 1 = list of subdirectories in working_directory

    # Search through directory list to find matching string name
    for directory_name in directory_list:
        if test_code:
            print('\n\nCurrently on:\n\'{}\'\n'.format(directory_name))  # TEST CODE
        if directory_name.__contains__(search_term):
            result = working_directory + '\\' + directory_name + '\\'
            if test_code:
                print('SUCCESS: \'{}\' contains \'{}\''.format(directory_name, search_term))  # TEST CODE
                print('result = ', result)  # TEST CODE
        else:  # TEST CODE
            if test_code:
                print('\'{}\' does not contain \'{}\''.format(directory_name, search_term))  # TEST CODE
    return result


def collect_file_paths(working_directory, file_type, test_code):
    """
    Collects all files of a certain type that are contained within the
    working directory and its subfolders.
    :param working_directory: String of the base directory to collect files from
    :param file_type: String of the file extension to be collected
    :param test_code: If true, runs prints statements
    :return: If found, returns list of file paths. If not found, empty list.
    """
    file_path_list = []
    for directory_name in os.walk(working_directory):
        file_path = os.path.join(directory_name[0], file_type)
        if test_code:
            print(file_path)  # TEST CODE
        for image_location in glob.glob(file_path):
            file_path_list.append(image_location)
    if test_code:
        print(len(file_path_list))  # TEST CODE
    return file_path_list


def create_file_paths(category_names, search_term_string, file_type):
    """
    Returns a list of strings of file paths ending in the file type for each string in the array of category names
    :param category_names: An array of strings containing category names
    :param search_term_string: An array of strings containing search terms for each category name
    :param file_type: A string containing the filetype in the format "*.file_type"
    :return:
    """
    print('-----Updating image data lists...')

    # Create string containing the main directory for each file category
    pwd = os.getcwd()

    # Create strings for each category
    directory_string_tuple = []
    file_type_paths = []

    # Collect file paths for each category directory
    for i in range(len(category_names)):
        directory_string_tuple.append(assign_image_categories(pwd, search_term_string[i], test_code=False))
        file_type_paths.append(collect_file_paths(directory_string_tuple[i], file_type, test_code=False))

    # Check number of images
    total_images_size = 0
    for i in range(len(category_names)):
        print('{}  = Number of {} Images'.format(len(file_type_paths[i]), category_names[i]))
        total_images_size += len(file_type_paths[i])
    print('{} = Number of Total Images\n'.format(total_images_size))
    return file_type_paths


def correct_orientation(image_data):
    """
    Corrects any possible wrong image orientation
    that was read in by PIL.Image.open()
    :param image_data: Image from a filepath
    :return: Correctly oriented image
    """
    # Code below adapted from:
    # https://stackoverflow.com/questions/4228530/pil-thumbnail-is-rotating-my-image
    orientation = 274
    if image_data._getexif() is not None:
        exif = dict(image_data._getexif())
        if orientation in exif:  # 274 = orientation exif data
            if exif[orientation] == 3:
                image_data = image_data.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image_data = image_data.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image_data = image_data.rotate(90, expand=True)
    return image_data


def load_transform(file_path, width, height, grayscale_boolean, test_code):
    """
    Loads in an image with format: filepath\\image.extension,
    then transforms the image to specified width and height
    :param file_path:
    :param width: Optional argument for resize width
    :param height: Optional argument for resize height
    :param grayscale_boolean: Optional argument to convert to grayscale
    :param test_code: If true, runs prints statements and test plots
    :return: 1-dimensional array of size (width*height)
    """
    # load image_data
    image_data = Image.open(file_path)

    # correct image_data orientation
    image_data = correct_orientation(image_data)
    if test_code:
        print('\n-----load_transform function test code-----')
        print('Original Image Data Size from filepath:      {}'.format(image_data.size))  # TEST CODE

    # resize image_data
    if test_code:
        print('Resizing image of size {} to ({},{})'.format(image_data.size, width, height))
    image_data = image_data.resize((width, height))
    if test_code:
        print('Transformed image data size, grayscale = {}: {}'.format(grayscale_boolean, image_data.size))  # TEST CODE

    # convert image_data to grayscale
    if grayscale_boolean:
        image_data = image_data.convert('L')

    # convert image_data to array
    if test_code:
        print('Converting image to np array with np.assarrray(image_data):')
    image_data = np.asarray(image_data)
    if test_code:
        print('image_data.shape = {}'.format(image_data.shape))

    # Remove possible transparent alpha layer
    if image_data.shape[2] == 4:
        image_data = image_data[:, :, :3]

    if grayscale_boolean:
        image_data = image_data.reshape((width, height))
        if test_code:
            print('Transformed grayscale image data shape after np.asarrray:         {}'.format(image_data.shape))
    elif not grayscale_boolean:
        image_data = image_data.reshape((width, height, 3))
        if test_code:
            print('Transformed color data shape, np.asarrray:   {}'.format(image_data.shape))

    # flatten image_data
    image_data = image_data.flatten()
    if test_code:
        print('Image array after .flatten():                {}'.format(image_data.shape))

    if test_code:
        if grayscale_boolean:
            print('Grayscale = {}'.format(grayscale_boolean))
            plt.imshow(image_data.reshape(width, height), cmap='gray')
            plt.show()
        elif not grayscale_boolean:
            print('Grayscale = {}'.format(grayscale_boolean))
            plt.imshow(image_data.reshape((width, height, 3)))
            plt.show()
    return image_data


def create_tuples(file_path_list, category_value, category_string,
                  width, height, grayscale_boolean, test_code):
    """
    Creates a tuple of file paths, image data, and
    category values for a specific category of data.
    After running, the tuple should be of the format:
    tuple[x] = ['file path', image_data, category_value]
    :param file_path_list: List of file paths with image extensions
    :param category_value: number from 0 to n to denote the category type
    :param category_string: string of category name
    :param width: width of images
    :param height: height of images
    :param grayscale_boolean: boolean to determine if grayscale transformation is applied
    :param test_code: If true, runs prints statements
    :return: the completed tuple
    """
    category_tuple = []
    category_data = []
    if grayscale_boolean:
        category_data = np.zeros((len(file_path_list), width * height))
    elif not grayscale_boolean:
        category_data = np.zeros((len(file_path_list), width * height * 3))
    if test_code:
        print('\n-----create_tuples function test code-----')
        print('Size of empty {} array: {}'.format(category_string, category_data.shape))  # TEST CODE
    for i in tqdm(range(len(file_path_list))):
        category_data[i, :] = load_transform(file_path_list[i], width, height, grayscale_boolean,
                                             test_code=False)  # Set = True for cute pixel pictures
        category_tuple.append([file_path_list[i], category_data[i, :], category_value])
    return category_tuple


def hidden_layer_to_strings(hidden_neuron_sizes):
    """
    |  Creates two string variables from an int array:
    |
    |  hidden_layer_save_string:
    |  A string to be added to the saved .PNG file name which describes the number of neurons in each hidden layer.
    |
    |  hidden_layer_title_string:
    |  A string to be added to the plot title which describes the number of neurons in each hidden layer.
    :param hidden_neuron_sizes: An int array containing the number of hidden
    neurons in each layer
    :return: hidden_layer_string, hidden_layer_title_string
    """
    hidden_layer_save_string = ''
    hidden_layer_title_string = ''
    comma = ''
    for i in range(len(hidden_neuron_sizes)):
        hidden_layer_save_string = hidden_layer_save_string + 'h' + str(hidden_neuron_sizes[i])
        if i > 1:
            comma = ', '
        elif i == 0:
            comma = ''
        hidden_layer_title_string = hidden_layer_title_string + comma + str(
            hidden_neuron_sizes[i]) + ' hidden neurons in layer ' + str(i + 2)
    return hidden_layer_save_string, hidden_layer_title_string


def string_to_boolean(string_input):
    """
    converts certain string inputs to True or False
    :param string_input: String taken in from an input field
    """
    yes = ['y', 'yes', str(0)]
    no = ['n', 'no', str(1)]
    boolean_convert = 0
    if string_input.lower() in yes:
        boolean_convert = True
    elif string_input.lower() in no:
        boolean_convert = False
    else:
        print('ERROR: INCORRECT INPUT: {}'.format(string_input))
    return boolean_convert


def string_to_filepath(filepath_string):
    """
    Converts a copied and pasted directory path to a path string that can work with python code
    :param filepath_string: python-appropriate string of filepath
    """
    return filepath_string.replace('\\', '\\\\')


def user_variable_printout(user_input_tuple):
    """
    Function to take user input to determine attributes of neural network or support vector machine
    :param user_input_tuple: A list of variables which can be different for neural networks or svm
    :return:
    """
    method = user_input_tuple[0]
    width = user_input_tuple[1]
    height = user_input_tuple[2]
    grayscale_boolean = user_input_tuple[3]
    refresh_boolean = user_input_tuple[4]
    cache_name = user_input_tuple[5]
    print('Selected Method: {}'.format(method))
    print('reshaped image width = {}\n'
          'reshaped image height = {}\n'
          .format(width, height))
    if method == 'neural network':
        hidden_neuron_sizes = user_input_tuple[6]
        learning_rates = user_input_tuple[7]
        epoch_size = user_input_tuple[8]
        optimizer_string = user_input_tuple[9]
        loss_string = user_input_tuple[10]
        model_type_string = user_input_tuple[11]
        verbose_boolean = user_input_tuple[12]
        print('--Neural Network Variables:--\n'
              'hidden_neuron_sizes = {}     \n'
              '     learning_rates = {}     \n'
              '         epoch_size = {}     \n'
              '   optimizer_string = \"{}\"     \n'
              '        loss_string = \"{}\"     \n'
              '  model_type_string = \"{}\"     \n'
              '    verbose_boolean = {}     \n'
              '  grayscale_boolean = {}     \n'
              .format(hidden_neuron_sizes, learning_rates, epoch_size, optimizer_string, loss_string,
                      model_type_string, verbose_boolean, grayscale_boolean))
    elif method == 'svm':
        print('--------SVM Variables--------\n')
    print('refresh_boolean = {}\n'
          'cache_name = {}\n'.format(refresh_boolean, cache_name))


def user_input():
    """
    Prompt user for input options
    :return: cache_name, width, height, grayscale, model_type_string, hidden_neuron_sizes,
        epoch_size, verbose_switch, learning_rates, refresh_boolean, optimizer_string, loss_string
    """
    # Initialize Default Values
    method = input('Choose method, options: \"Neural Network\", \"SVM\"\n').lower()
    width = 100
    height = width
    hidden_neuron_sizes = 250
    learning_rates = [0.01, 0.05, 0.1]
    epoch_size = 200
    optimizer_string = 'SGD'
    loss_string = 'CCE'
    model_type_string = loss_string + ', ' + optimizer_string + ' - '
    verbose_boolean = True
    cache_name_base = 'save_state_'
    cache_name = cache_name_base + str(width)
    user_input_tuple = []
    grayscale_boolean = string_to_boolean(input('\nDo you want the images to be grayscale? (y/n)\n').lower())
    if not grayscale_boolean:
        cache_name = cache_name + '_no_grayscale'
    elif grayscale_boolean:
        cache_name = cache_name + '_yes_grayscale'
    refresh_boolean = string_to_boolean(input('\nDo you want to update image cache before running?(y/n)\n').lower())

    print('\nCurrent Values:')
    if method == 'neural network':
        user_input_tuple = [method, width, height, grayscale_boolean, refresh_boolean, cache_name,
                            hidden_neuron_sizes, learning_rates, epoch_size,
                            optimizer_string, loss_string, model_type_string, verbose_boolean]
        user_variable_printout(user_input_tuple)
    elif method == 'svm':
        user_input_tuple = [method, width, height, grayscale_boolean, refresh_boolean, cache_name]
        user_variable_printout(user_input_tuple)

    change_boolean = string_to_boolean(input('Do you want to change the default values? (y/n)\n'))
    if change_boolean:
        choose_dimension = input('\nChoose square transform image dimension (100, 200, etc.):\n')
        cache_name = cache_name_base + choose_dimension
        width = int(choose_dimension)
        height = width
        if change_boolean and method == 'neural network':
            # List of optimizers: https://keras.io/api/optimizers/
            optimizer_string = input('\nChoose optimizer type,\noptions = SGD or Adam:\n').upper()
            # List of losses: https://keras.io/api/losses/
            loss_string = input('\nChoose loss/cost function,\noptions = MSE or CCE:\n').upper()
            model_type_string = loss_string + ', ' + optimizer_string + ' - '
            hidden_neuron_sizes = list(map(int, input('\nEnter number of dense neurons in each hidden layer:\n'
                                                      'Example input: 2, 5, 10, 20, 30\n').split(', ')))
            epoch_size = int(input('\nChoose number of epochs:\n'))
            verbose_boolean = string_to_boolean(input('\nPrint each epoch\'s values? (y/n)\n').lower())
            learning_rates = list(map(float, input('\nEnter the gamma values for each test of the network:\n'
                                                   'Example input: 0.2, 0.5, 1, 20, 300.\n ').split(', ')))
            user_input_tuple = [method, width, height, grayscale_boolean, refresh_boolean, cache_name,
                                hidden_neuron_sizes, learning_rates, epoch_size,
                                optimizer_string, loss_string, model_type_string, verbose_boolean]
        elif change_boolean and method == 'svm':
            user_input_tuple = [method, width, height, grayscale_boolean, refresh_boolean, cache_name]
        print('Updated Values:\n')
        user_variable_printout(user_input_tuple)
    return user_input_tuple


def verify_tuples(all_tuples, training_inputs, test_inputs,
                  training_targets, test_targets, width, height, grayscale_boolean=False, test_code=False):
    """
    Prints test statements to make sure the creation or loading of tuples and training/test data worked
    :param all_tuples: A tuple with the filepaths, image array data, and category value for each image
    :param training_inputs: The data to train the neural network with
    :param test_inputs: The test labels for the training data
    :param training_targets: The data to test the neural network against
    :param test_targets: The test labels for the target data
    :param width: Width of images
    :param height: Height of images
    :param grayscale_boolean: Boolean to determine if grayscale transformation is applied
    :param test_code: If true, runs test plots and additional prints statements
    :return:
    """
    print('\n-----Test creation/loading of tuples worked:')
    print('all_tuples length:            {}'.format(len(all_tuples)))
    print('all_tuples[0][0], file path:  {}'.format(all_tuples[0][0]))
    print('all_tuples[0][1], image data: {}'.format(all_tuples[0][1]))
    if test_code:
        print('all_tuples[0][1]:')
        if grayscale_boolean:
            plt.imshow(all_tuples[0][1].reshape((width, height, 3)))
            plt.show()
        elif not grayscale_boolean:
            plt.imshow(all_tuples[0][1].reshape((width, height, 3)).astype(np.uint8))
            plt.show()
    print('all_tuples[0][1], length:     {}'.format(len(all_tuples[0][1])))
    print('all_tuples[0][2], category:   {}\n'.format(all_tuples[0][2]))

    print('-----Test that train_test_split worked:')
    print('training_inputs length:   {}'.format(len(training_inputs)))
    print('training_inputs[0] shape: {}'.format(training_inputs[0].shape))
    print('test_inputs length:       {}'.format(len(test_inputs)))
    print('test_inputs[0] shape:     {}'.format(test_inputs[0].shape))
    if test_code:
        print('training_targets length:  {}'.format(len(training_targets)))
        print('training_targets[0]:      {}'.format(training_targets[0]))
        print('test_targets length:      {}'.format(len(test_targets)))
        print('test_targets[0]:          {}\n'.format(test_targets[0]))


def cat_recognizer_neural_network(width, height, training_inputs, training_targets, test_inputs, test_targets,
                                  epoch_size, learning_rates, hidden_neuron_sizes, optimizer_string, loss_string,
                                  model_type_string, model_output_path, grayscale_boolean=False, verbose_boolean=False):
    """
    runs neural network
    :param width: width of images
    :param height: height of images
    :param training_inputs: List of flattened image data to be used for training
    :param training_targets: List of int target values in array, size of category_names, that determines truth value
    :param test_inputs: List of flattened image data to be used for testing
    :param test_targets: List of int test values in array, size of category_names, that determines truth value
    :param epoch_size: Number of epochs
    :param learning_rates: List of values to determine the learning rates, gamma, used in each loop of neural network
    :param optimizer_string: A string containing the name of the optimizer to use
    :param loss_string: A string containing the name of the loss function to use
    :param model_type_string: A string concatenating the optimizer and loss strings
    :param hidden_neuron_sizes: List of values to determine the hidden neuron sizes of each network layer
    :param model_output_path: file path to save the finished model to
    :param grayscale_boolean: Boolean to determine if grayscale transformation is applied
    :param verbose_boolean: Boolean to determine if neural network prints each epoch while running
    :return:
    |  time_string: A string describing the time the neural network completed
    |
    |  plot_tuple: A tuple of size 2 containing the data for loss and accuracy of the completed network
    |
    |  history: A variable containing loss, accuracy, and all other data from model.fit -  also saved model_output_path
    |
    |  learning_rates: The array of learning rates used to create the data
    |
    |  model_type_string: A string describing the optimizer and loss function used
    """
    print('Importing relevant python libraries...')
    import tensorflow as tf  # tensorflow for neural network
    from tensorflow import keras  # keras for creating network
    from keras import layers  # layers for creating layers

    # BEGIN Run Neural Network
    keras.backend.clear_session()
    print('-----Creating Neural Network Model...')
    if grayscale_boolean:
        input_layer_length = width * height
    else:
        input_layer_length = width * height * 3

    # Initialize things to make pycharm happy
    loss_plot = []
    accuracy_plot = []
    val_loss_plot = []
    val_accuracy_plot = []
    optimizer = 0
    loss = 0
    history = 0
    verbose_switch = 0
    if verbose_boolean:
        verbose_switch = 1
    model = keras.backend.clear_session()
    for gamma in tqdm(learning_rates):
        # Initialize model
        model = keras.Sequential()
        # Create Input Layer (l_1)'
        model.add(keras.Input(shape=(input_layer_length,)))
        # Normalizing between 0 and 1
        model.add(layers.Rescaling(1.0 / 255))
        # Create Hidden Layers (l_2 - l_n-1)
        for i in range(len(hidden_neuron_sizes)):
            model.add(layers.Dense(hidden_neuron_sizes[i], activation="sigmoid"))
        # Create Output Layer (l_n)
        model.add(layers.Dense(category_size, activation="sigmoid"))
        # Print Model Summary
        model.summary()
        if optimizer_string == 'SGD':
            optimizer = keras.optimizers.SGD(learning_rate=gamma)
        elif optimizer_string == 'Adam':
            optimizer = keras.optimizers.Adam(learning_rate=gamma)
        if loss_string == 'MSE':
            loss = keras.losses.MeanSquaredError()
        elif loss_string == 'CCE':
            loss = keras.losses.CategoricalCrossentropy()
        # Compile
        model.compile(
            optimizer,
            loss,
            metrics=[keras.metrics.CategoricalAccuracy()],
        )

        # create one_hot vectors of training targets
        training_target_vectors = tf.one_hot(training_targets, depth=len(set(training_targets)))
        test_target_vectors = tf.one_hot(test_targets, depth=len(training_target_vectors[0]))
        print('\n-----Running Neural Network for gamma = {}'.format(gamma))
        print('TIME STARTED: {}'.format(time.strftime("%Y-%m-%d %H-%M-%S")))

        # run neural network with model.fit, store to history variable
        history = model.fit(
            tf.stack(training_inputs),
            training_target_vectors,
            batch_size=10,
            epochs=epoch_size,
            validation_data=(tf.stack(test_inputs), test_target_vectors),
            verbose=verbose_switch
        )
        loss_plot.append(history.history['loss'])  # store training loss plot
        accuracy_plot.append(history.history['categorical_accuracy'])  # store training accuracy plot
        val_loss_plot.append(history.history['val_loss'])  # store test loss plot
        val_accuracy_plot.append(history.history['val_categorical_accuracy'])  # store test accuracy plot
        print('TIME COMPLETED: {}'.format(time.strftime("%Y-%m-%d %H-%M-%S")))
    time_string = time.strftime("%Y-%m-%d %H-%M-%S - ")  # store when the entire for loop completed for file name output
    plot_tuple = [loss_plot, accuracy_plot, val_loss_plot, val_accuracy_plot]

    # Save time_string, plot_tuple, history, and learning_rates variables
    hidden_layer_string, hidden_layer_title_string = hidden_layer_to_strings(hidden_neuron_sizes)
    backup_string = time_string + 'cat_recognizer_backup - ' + model_type_string + hidden_layer_string
    pkl.dump([time_string, plot_tuple, history, learning_rates, hidden_neuron_sizes, model_type_string],
             open(model_output_path + '\\' + backup_string + '.pkl', 'wb'))

    # Save Neural Network Model:
    model.save(os.path.join(model_output_path,
                            'cat_recognizer_model - ' + model_type_string + hidden_layer_string, 'wb'))


def create_plots(time_string, plot_tuple, learning_rates, hidden_neuron_sizes, model_type_string, colors,
                 plot_output_path, show_plot=False):
    """
    Plots results from a previously run iteration of a neural network
    :param time_string: A string describing the time the neural network completed
    :param plot_tuple: A tuple with history objects from keras containing training/test loss/accuracy values over epochs
    :param learning_rates: The array of learning rates used to create the data
    :param hidden_neuron_sizes: An int array containing the number of dense neurons in each hidden layer
    :param model_type_string: A string describing the optimizer and loss function used
    :param colors: A string array of different colors for plotting
    :param plot_output_path: A file path to save the finished plots to
    :param show_plot: A boolean to decide if the plot should be shown in a new window or jupyter printout
    """
    id_string = ['Training', 'Test']
    for index in range(2):
        # Create Strings to use in plots
        plot_tuple_temp = [plot_tuple[2 * index], plot_tuple[2 * index + 1]]
        plot_string = ['Loss', 'Accuracy']
        ylab_string = ['Loss', 'Accuracy']
        title_string = ['Loss on the {} Data'.format(id_string[index]),
                        'Accuracy on the {} Data'.format(id_string[index])]
        hidden_layer_string, hidden_layer_title_string = hidden_layer_to_strings(hidden_neuron_sizes)
        for i in range(2):
            plt.clf()
            plt.ioff()
            fig, ax = plt.subplots()
            plot_save_path = os.path.join(plot_output_path, time_string + model_type_string + id_string[index] + ' '
                                          + plot_string[i] + ', ' + hidden_layer_string)
            plt.title(title_string[i] + '\n' + model_type_string + hidden_layer_title_string)
            for j in range(len(learning_rates)):
                plt.ioff()
                if ylab_string[i] == 'Accuracy':
                    plt.ylim(0, 1)
                    legend_label = 'learning_rate = {}\n' \
                                   'Final Accuracy:\n{}%'.format(str(learning_rates[j]),  # round final accuracy: ".3f"
                                                                 round(100 * plot_tuple_temp[i][j][-1], 3))
                else:
                    legend_label = 'learning_rate = {}'.format(str(learning_rates[j]))
                ax.plot(plot_tuple_temp[i][j],
                        colors[j],
                        zorder=len(learning_rates) - j,
                        label=legend_label
                        )
            plt.xlabel('Epochs')
            plt.ylabel(ylab_string[i])
            ax.legend(bbox_to_anchor=(1.45, 1.02), loc='upper right', frameon=True)
            plt.savefig(os.path.join(plot_save_path + '.png'), dpi=240, bbox_inches='tight')
            print('\n{} {} plot saved to: {}.png'.format(id_string[index], ylab_string[i], plot_save_path))
            if show_plot:
                plt.show()


def run_neural_network(user_input_tuple, training_inputs, training_targets, test_inputs, test_targets):
    """
    A function to run all steps and functions related to the neural network
    after training and test data have been created.
    :param user_input_tuple:
    :param training_inputs:
    :param training_targets:
    :param test_inputs:
    :param test_targets:
    """
    width = user_input_tuple[1]
    height = user_input_tuple[2]
    grayscale_boolean = user_input_tuple[3]
    hidden_neuron_sizes = user_input_tuple[6]
    learning_rates = user_input_tuple[7]
    epoch_size = user_input_tuple[8]
    optimizer_string = user_input_tuple[9]
    loss_string = user_input_tuple[10]
    model_type_string = user_input_tuple[11]
    verbose_boolean = user_input_tuple[12]
    colors = ['red', 'orange', 'lime', 'deepskyblue', 'magenta', 'blue', 'yellow', 'black', 'gray']

    # Create strings to save plots, models and history variables
    # for user input, use function: string_to_filepath()
    model_output_path = os.getcwd() + '\\Cat Neural Network\\Neural Network Models'
    if string_to_boolean(input('Current location of cat_recognizer_backup files:\n\"{}\"\n'
                               'Do you want to update the file path? (y/n)\n'.format(model_output_path))):
        model_output_path = (input('Enter desired file path of cat_recognizer_backup files:\n'))
        print('New location of cat_recognizer_backup files:\n\"{}\"'.format(model_output_path))
    plot_output_path = os.getcwd() + '\\Cat Neural Network\\Neural Network Output Graphs'
    if string_to_boolean(input('Current location of plot output files:\n\"{}\"\n'
                               'Do you want to update the file path? (y/n)\n'.format(plot_output_path))):
        model_output_path = (input('Enter desired file path of plot output files:\n'))
        print('New location of plot output files:\n\"{}\"'.format(plot_output_path))

    # Prompt for running of neural network straight to plots
    startup_choice = string_to_boolean(input('Do you want to run the network before plotting? (y/n)\n').lower())
    if startup_choice:
        cat_recognizer_neural_network(width, height,
                                      training_inputs, training_targets, test_inputs, test_targets,
                                      epoch_size, learning_rates, hidden_neuron_sizes,
                                      optimizer_string, loss_string, model_type_string, model_output_path,
                                      grayscale_boolean=grayscale_boolean, verbose_boolean=verbose_boolean)

    # Create list of strings representing the files in working_directory
    print(model_output_path)
    all_files_list = next(os.walk(model_output_path))[2]  # 2 = list of files in working_directory
    pkl_ext = '.pkl'
    pkl_files_list = [''] * len(all_files_list)
    pkl_choice = 0
    print('List of all .pkl backup files in model output path:')
    for i in range(len(all_files_list)):  # Search through directory list to find matching string name
        if all_files_list[i].__contains__(pkl_ext):
            pkl_files_list[i] = (os.path.join(model_output_path, all_files_list[i]))
            print('{} - {}'.format(i + 1, all_files_list[i]))
    if len(all_files_list) > 0:
        pkl_index = int(input('Select the number above corresponding to the backup to load:\n')) - 1  # -1 for 0-index
        pkl_choice = pkl_files_list[pkl_index]
    elif len(all_files_list) == 0:
        print('ERROR: No history files to load')
    time_string, \
        plot_tuple, \
        history, \
        learning_rates, \
        hidden_neuron_sizes, \
        model_type_string = pkl.load(open(pkl_choice, 'rb'))
    # Create loss and accuracy plots for training data and test data
    create_plots(time_string, plot_tuple, learning_rates, hidden_neuron_sizes,
                 model_type_string, colors, plot_output_path)
    # Load Neural Network Model:
    # model = keras.models.load_model(os.path.join(model_output_path, 'cat_recognizer_model'
    #                                              + model_type_string + hidden_layer_string, 'wb'))


category_names = ['Piccolo', 'Gohan', 'Both', 'Neither']
search_term_string = ['Piccolo', 'Gohan', 'Both', 'Moto G Power']
category_size = len(category_names)
category_values = np.array([i for i in range(len(category_names))])

# create base file paths for each category found with each search term string
file_paths = create_file_paths(category_names, search_term_string, file_type='*.jpg')

user_input_tuple = user_input()
method = user_input_tuple[0]
width = user_input_tuple[1]
height = user_input_tuple[2]
grayscale_boolean = user_input_tuple[3]
refresh_boolean = user_input_tuple[4]
cache_name = user_input_tuple[5]

# Create all_tuples
all_tuples = []
if refresh_boolean:
    for i in range(category_size):
        all_tuples.append((create_tuples(file_paths[i], category_values[i], category_names[i],
                                         width, height, grayscale_boolean, test_code=False)))
    pkl.dump(all_tuples, open(cache_name + '.pkl', 'wb'))  # Save all_tuples to a .pkl file
elif not refresh_boolean:
    all_tuples = pkl.load(open(cache_name + '.pkl', 'rb'))

# Create training and test data
print('\nCreating training and test data...')
training_inputs, test_inputs, training_targets, test_targets = \
    train_test_split([all_tuples[i][1] for i in range(len(all_tuples))],  # Image data[::101]
                     [all_tuples[i][2] for i in range(len(all_tuples))],  # Category value[::101]
                     test_size=0.25,
                     stratify=[all_tuples[i][2] for i in range(len(all_tuples))],  # Stratify for unbalanced data[::101]
                     random_state=523427)

verify_tuples(all_tuples, training_inputs, test_inputs, training_targets, test_targets, width, height,
              grayscale_boolean, test_code=False)

if method == 'neural network':
    run_neural_network(user_input_tuple, training_inputs, training_targets, test_inputs, test_targets)
elif method == 'svm':
    training_targets = np.array(training_targets) + 1
    test_targets = np.array(test_targets) + 1
    # # create one_hot vectors of training targets
    # training_target_vectors = tf.one_hot(training_targets, depth=len(set(training_targets)))
    # test_target_vectors = tf.one_hot(test_targets, depth=len(training_target_vectors[0]))
    classifier = SVC(kernel='poly')
    print('TIME STARTED: {}'.format(time.strftime("%Y-%m-%d %H-%M-%S")))
    classifier.fit(training_inputs, training_targets)
    print('TIME COMPLETED: {}'.format(time.strftime("%Y-%m-%d %H-%M-%S")))
    print('classifier.coef_: {}'.format(classifier.coef0))
    print('classifier.intercept_: {}'.format(classifier.intercept_))
    print('TIME STARTED: {}'.format(time.strftime("%Y-%m-%d %H-%M-%S")))
    print('TRAINING DATA - classifier.score_: {}'.format(classifier.score(training_inputs, training_targets)))
    print('    TEST DATA - classifier.score_: {}'.format(classifier.score(test_inputs, test_targets)))
    print('TIME COMPLETED: {}'.format(time.strftime("%Y-%m-%d %H-%M-%S")))
    # test_predictions = np.sign(test_inputs @ classifier.coef_[0] + classifier.intercept_)
    # test_accuracy = sum(test_predictions == test_targets) / len(test_targets)
    # print("Test Accuracy = {}".format(test_accuracy))
