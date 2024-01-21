import time
import numpy as np
import pickle as pkl  # save transformed image data to file
import pet_neural_network as pnn
from sklearn.svm import SVC


category_names = ['Piccolo', 'Gohan', 'Both', 'Neither']
search_term_string = ['Piccolo', 'Gohan', 'Both', 'Moto G Power']
category_size = len(category_names)
category_values = np.array([i for i in range(len(category_names))])

# create base file paths for each category found with each search term string
file_paths = pnn.create_file_paths(category_names, search_term_string, file_type='*.jpg')

user_input_tuple = pnn.user_input()
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
        all_tuples.append((pnn.create_tuples(file_paths[i], category_values[i], category_names[i],
                                             width, height, grayscale_boolean, test_code=False)))
    pkl.dump(all_tuples, open(cache_name + '.pkl', 'wb'))  # Save all_tuples to a .pkl file
elif not refresh_boolean:
    all_tuples = pkl.load(open(cache_name + '.pkl', 'rb'))

# Create training and test data
print('\nCreating training and test data...')
training_inputs, test_inputs, training_targets, test_targets = \
    pnn.train_test_split([all_tuples[i][1] for i in range(len(all_tuples))],  # Image data[::101]
                         [all_tuples[i][2] for i in range(len(all_tuples))],  # Category value[::101]
                         test_size=0.25,
                         stratify=[all_tuples[i][2] for i in range(len(all_tuples))],
                         # Stratify for unbalanced data[::101]
                         random_state=523427)

pnn.verify_tuples(all_tuples, training_inputs, test_inputs, training_targets, test_targets, width, height,
                  grayscale_boolean, test_code=False)

if method == 'neural network':
    pnn.run_neural_network(user_input_tuple, training_inputs, training_targets, test_inputs, test_targets,
                           category_size)
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
