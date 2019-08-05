#Inspired by https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3%20-%20Neural%20Networks/recurrent_network.py
import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data

from packer_identificator import data_processor as dp

# configuration
#                        O * W + b -> 10 labels for each image, O[? 28], W[28 10], B[10]
#                       ^ (O: output 28 vec from 28 vec input)
#                       |
#      +-+  +-+       +--+
#      |1|->|2|-> ... |28| time_step_size = 28
#      +-+  +-+       +--+
#       ^    ^    ...  ^
#       |    |         |
# img1:[28] [28]  ... [28]
# img2:[28] [28]  ... [28]
# img3:[28] [28]  ... [28]
# ...
# img128 or img256 (batch_size or test_size 256)
#      each input size = input_vec_size=lstm_size=28

# configuration variables
input_vec_size = 256
lstm_size = 256
time_step_size = 15

batch_size = 128
test_size = 200

class_numbers = 17

packer_dictionary = dict()

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def pre_process_class(Y):
    class_number = 0
    int_y = np.zeros((1,len(Y)))
    for i in range(len(Y)):
        cls = ''.join(Y[i, :])
        # cls = Y[i, 1]
        value_set = list(packer_dictionary.values())
        # print(packer_dictionary.items())
        if cls not in value_set:
            # packer_dictionary[cls] = class_number
            packer_dictionary[class_number] = cls
            class_number += 1
        # print(cls)
        # print([name for name, idx in packer_dictionary.items() if idx == cls][0])
        int_y[0, i] = [name for name, idx in packer_dictionary.items() if idx == cls][0]
    # print(len(Y))
    # print(int_y)
    # print(int_y.shape)
    return int_y


def model(X, W, B, lstm_size):
    # X, input shape: (batch_size, time_step_size, input_vec_size)
    XT = tf.transpose(X, [1, 0, 2])  # permute time_step_size and batch_size
    # XT shape: (time_step_size, batch_size, input_vec_size)
    XR = tf.reshape(XT, [-1, input_vec_size]) # each row has input for each lstm cell (lstm_size=input_vec_size)
    # XR shape: (time_step_size * batch_size, input_vec_size)
    X_split = tf.split(XR, time_step_size, 0) # split them to time_step_size (15 arrays)
    # Each array shape: (batch_size, input_vec_size)

    # Make lstm with lstm_size (each input vector size)
    lstm = rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)

    # Get lstm cell output, time_step_size (28) arrays with lstm_size output: (batch_size, lstm_size)
    outputs, _states = rnn.static_rnn(lstm, X_split, dtype=tf.float32)

    # Linear activation
    # Get the last output
    return tf.matmul(outputs[-1], W) + B, lstm.state_size # State size to initialize the stat


def reshape_data(data):
    def byte_to_one_hot(byte, byte_numbers=256):
        targets = np.array(byte).reshape(-1)
        return np.eye(byte_numbers)[targets]

    a = np.zeros([len(data), len(data[0]), 256])
    for i in range(len(data)):
        for j in range(len(data[i])):
            a[i, j] = (byte_to_one_hot(data[i, j]))
    return a

training_data_set = dp.load_csv_data("D:/PackerIdentificator/data/training_sample_data_10000.csv")
test_data_set = dp.load_csv_data("D:/PackerIdentificator/data/test_sample_data_5865.csv")
# print(type(training_data_set))
# print(test_data_set.shape)
trX = training_data_set.iloc[:, 5:20].values
trY = training_data_set.iloc[:, 1:3].values
teX = test_data_set.iloc[:, 5:20].values
teY = test_data_set.iloc[:, 1:3].values

trX = reshape_data(trX)
teX = reshape_data(teX)

trY = pre_process_class(trY)
trY = trY.reshape(10000)
teY = pre_process_class(teY)
teY = teY.reshape(5865)

print(trX.shape)
print(trY.shape)

X = tf.placeholder("float", [None, time_step_size, input_vec_size])
Y = tf.placeholder("float", [None, class_numbers])

# get lstm_size and output 10 labels
W = init_weights([lstm_size, class_numbers])
B = init_weights([class_numbers])

py_x, state_size = model(X, W, B, lstm_size)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

session_conf = tf.ConfigProto()
session_conf.gpu_options.allow_growth = True

# Launch the graph in a session
with tf.Session(config=session_conf) as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    trY = tf.one_hot(trY, depth=len(packer_dictionary)).eval(session=sess)
    teY = tf.one_hot(teY, depth=len(packer_dictionary)).eval(session=sess)

    for j in range(10):
        for i in range(50):
            for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size)):
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
            print(i+1, 'th training...')
            # print(teY[i])
            # print(i, np.mean(np.argmax(teY[i], axis=0) == sess.run(predict_op, feed_dict={X: teX[i]})))

        epoch = (j+1)*50
        with open('D:/FunctionIdentificator/packer_identificator/report/report_' + str(epoch) + 'epoch.csv', 'w') as f:
            predictY = np.zeros((1, len(teX)))
            for i in range(len(teX)):
                predict = sess.run(predict_op, feed_dict={X: teX[i].reshape(1, 15, 256)})
                predictY[0, i] = predict
                # for name, idx in packer_dictionary.items():
                #     if idx == predict:
                #         f.write(name)
                #         f.write('\n')

            report_data = pd.DataFrame(columns = ['actual', 'predicted'])

            # print(teY)
            # while True:
            #     continue

            for i in range(len(teX)):
                class_data = [[np.argmax(teY[i], axis=0), int(predictY[0, i])]]
                # print(class_data)
                report_data = report_data.append(pd.DataFrame(class_data, columns=['actual', 'predicted']))
            # print(report_data)
            # print(report_data.shape)

            f.write('packer,#,precision,recall,f-measure\n')
            for i in range(len(packer_dictionary)):
                i_total = report_data[(report_data.actual == i)]
                i_TP = report_data[(report_data.actual == i) & (report_data.predicted == i)]
                i_FP = report_data[(report_data.actual != i) & (report_data.predicted == i)]
                i_FN = report_data[(report_data.actual == i) & (report_data.predicted != i)]

                class_total = i_total.actual.count()
                TP = float(i_TP.actual.count())
                FP = float(i_FP.actual.count())
                FN = float(i_FN.actual.count())

                try:
                    precision = TP/(TP+FP)
                except ZeroDivisionError:
                    precision = 0

                try:
                    recall = TP/(TP+FN)
                except ZeroDivisionError:
                    recall = 0

                try:
                    f_measure = 2*precision*recall/(precision+recall)
                except ZeroDivisionError:
                    f_measure = 0

                f.write('{},{},{},{},{}\n'.format(packer_dictionary[i], class_total, precision, recall, f_measure))

        print('complete')
        # print(np.mean( == np.argmax(teY[:], axis=0)))
