import tensorflow as tf
import pandas
import numpy as np
import sys


def load_data(file_path):
    # data = np.loadtxt(file_path, delimiter=',')
    data = pandas.read_csv(file_path, sep=',')
    # print(type(data))
    _x_data = data.iloc[:, -4:].values
    _y_data = data.iloc[:, [1]].values

    # print(type(x_data))
    return _x_data, _y_data


def pre_process_class(_y_data):
    class_number = 0
    for i in range(len(_y_data)):
        cls = _y_data[i, 0]
        # print(cls)
        key_set = function_dictionary.keys()
        # print(len(key_set))
        if cls not in key_set:
            function_dictionary[cls] = class_number
            class_number += 1
        _y_data[i, 0] = function_dictionary[cls]
    # print(y_data)
    # print(class_dictionary)


def split_data(data):
    splitted_data = np.vsplit(data, len(function_dictionary))
    splitted_train_data = list(map(lambda x: x[:int(len(x)*0.9)], splitted_data))
    splitted_test_data = list(map(lambda x: x[int(len(x)*0.9):], splitted_data))
    train_data = np.vstack(splitted_train_data)
    test_data = np.vstack(splitted_test_data)
    return train_data, test_data


def compose_neural_network(number_of_input, number_of_output):
    # 첫번째 가중치의 차원은 [특성, 히든 레이어의 뉴런갯수]
    layer1_out = 15*15
    w1 = tf.Variable(tf.random_uniform([number_of_input, layer1_out], -1., 1.))
    # 두번째 가중치의 차원을 [전 레이어의 뉴런 갯수, 분류 갯수]
    w2 = tf.Variable(tf.random_uniform([layer1_out, number_of_output], -1., 1.))

    # 편향을 각각 각 레이어의 아웃풋 갯수로 설정합니다.
    # b1 은 히든 레이어의 뉴런 갯수로, b2 는 최종 결과값 즉, 분류 갯수인 3으로 설정합니다.
    b1 = tf.Variable(tf.zeros([layer1_out]))
    b2 = tf.Variable(tf.zeros([number_of_output]))

    # 신경망의 히든 레이어에 가중치 W1과 편향 b1을 적용합니다
    layer1 = tf.add(tf.matmul(X, w1), b1)
    layer1 = tf.nn.relu(layer1)

    # 최종적인 아웃풋을 계산합니다.
    # 히든레이어에 두번째 가중치 W2와 편향 b2를 적용하여 15개의 출력값을 만들어냅니다.
    _model = tf.add(tf.matmul(layer1, w2), b2)

    # 텐서플로우에서 기본적으로 제공되는 크로스 엔트로피 함수를 이용해
    # 복잡한 수식을 사용하지 않고도 최적화를 위한 비용 함수를 다음처럼 간단하게 적용할 수 있습니다.
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=_model))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_optimizer = optimizer.minimize(cost)
    return _model, train_optimizer, cost


def train(_x_data, _y_data):
    print('========== training ==========')
    _model, train_optimizer, cost = compose_neural_network(4, len(function_dictionary))

    init = tf.global_variables_initializer()
    _sess = tf.Session()
    _sess.run(init)

    _y_data = tf.one_hot(_y_data, depth=len(function_dictionary)).eval(session=_sess)
    _y_data = tf.reshape(_y_data, shape=[-1, len(function_dictionary)]).eval(session=_sess)
    # print(y_train_data)

    epoch = 1000
    for step in range(epoch):
        _sess.run(train_optimizer, feed_dict={X: _x_data, Y: _y_data})

        if (step + 1) % 10 == 0:
            print(step + 1, _sess.run(cost, feed_dict={X: _x_data, Y: _y_data}))

    return _sess, _model


def test(_x_data, _y_data):
    print('========== testing ==========')
    _y_data = tf.one_hot(_y_data, depth=len(function_dictionary)).eval(session=sess)
    _y_data = tf.reshape(_y_data, shape=[-1, len(function_dictionary)]).eval(session=sess)

    # print([_x_data[0]])
    prediction = tf.argmax(model, 1)
    target = tf.argmax(Y, 1)
    for i in range(len(_y_data)):
        predict = sess.run(prediction, feed_dict={X: [_x_data[i]]})
        solution = sess.run(target, feed_dict={Y: [_y_data[i]]})
        print([name for name, idx in function_dictionary.items() if idx == predict[0]], ' | ', [name for name, idx in function_dictionary.items() if idx == solution[0]])

    is_correct = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: _x_data, Y: _y_data}))

if __name__ == '__main__':
    function_dictionary = dict()
    assembly_dictionary = dict()

    argc = len(sys.argv)
    if argc < 2:
        x_data, y_data = load_data('./data/selected_function.csv')
    else:
        x_data, y_data = load_data(sys.argv[1])
    pre_process_class(y_data)
    print(function_dictionary)
    x_train_data, x_test_data = split_data(x_data)
    y_train_data, y_test_data = split_data(y_data)

    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    sess, model = train(x_data, y_data)
    test(x_test_data, y_test_data)
