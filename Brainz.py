import tensorflow as tf
import numpy as np
import pickle
import cv2

TRAIN = False
if not TRAIN:
    NO_TRAIN = True
else:
    NO_TRAIN = False

if NO_TRAIN:
    session = tf.Session()
if TRAIN:
    file = open('GamePlay.illuminati', 'rb')
    game_play = np.asarray(pickle.load(file))
    file.close()

training_iterations = 25
input_size = 80*60
classes = 6

structure = [4000, 3000, 2000, 1000, 500, 100, classes]


def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

def data_test():
    for frame in game_play:
        print(frame[1])

def initialize_trainable_vars(input_size, structure):
    w = []
    b = []
    for i in range(len(structure)):
        if i == 0:
            w.append(tf.Variable(glorot_init([input_size, structure[i]])))
        else:
            w.append(tf.Variable(glorot_init([structure[i - 1], structure[i]])))
        b.append(tf.Variable(glorot_init([structure[i]])))

    return [w, b]


weights, biases = initialize_trainable_vars(input_size, structure)

def predict(x, w, b, activation=tf.nn.sigmoid):
    layer = None
    for i in range(len(structure)):
        if i == 0:
            layer = activation(tf.add(tf.matmul(x, w[i]), b[i]))
        else:
            layer = activation(tf.add(tf.matmul(layer, w[i]), b[i]))

    return layer

X = tf.placeholder(tf.float32, [None, input_size])
Y = tf.placeholder(tf.float32, [None, classes])

prediction = predict(X, weights, biases)
loss = tf.reduce_mean(tf.pow(Y-prediction, 2))
optimizer = tf.train.RMSPropOptimizer(0.01)
train = optimizer.minimize(loss, var_list=[weights, biases])
pure_prediction = tf.argmax(tf.reshape(prediction, [-1]))

SAVER = tf.train.Saver()
init = tf.global_variables_initializer()

if NO_TRAIN:
    session.run(init)
    SAVER.restore(session, 'ModelSave/Brainz')

def showcase(frame):
    cv2.imshow("NeuralView", cv2.resize(frame, (960, 557)))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

def predict(x):
    return session.run(pure_prediction, feed_dict={X:x})

if TRAIN:
    with tf.Session() as sess:
        sess.run(init)
        for step in range(training_iterations):
            for frame_num in range(len(game_play)):
                x, y = np.asarray(game_play[frame_num][0].reshape([1, 60*80])), np.asarray(game_play[frame_num][1]).reshape([1, -1])
                sess.run(train, feed_dict={X:x, Y:y})
                if (step*frame_num) % 100 == 0 and not step*frame_num==0:
                    print("Step:", step, "Frame_num:", frame_num, "Loss", sess.run(loss, feed_dict={X:x, Y:y}))
                    showcase(x.reshape([60, 80]))
        SAVER.save(sess, 'ModelSave/Brainz')