import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

tf.set_random_seed(777)  # for reproducibilityfor i in range


xy = np.loadtxt('parkinsons-origin.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

scaler = MinMaxScaler(feature_range=(0,1))
x_data = scaler.fit_transform(x_data)

i = 0
a = [0 for i in range(15)]
learning_late = 1.0e-4

kf = KFold(n_splits=15, random_state=None, shuffle=False)
kf.get_n_splits(x_data) # returns the number of splitting iterations in the cross-validator

for train_index, test_index in kf.split(x_data):
    x_train, x_test = x_data[train_index], x_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]

    X = tf.placeholder(tf.float32, shape=[None, 22])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    # W1 = tf.get_variable("1", shape=[2304, 1296],
    #                      initializer=tf.contrib.layers.xavier_initializer())
    W1 = tf.Variable(tf.random_normal([22, 40]), name='weight1')
    b1 = tf.Variable(tf.random_normal([40]), name='bias1')

    W2 = tf.Variable(tf.random_normal([40, 40]), name='weight2')
    b2 = tf.Variable(tf.random_normal([40]), name='bias2')

    W3 = tf.Variable(tf.random_normal([40, 1]), name='weight3')
    b3 = tf.Variable(tf.random_normal([1]), name='bias3')

    h1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
    hypothesis = tf.sigmoid(tf.matmul(h2, W3) + b3)
    hypothesis = tf.clip_by_value(hypothesis, 1e-7, 1 - (1e-7))

    # cost/loss function
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
    train = tf.train.AdamOptimizer(learning_rate=learning_late).minimize(cost)
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(30001):
            cost_val, _ = sess.run([cost, train], feed_dict={X: x_train, Y: y_train})

            if(step % 5000 == 0):
                # pass
                print(step, cost_val)

        h, c, a[i] = sess.run([hypothesis, predicted, accuracy],
                           feed_dict={X: x_test, Y: y_test})
        print(i+1, "Accuracy: ", a[i])
        print("predict", c,"\ntest", y_test)
        i += 1

b = 0
for i in range(15):
    b += a[i]
    print(b)

print ("accuracy : ", a, "\ncross-validation : ", b / 15.0, "learning-late : ", learning_late)
