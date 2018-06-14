from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy import ndimage
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import mnist_data
import cnn_model

#setting parameters
MODEL_DIRECTORY = "model/model.ckpt"
LOGS_DIRECTORY = "logdir"

training_epochs = 40
TRAIN_BATCH_SIZE = 50
display_step = 100
validation_step = 500
TEST_BATCH_SIZE = 10000

#change the laber into one_hot version
def one_hot(label):
    tmp = np.zeros((label.shape[0], 10))
    for i in range(label.shape[0]):
        tmp[i][label[i]] = 1
    return tmp

def train():

    batch_size = TRAIN_BATCH_SIZE
    num_labels = 10

    #input labels
    train_labels0 = np.fromfile("mnist_train_label",dtype=np.uint8)
    test_labels = np.fromfile("mnist_test_label",dtype=np.uint8)
    
    #train labels= (label,label) (120000=60000*2)
    train_labels0 = np.hstack((train_labels0,train_labels0))
    train_labels0 = one_hot(train_labels0)

    #input data
    train_total_data0 = np.load('120000_train.npy')
    #regularizztion
    train_total_data0 =  (train_total_data0 - (255 / 2.0)) / 255
    train_total_data0 = np.reshape(train_total_data0,(120000, 400))

    #shuffle the train data and label at the same time
    #permutation = np.random.permutation(train_total_data0.shape[0])
    #train_total_data0 = train_total_data0[permutation, :]
    #train_labels0= train_labels0[permutation]

    #validation/train set split, train:val=5:1
    train_total_data = train_total_data0[20000:,]
    validation_data = train_total_data0[:20000,]

    train_labels  = train_labels0[20000:,]
    validation_labels = train_labels0[:20000,]

    train_total_data = np.concatenate((train_total_data, train_labels),axis=1)
    train_size =  train_total_data.shape[0]
    #print(train_size)
    test_data = np.load('10000_test.npy')
    test_data =  (test_data - (255 / 2.0)) / 255
    test_data = np.reshape(test_data,(10000, 400))

    is_training = tf.placeholder(tf.bool, name='MODE')

    x = tf.placeholder(tf.float32, [None, 400])
    y_ = tf.placeholder(tf.float32, [None, 10]) 

    #train the model
    y = cnn_model.CNN(x)

    #Calculate the loss
    with tf.name_scope("LOSS"):
        loss = slim.losses.softmax_cross_entropy(y,y_)

    # Create a summary to monitor loss tensor
    tf.summary.scalar('loss', loss)

    # Define optimizer
    with tf.name_scope("ADAM"):
        batch = tf.Variable(0)

        learning_rate = tf.train.exponential_decay(
            1e-4,  #learning rate
            batch * batch_size,  
            train_size, 
            0.95,  # Decay rate.
            staircase=True)
        # Use simple momentum for the optimization.
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=batch)

    # Calculate learning rate
    tf.summary.scalar('learning_rate', learning_rate)

    # Calculate accuracy
    with tf.name_scope("ACC"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('acc', accuracy)

    # Merge
    merged_summary_op = tf.summary.merge_all()
    
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

    total_batch = int(train_size / batch_size)

    #Write into the log 
    summary_writer = tf.summary.FileWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())

    max_acc = 0.

    #train
    for epoch in range(training_epochs)
        #shuffling
        np.random.shuffle(train_total_data)
        train_data_ = train_total_data[:, :-num_labels]
        train_labels_ = train_total_data[:, -num_labels:]

        for i in range(total_batch):
            
            #get the train x and y
            offset = (i * batch_size) % (train_size)
            batch_xs = train_data_[offset:(offset + batch_size), :]
            batch_ys = train_labels_[offset:(offset + batch_size), :]

            #run the session
            _, train_accuracy, summary = sess.run([train_step, accuracy, merged_summary_op] , feed_dict={x: batch_xs, y_: batch_ys, is_training: True})

            summary_writer.add_summary(summary, epoch * total_batch + i)

            if i % display_step == 0:
                print("Epoch:", '%04d,' % (epoch + 1),
                "batch_index %4d/%4d, training accuracy %.5f" % (i, total_batch, train_accuracy))

            # display validation accuracy
            if i % validation_step == 0:
                validation_accuracy = sess.run(accuracy,
                feed_dict={x: validation_data, y_: validation_labels, is_training: False})

                print("Epoch:", '%04d,' % (epoch + 1),
                "batch_index %4d/%4d, validation accuracy %.5f" % (i, total_batch, validation_accuracy))

            # Save the model
            if validation_accuracy > max_acc:
                max_acc = validation_accuracy
                save_path = saver.save(sess, MODEL_DIRECTORY)
                print("Model updated and saved in file: %s" % save_path)

    print("Optimization Finished!")
    saver.restore(sess, MODEL_DIRECTORY)

    # Calculate test accuracy
    test_size = test_labels.shape[0]
    batch_size = TEST_BATCH_SIZE
    total_batch = int(test_size / batch_size)

    acc_buffer = []

    for i in range(total_batch):
        offset = (i * batch_size) % (test_size)
        batch_xs = test_data[offset:(offset + batch_size), :]
        batch_ys = test_labels[offset:(offset + batch_size), :]

        y_final = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys, is_training: False})
        correct_prediction = np.equal(np.argmax(y_final, 1), np.argmax(batch_ys, 1))
        acc_buffer.append(np.sum(correct_prediction) / batch_size)

    print("test accuray for the stored model: %.5f" % np.mean(acc_buffer))

if __name__ == '__main__':
    train()