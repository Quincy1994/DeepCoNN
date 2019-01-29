# coding=utf-8

import numpy as np
import tensorflow as tf
import math
from tensorflow.contrib import learn

import sys
sys.path.append("..")

import datetime

from model import DeepCoNN

import pickle

# =================================== input data ==================================
tf.flags.DEFINE_string("word2vec", "../../data/embedding/GoogleNews-vectors-negative300.bin", "Word2file with pre-trained embeddings")
tf.flags.DEFINE_string("valid_data", "../../data/music/music.valid", " Data for validation")
tf.flags.DEFINE_string("para_data", "../../data/music/music.para", "Data parameters")
tf.flags.DEFINE_string("train_data", "../../data/music/music.train", "Data for training")
tf.flags.DEFINE_string("weight_user", "../../data/embedding/W_user.pk", "word2vec file from user vocabulary")
tf.flags.DEFINE_string("weight_item", "../../data/embedding/W_item.pk", "Word2vec file from item vocabulary")


# Model Hyperparameters
# =================================================================================
tf.flags.DEFINE_string("embedding_dim", 300, "Dimensionality of character embedding")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes ")
tf.flags.DEFINE_string("num_filters", 100, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda")
tf.flags.DEFINE_float("l2_reg_V", 0, "L2 regularization V")

# Training parmeters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 40, "Number of training epochs ")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps ")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps ")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def train_step(u_batch, i_batch, uid, iid, y_batch, batch_num):
    """
    A single training step
    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.dropout_keep_prob: FLAGS.dropout_keep_prob
    }

    _, step, loss, accuracy, mae = sess.run(
        [train_op, global_step, deep.loss, deep.accuracy, deep.mae],
        feed_dict
    )
    time_str = datetime.datetime.now().isoformat()

    print("{}: step {}, loss {:g}, rmse {:g}, mae {:g}".format(time_str, batch_num, loss, accuracy, mae))
    return accuracy, mae

def dev_step(u_batch, i_batch, uid, iid, y_batch, writer=None):
    """
    Evaluates model on a dev set
    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.dropout_keep_prob: 1.0
    }
    step, loss, accuracy, mae = sess.run(
        [global_step, deep.loss, deep.accuracy, deep.mae],
        feed_dict
    )
    time_str = datetime.datetime.now().isoformat()
    # print("{}: step{}, loss {:g}, rmse {:g}, mae {:g}".format(time_str, step, loss, accuracy, mae))

    return loss, accuracy, mae






if __name__ == '__main__':

    print("Loading data...")

    pkl_file = open(FLAGS.para_data, "rb")

    para = pickle.load(pkl_file)
    user_num = para['user_num']
    item_num = para['item_num']
    user_length = para['user_length']
    item_length = para['item_length']
    vocabulary_user = para['user_vocab']
    vocabulary_item = para['item_vocab']
    train_length = para['train_length']
    test_length = para['test_length']
    u_text = para['u_text']
    i_text = para['i_text']

    np.random.seed(2017)
    random_seed = 2017

    with tf.Graph().as_default():

        session_conf = tf.ConfigProto(
            allow_soft_placement = FLAGS.allow_soft_placement,
            log_device_placement = FLAGS.log_device_placement
        )
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            deep = DeepCoNN(
                user_num=user_num,
                item_num=item_num,
                user_length=user_length,
                item_length=item_length,
                num_classes=1,
                user_vocab_size=len(vocabulary_user),
                item_vocab_size=len(vocabulary_item),
                embedding_size=FLAGS.embedding_dim,
                fm_k=8,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                l2_reg_V=FLAGS.l2_reg_V,
                n_latent=32
            )
        tf.set_random_seed(random_seed)
        global_step = tf.Variable(0, name="global_step", trainable=False)

        optimizer = tf.train.AdamOptimizer(0.002, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(deep.loss)
        train_op = optimizer

        sess.run(tf.initialize_all_variables())

        if FLAGS.word2vec:
            # initial matrix with random uniform
            initW = np.random.uniform(-1.0, 1.0, (len(vocabulary_user), FLAGS.embedding_dim))
            W_u_file = open(FLAGS.weight_user, 'rb')
            initW = pickle.load(W_u_file)
            sess.run(deep.W1.assign(initW))

            # load any vectors from the word2vec
            initW = np.random.uniform(-1.0, 1.0, (len(vocabulary_item), FLAGS.embedding_dim))
            W_i_file = open(FLAGS.weight_item, 'rb')
            initW = pickle.load(W_i_file)
            sess.run(deep.W2.assign(initW))
            print("get pre-trained initW")

    l = (train_length / FLAGS.batch_size) + 1
    print(l)
    ll = 0
    epoch = 1
    best_mae = 5
    best_rmse = 5
    train_mae = 0
    train_rmse = 0

    pkl_file = open(FLAGS.train_data, 'rb')

    train_data = pickle.load(pkl_file)

    train_data = np.array(train_data)
    pkl_file.close()

    pkl_file = open(FLAGS.valid_data, 'rb')

    test_data = pickle.load(pkl_file)
    test_data = np.array(test_data)
    pkl_file.close()

    data_size_train = len(train_data)
    data_size_test = len(test_data)
    batch_size = 100
    ll = int(len(train_data) / batch_size)

    for epoch in range(40):

        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size_train))
        shuffled_data = train_data[shuffle_indices]
        for batch_num in range(ll):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size_train)
            data_train = shuffled_data[start_index: end_index]

            uid, iid, y_batch = zip(*data_train)

            u_batch = []
            i_batch = []
            for i in range(len(uid)):
                u_batch.append(u_text[uid[i][0]])
                i_batch.append(i_text[iid[i][0]])
            u_batch = np.array(u_batch)
            i_batch = np.array(i_batch)

            t_rmse, t_mae = train_step(u_batch, i_batch, uid, iid, y_batch, batch_num)
            current_step = tf.train.global_step(sess, global_step)
            train_rmse += t_rmse
            train_mae += t_mae

            if batch_num % 100 == 0 and batch_num > 1:
                print("\nEvaluation:")
                print(batch_num)
                loss_s = 0.0
                accuracy_s = 0.0
                mae_s = 0.0

                ll_test = int(len(test_data) / batch_size) + 1
                for batch_num2 in range(ll_test):
                    start_index = batch_num2 * batch_size
                    end_index = min((batch_num2 + 1) * batch_size, data_size_test)
                    data_test = test_data[start_index: end_index]

                    userid_valid, itemid_valid, y_valid = zip(*data_test)

                    u_valid = []
                    i_valid = []
                    for i in range(len(userid_valid)):
                        u_valid.append(u_text[userid_valid[i][0]])
                        i_valid.append(i_text[itemid_valid[i][0]])
                    u_valid = np.array(u_valid)
                    i_valid = np.array(i_valid)

                    loss, accuarcy, mae = dev_step(u_valid, i_valid, userid_valid, itemid_valid, y_valid)
                    loss_s = loss_s + len(u_valid) * loss
                    accuracy_s = accuracy_s + len(u_valid) * np.square(accuarcy)
                    mae_s = mae_s + len(u_valid) * mae
                print("loss valid {:g}, rmse_valid {:g}, mae_valid {:g}".format(loss_s / test_length,
                                                                                np.sqrt(
                                                                                    accuracy_s / test_length),
                                                                                    mae_s / test_length
                                                                                ))
        print("epoch " + str(epoch) + ":\n")
        print("\nEvaluation:")
        print("train:rmse, mae:", train_rmse / ll, train_mae / ll)

        train_rmse = 0.0
        train_mae = 0.0

        loss_s = 0.0
        accuracy_s = 0.0
        mae_s = 0.0

        ll_test = int(len(test_data) / batch_size) + 1
        for batch_num in range(ll_test):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size_test)
            data_test = test_data[start_index: end_index]

            userid_valid, itemid_valid, y_valid = zip(*data_test)
            u_valid = []
            i_valid = []
            for i in range(len(userid_valid)):
                u_valid.append(u_text[userid_valid[i][0]])
                i_valid.append(i_text[itemid_valid[i][0]])
            u_valid = np.array(u_valid)
            i_valid = np.array(i_valid)

            loss, accuracy, mae = dev_step(u_valid, i_valid, userid_valid, itemid_valid, y_valid)
            loss_s = loss_s + len(u_valid) * loss
            accuracy_s = accuracy_s + len(u_valid) * np.square(accuracy)
            mae_s = mae_s + len(u_valid) * mae

        print("loss_valid {:g}, rmse_valid {:g}, mae_valid {:g}".format(loss_s / test_length,
                                                                            np.sqrt(accuracy_s / test_length),
                                                                            mae_s / test_length))

        rmse = np.sqrt(accuracy_s / test_length)
        mae = mae_s / test_length
        if best_rmse > rmse:
            best_rmse = rmse
        if best_mae > mae:
            best_mae = mae
        print("")
    print('best rmse:', best_rmse)
    print('best mae:', best_mae)

print("end")

