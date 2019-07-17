import tensorflow as tf

from tensorflow import ConfigProto
from tensorflow.python.ops import control_flow_ops

import time, os
import numpy as np
from random import shuffle

from tensorflow.keras.datasets.cifar10 import load_data

home_path = os.path.dirname(os.path.abspath(__file__))

tf.app.flags.DEFINE_string('train_dir', 'test',
                           'Directory where checkpoints and event logs are written to.')
FLAGS = tf.app.flags.FLAGS
def main(_):
    ### define path and hyper-parameter
    Learning_rate =1e-1

    batch_size = 128
    val_batch_size = 200
    train_epoch = 100
    
    weight_decay = 5e-4

    should_log          = 200
    save_summaries_secs = 20
    tf.logging.set_verbosity(tf.logging.INFO)
    gpu_num = '0'

    (train_images, train_labels), (val_images, val_labels) = load_data()
    num_label = int(np.max(train_labels)+1)

    dataset_len, *image_size = train_images.shape

    with tf.Graph().as_default() as graph:
        # make placeholder for inputs
        image_ph = tf.placeholder(tf.uint8, [None]+image_size)
        label_ph = tf.placeholder(tf.int32, [None])
        is_training_ph = tf.placeholder(tf.bool,[])
        
        # pre-processing
        image = pre_processing(image_ph, is_training_ph)
        label = tf.contrib.layers.one_hot_encoding(label_ph, num_label, on_value=1.0)
     
        # make global step
        global_step = tf.train.create_global_step()
        epoch = tf.floor_div(tf.cast(global_step, tf.float32)*batch_size, dataset_len)
        max_number_of_steps = int(dataset_len*train_epoch)//batch_size+1

        # make learning rate scheduler
        LR = learning_rate_scheduler(Learning_rate, [epoch, train_epoch], [0.3, 0.6, 0.8], 0.1)
        
        ## load Net
        with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.fully_connected], 
                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_OUT'),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                        biases_initializer=None, activation_fn = None,
                        ):
            with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm],
                                scale = True, center = True, activation_fn=tf.nn.relu, decay=0.9, epsilon = 1e-5,
                                param_regularizers={'gamma': tf.contrib.layers.l2_regularizer(weight_decay),
                                                    'beta' : tf.contrib.layers.l2_regularizer(weight_decay)},
                                variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES,'BN_collection']):
                
                conv = tf.contrib.layers.conv2d(image, 16, [5,5], 2, scope='conv0', trainable=True)
                conv = tf.contrib.layers.batch_norm(conv, scope='bn0', trainable = True, is_training=is_training_ph)
                conv = tf.contrib.layers.max_pool2d(conv, [2,2], scope = 'pool0')
                conv = tf.contrib.layers.conv2d(image, 32, [5,5], 2, scope='conv1', trainable=True)
                conv = tf.contrib.layers.batch_norm(conv, scope='bn1', trainable = True, is_training=is_training_ph)
                
                fc = tf.contrib.layers.flatten(conv)
                fc = tf.contrib.layers.fully_connected(fc , 256, biases_initializer = tf.zeros_initializer(),
                                                       trainable=True, scope = 'fc0')
                fc = tf.contrib.layers.dropout(fc, is_training = is_training_ph)
                logit = tf.contrib.layers.fully_connected(fc , label.get_shape().as_list()[-1],
                                                       biases_initializer = tf.zeros_initializer(),
                                                       trainable=True, scope = 'fc1')
                
                loss = tf.losses.softmax_cross_entropy(label,logit)
                accuracy = tf.contrib.metrics.accuracy(tf.to_int32(tf.argmax(logit, 1)), tf.to_int32(tf.argmax(label, 1)))
                
        variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        teacher_variables = tf.get_collection('Teacher')
        variables = list(set(variables)-set(teacher_variables))
        
        # make optimizer w/ learning rate scheduler
        optimize = tf.train.MomentumOptimizer(LR, 0.9, use_nesterov=True)
        
        # training main-task
        total_loss = loss + tf.add_n(tf.losses.get_regularization_losses())
        tf.summary.scalar('loss/total_loss', total_loss)
        gradients  = optimize.compute_gradients(total_loss, var_list = variables)
                        
        # merge update operators and make train operator
        update_ops.append(optimize.apply_gradients(gradients, global_step=global_step))
        update_op = tf.group(*update_ops)
        train_op = control_flow_ops.with_dependencies([update_op], total_loss, name='train_op')
        
        ## collect summary ops for plotting in tensorboard
        summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES), name='summary_op')
        
        ## make placeholder and summary op for training and validation results
        train_acc_place = tf.placeholder(dtype=tf.float32)
        val_acc_place   = tf.placeholder(dtype=tf.float32)
        val_summary = [tf.summary.scalar('accuracy/training_accuracy',   train_acc_place),
                       tf.summary.scalar('accuracy/validation_accuracy', val_acc_place)]
        val_summary_op = tf.summary.merge(list(val_summary), name='val_summary_op')
        
        ## start training
        train_writer = tf.summary.FileWriter('%s'%FLAGS.train_dir,graph,flush_secs=save_summaries_secs)
        config = ConfigProto()
        config.gpu_options.visible_device_list = gpu_num
        config.gpu_options.allow_growth=True
        
        val_itr = len(val_labels)//val_batch_size
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
          
            sum_train_accuracy = []; time_elapsed = []; total_loss = []
            idx = list(range(train_labels.shape[0]))
            shuffle(idx)
            epoch_ = 0
            for step in range(max_number_of_steps):
                start_time = time.time()
                
                ## feed data
                tl, log, train_acc = sess.run([train_op, summary_op, accuracy],
                                              feed_dict = {image_ph : train_images[idx[:batch_size]],
                                                           label_ph : np.squeeze(train_labels[idx[:batch_size]]),
                                                           is_training_ph : True})
    
                time_elapsed.append( time.time() - start_time )
                total_loss.append(tl)
                sum_train_accuracy.append(train_acc)
                idx[:batch_size] = []
                if len(idx) < batch_size:
                    idx_ = list(range(train_labels.shape[0]))
                    shuffle(idx_)
                    idx += idx_
                
                step += 1
                if (step*batch_size)//dataset_len>=epoch_:
                    ## do validation
                    sum_val_accuracy = []
                    for i in range(val_itr):
                        val_batch = val_images[i*val_batch_size:(i+1)*val_batch_size]
                        acc = sess.run(accuracy, feed_dict = {image_ph : val_batch,
                                                              label_ph : np.squeeze(val_labels[i*val_batch_size:(i+1)*val_batch_size]),
                                                              is_training_ph : False})
                        sum_val_accuracy.append(acc)
                        
                    sum_train_accuracy = np.mean(sum_train_accuracy)*100
                    sum_val_accuracy= np.mean(sum_val_accuracy)*100
                    tf.logging.info('Epoch %s Step %s - train_Accuracy : %.2f%%  val_Accuracy : %.2f%%'
                                    %(str(epoch_).rjust(3, '0'), str(step).rjust(6, '0'), 
                                    sum_train_accuracy, sum_val_accuracy))

                    result_log = sess.run(val_summary_op, feed_dict={train_acc_place : sum_train_accuracy,
                                                                     val_acc_place   : sum_val_accuracy   })
    
                    if step == max_number_of_steps:
                        train_writer.add_summary(result_log, train_epoch)
                    else:
                        train_writer.add_summary(result_log, epoch_)
                    sum_train_accuracy = []

                    epoch_ += 1
                    
                if step % should_log == 0:
                    tf.logging.info('global step %s: loss = %.4f (%.3f sec/step)',str(step).rjust(6, '0'), np.mean(total_loss), np.mean(time_elapsed))
                    train_writer.add_summary(log, step)
                    time_elapsed = []
                    total_loss = []
                
                elif (step*batch_size) % dataset_len == 0:
                    train_writer.add_summary(log, step)

            ## close all
            tf.logging.info('Finished training! Saving model to disk.')
            train_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.STOP))
            train_writer.close()

def pre_processing(image, is_training):
        with tf.variable_scope('preprocessing'):
            image = tf.cast(image, tf.float32)
            image = (image-np.array([112.4776,124.1058,129.3773]))/np.array([70.4587,65.4312,68.2094])
            def augmentation(image):
                image = tf.image.random_flip_left_right(image) # tf.__version__ > 1.10
                sz = tf.shape(image)
                image = tf.pad(image, [[0,0],[4,4],[4,4],[0,0]], 'REFLECT')
                image = tf.random_crop(image,sz)
                return image
            image = tf.cond(is_training, lambda : augmentation(image), lambda : image)
        return image
    
def learning_rate_scheduler(Learning_rate, epochs, decay_point, decay_rate):
    with tf.variable_scope('learning_rate_scheduler'):
        e, te = epochs
        for i, dp in enumerate(decay_point):
            Learning_rate = tf.cond(tf.greater_equal(e, int(te*dp)), lambda : Learning_rate*decay_rate, 
                                                                     lambda : Learning_rate)
        tf.summary.scalar('learning_rate', Learning_rate)
        return Learning_rate

if __name__ == '__main__':
    tf.app.run()


