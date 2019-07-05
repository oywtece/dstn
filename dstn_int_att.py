import numpy as np
import tensorflow as tf
import datetime
import ctr_funcs as func
import config_dstn as cfg
import os
import shutil

# config
str_txt = cfg.output_file_name
base_path = './tmp'
model_saving_addr = base_path + '/dstn_i_' + str_txt + '/'
output_file_name = base_path + '/dstn_i_' + str_txt + '.txt'
num_csv_col = cfg.num_csv_col
train_file_name = cfg.train_file_name
val_file_name = cfg.val_file_name
test_file_name = cfg.test_file_name
batch_size = cfg.batch_size
n_ft = cfg.n_ft
k = cfg.k
eta = cfg.eta
kp_prob = cfg.kp_prob
n_epoch = cfg.n_epoch
max_num_lower_ct = cfg.max_num_lower_ct
record_step_size = cfg.record_step_size
layer_dim = cfg.layer_dim
opt_alg = cfg.opt_alg
n_one_hot_slot = cfg.n_one_hot_slot
n_mul_hot_slot = cfg.n_mul_hot_slot
num_aux_type = cfg.num_aux_type
n_one_hot_slot_aux = cfg.n_one_hot_slot_aux
n_mul_hot_slot_aux = cfg.n_mul_hot_slot_aux
max_len_per_slot_aux = cfg.max_len_per_slot_aux
num_aux_inst_in_data = cfg.num_aux_inst_in_data
max_num_aux_inst_used = cfg.max_num_aux_inst_used
max_len_per_slot = cfg.max_len_per_slot
att_hidden_dim = cfg.att_hidden_dim

label_col_idx = 0
record_defaults = [[0]]*num_csv_col
record_defaults[0] = [0.0]
total_num_ft_col = num_csv_col - 1

# create dir
if not os.path.exists(base_path):
    os.mkdir(base_path)

# remove dir
if os.path.isdir(model_saving_addr):
    shutil.rmtree(model_saving_addr)

###########################################################
###########################################################
print('Loading data start!')
tf.set_random_seed(123)

# load training data
train_ft, train_label = func.tf_input_pipeline(train_file_name, batch_size, n_epoch, label_col_idx, record_defaults)

# load val data
n_val_inst = func.count_lines(val_file_name[0])
val_ft, val_label = func.tf_input_pipeline(val_file_name, n_val_inst, 1, label_col_idx, record_defaults)
n_val_batch = n_val_inst//batch_size

# load test data
test_ft, test_label = func.tf_input_pipeline_test(test_file_name, batch_size, 1, label_col_idx, record_defaults)
print('Loading data done!')

########################################################################
def partition_input(x_input):
    # generate idx_list
    len_list = []
    len_list.append(n_one_hot_slot)
    len_list.append(n_mul_hot_slot*max_len_per_slot)
    
    for i in range(num_aux_type):
        len_list.append(n_one_hot_slot_aux[i]*num_aux_inst_in_data[i])
        len_list.append(n_mul_hot_slot_aux[i]*max_len_per_slot_aux[i]*num_aux_inst_in_data[i])
    
    len_list = np.array(len_list)
    idx_list = np.cumsum(len_list)

    # shape=[None, n_one_hot_slot]
    x_input_one_hot = x_input[:, 0:idx_list[0]]
    x_input_mul_hot = x_input[:, idx_list[0]:idx_list[1]]
    # shape=[None, n_mul_hot_slot, max_len_per_slot]
    x_input_mul_hot = tf.reshape(x_input_mul_hot, (-1, n_mul_hot_slot, max_len_per_slot))
    
    # aux
    x_input_one_hot_aux = {}
    x_input_mul_hot_aux = {}
    for i in range(num_aux_type):
        # take out
        temp_1 = x_input[:, idx_list[2*i+1]:idx_list[2*i+2]]
        # reshape
        temp_1 = tf.reshape(temp_1, (-1, num_aux_inst_in_data[i], n_one_hot_slot_aux[i]))
        # shape=[None, max_num_ctxt, n_one_hot_slot]
        x_input_one_hot_aux[i] = temp_1[:, 0:max_num_aux_inst_used[i], :]
        # take out
        temp_2 = x_input[:, idx_list[2*i+2]:idx_list[2*i+3]]
        temp_2 = tf.reshape(temp_2, (-1, num_aux_inst_in_data[i], n_mul_hot_slot_aux[i], \
                                     max_len_per_slot_aux[i]))
        # shape=[None, max_num_ctxt, n_mul_hot_slot, max_len_per_slot]
        x_input_mul_hot_aux[i] = temp_2[:, 0:max_num_aux_inst_used[i], :, :]
        
    return x_input_one_hot, x_input_mul_hot, x_input_one_hot_aux, x_input_mul_hot_aux

# add mask
def get_masked_one_hot(x_input_one_hot):
    data_mask = tf.cast(tf.greater(x_input_one_hot, 0), tf.float32)
    data_mask = tf.expand_dims(data_mask, axis = 2)
    data_mask = tf.tile(data_mask, (1,1,k))
    # output: (?, n_one_hot_slot, k)
    data_embed_one_hot = tf.nn.embedding_lookup(emb_mat, x_input_one_hot)
    data_embed_one_hot_masked = tf.multiply(data_embed_one_hot, data_mask)
    return data_embed_one_hot_masked

def get_masked_mul_hot(x_input_mul_hot):
    data_mask = tf.cast(tf.greater(x_input_mul_hot, 0), tf.float32)
    data_mask = tf.expand_dims(data_mask, axis = 3)
    data_mask = tf.tile(data_mask, (1,1,1,k))
    # output: (?, n_mul_hot_slot, max_len_per_slot, k)
    data_embed_mul_hot = tf.nn.embedding_lookup(emb_mat, x_input_mul_hot)
    data_embed_mul_hot_masked = tf.multiply(data_embed_mul_hot, data_mask)
    return data_embed_mul_hot_masked

def get_masked_one_hot_aux(x_input_one_hot_ctxt):
    data_mask = tf.cast(tf.greater(x_input_one_hot_ctxt, 0), tf.float32)
    data_mask = tf.expand_dims(data_mask, axis = 3)
    data_mask = tf.tile(data_mask, (1,1,1,k))
    # output: (?, max_num_ctxt, n_one_hot_slot, k)
    data_embed_one_hot = tf.nn.embedding_lookup(emb_mat, x_input_one_hot_ctxt)
    data_embed_one_hot_masked = tf.multiply(data_embed_one_hot, data_mask)
    return data_embed_one_hot_masked

def get_masked_mul_hot_aux(x_input_mul_hot_ctxt):
    data_mask = tf.cast(tf.greater(x_input_mul_hot_ctxt, 0), tf.float32)
    data_mask = tf.expand_dims(data_mask, axis = 4)
    data_mask = tf.tile(data_mask, (1,1,1,1,k))
    # output: (?, n_mul_hot_slot, max_len_per_slot, k)
    data_embed_mul_hot = tf.nn.embedding_lookup(emb_mat, x_input_mul_hot_ctxt)
    data_embed_mul_hot_masked = tf.multiply(data_embed_mul_hot, data_mask)
    return data_embed_mul_hot_masked

def prepare_input_embed(x_input_one_hot, x_input_mul_hot):
    # output: (?, n_one_hot_slot, k)
    data_embed_one_hot = get_masked_one_hot(x_input_one_hot)
    data_embed_one_hot = tf.reshape(data_embed_one_hot, [-1, n_one_hot_slot*k])   
    # output: (?, n_mul_hot_slot, max_len_per_slot, k)
    data_embed_mul_hot = get_masked_mul_hot(x_input_mul_hot)
    data_embed_mul_hot_pooling = tf.reduce_sum(data_embed_mul_hot, 2)
    data_embed_mul_hot_pooling = tf.reshape(data_embed_mul_hot_pooling, [-1, n_mul_hot_slot*k])
    # concatenate (col-wise; keep num of rows unchanged)
    data_embed_ori = tf.concat([data_embed_one_hot, data_embed_mul_hot_pooling], 1)
    return data_embed_ori

##################################
# should keep max_num_ctxt dim
def prepare_input_embed_aux_interaction(x_input_one_hot_ctxt, x_input_mul_hot_ctxt, \
                                        max_num_ctxt, cur_n_one_hot_slot, cur_n_mul_hot_slot):
    # output: (?, max_num_ctxt, n_one_hot_slot, k)
    data_embed_one_hot_ctxt = get_masked_one_hot_aux(x_input_one_hot_ctxt)
    # output: (?, max_num_ctxt, n_mul_hot_slot, max_len_per_slot, k)
    data_embed_mul_hot_ctxt = get_masked_mul_hot_aux(x_input_mul_hot_ctxt)
    # if max_num_ctxt = 1, then this dim will be automatically collapsed
    data_embed_mul_hot_pooling_ctxt = tf.reduce_sum(data_embed_mul_hot_ctxt, 3) 
    data_embed_one_hot_ctxt = tf.reshape(data_embed_one_hot_ctxt, \
                                                 [-1, max_num_ctxt, cur_n_one_hot_slot*k])
    data_embed_mul_hot_pooling_ctxt = tf.reshape(data_embed_mul_hot_pooling_ctxt, \
                                                 [-1, max_num_ctxt, cur_n_mul_hot_slot*k])
    # output dim: none * max_num_ctxt * (n_one_hot_slot + n_mul_hot_slot)k
    data_embed_ctxt = tf.concat([data_embed_one_hot_ctxt, data_embed_mul_hot_pooling_ctxt], 2)
    return data_embed_ctxt

########################################
# data_embed_ori - none * total_embed_dim
# data_embed_ctxt - none* max_num_ctxt * total_embed_dim
def get_wgt_sum_embed_aux(data_embed_ori, data_embed_ctxt, W1_ctxt, b1_ctxt, W2_ctxt, b2_ctxt, \
                          max_num_ctxt, total_embed_dim_ctxt):
    # dim: none * 1 * total_embed_dim
    data_embed_ori_exp = tf.expand_dims(data_embed_ori, 1)
    # tile, dim: none * max_num_ctxt * total_embed_dim
    data_embed_ori_tile = tf.tile(data_embed_ori_exp, [1, max_num_ctxt, 1])
    # concat, dim: none * max_num_ctxt * 2 total_embed_dim
    data_concat = tf.concat([data_embed_ori_tile, data_embed_ctxt], 2)
    data_concat = tf.reshape(data_concat, [-1, total_embed_dim_ctxt])
    # dim: none * max_num_ctxt * att_hidden_dim
    hidden = tf.matmul(data_concat, W1_ctxt) + b1_ctxt
    hidden = tf.nn.relu(hidden)
    hidden = tf.nn.dropout(hidden, keep_prob)
    # dim: none * max_num_ctxt * 1 [must have 1 at the last dim]
    wgt_ctxt = tf.exp(tf.matmul(hidden, W2_ctxt) + b2_ctxt)
    wgt_ctxt = tf.reshape(wgt_ctxt, [-1, max_num_ctxt, 1])
    # dim: none * max_num_ctxt * total_embed_dim
    temp = wgt_ctxt * data_embed_ctxt
    # sum over dim max_num_ctxt
    # dim: none * total_embed_dim (same dim as data_embed_ori)
    output = tf.reduce_sum(temp, 1)
    return output
    
###########################################################
# input for DNN (embedding ids)
x_input = tf.placeholder(tf.int32, shape=[None, total_num_ft_col])

x_input_one_hot, x_input_mul_hot, x_input_one_hot_aux, x_input_mul_hot_aux \
    = partition_input(x_input)

# target vect
y_target = tf.placeholder(tf.float32, shape=[None, 1])
# dropout keep prob
keep_prob = tf.placeholder(tf.float32)

# emb_mat dim add 1 -> for padding (idx = 0)
with tf.device('/cpu:0'):
    emb_mat = tf.Variable(tf.random_normal([n_ft + 1, k], stddev=0.01))

# attention weight
W1_list = {}; b1_list = {}; W2_list = {}; b2_list = {}
total_embed_dim = {}
for i in range(num_aux_type):
    total_embed_dim[i] = k*(n_one_hot_slot + n_mul_hot_slot \
                          + n_one_hot_slot_aux[i] + n_mul_hot_slot_aux[i])
    std_a = np.sqrt(2.0/(total_embed_dim[i]+att_hidden_dim))
    std_b = np.sqrt(2.0/att_hidden_dim)
    W1_list[i] = tf.Variable(tf.random_normal([total_embed_dim[i], att_hidden_dim], \
                    stddev=std_a))
    b1_list[i] = tf.Variable(tf.random_normal([att_hidden_dim], stddev=std_b))
    W2_list[i] = tf.Variable(tf.random_normal([att_hidden_dim, 1], stddev=std_b))
    b2_list[i] = tf.Variable(tf.random_normal([1], stddev=0.01))

####### DNN part: ori ########
data_embed_ori = prepare_input_embed(x_input_one_hot, x_input_mul_hot)

# ####### DNN part: ctxt, clk, non_clk ########
# ####### interaction (data_embed_ori, data_embed_aux) ########
data_embed_aux = {}
wgt_sum_embed_aux = {}
for i in range(num_aux_type):
    data_embed_aux[i] = prepare_input_embed_aux_interaction(x_input_one_hot_aux[i], \
                        x_input_mul_hot_aux[i], max_num_aux_inst_used[i], \
                        n_one_hot_slot_aux[i], n_mul_hot_slot_aux[i])
    wgt_sum_embed_aux[i] = get_wgt_sum_embed_aux(data_embed_ori, data_embed_aux[i], \
                        W1_list[i], b1_list[i], W2_list[i], b2_list[i], \
                        max_num_aux_inst_used[i], total_embed_dim[i])
# ################################
# big concatenation
data_embed = tf.concat([data_embed_ori, wgt_sum_embed_aux[0]], 1)
for i in range(1, len(data_embed_aux)):
    data_embed = tf.concat([data_embed, wgt_sum_embed_aux[i]], 1)
    
################################
# include output layer
n_layer = len(layer_dim)
 
cur_layer = data_embed
 
data_embed_shape = data_embed.get_shape().as_list()
in_dim = data_embed_shape[1]
# loop to create DNN struct
for i in range(0, n_layer):
    out_dim = layer_dim[i]
    weight = tf.Variable(tf.random_normal(shape=[in_dim, out_dim], stddev=np.sqrt(2.0/(in_dim+out_dim))))
    bias = tf.Variable(tf.constant(0.1, shape=[out_dim]))
    # output layer, linear activation
    if i == n_layer - 1:
        cur_layer = tf.matmul(cur_layer, weight) + bias
    else:
        cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
        cur_layer = tf.nn.dropout(cur_layer, keep_prob)
    in_dim = layer_dim[i]
 
y_hat = cur_layer
 
# log loss
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=y_target))
pred_score = tf.sigmoid(y_hat)

if opt_alg == 'Adam':
    optimizer = tf.train.AdamOptimizer(eta).minimize(loss)
else:
    # default
    optimizer = tf.train.AdagradOptimizer(eta).minimize(loss)

########################################
# Launch the graph.
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    func.print_time()
    print('Load val data')

    # load val data
    val_ft_inst, val_label_inst = sess.run([val_ft, val_label])
    print('Done loading eval data')

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()    
    train_loss_list = []
    val_avg_auc_list = []
    epoch_list = []
    best_n_round = 0
    best_val_avg_auc = 0
    early_stop_flag = 0
    lower_ct = 0   

    func.print_time()
    print('Start train loop')
    
    epoch = -1
    try:
        while not coord.should_stop():           
            epoch += 1  
            train_ft_inst, train_label_inst = sess.run([train_ft, train_label])
            train_label_inst = np.transpose([train_label_inst])            
            
            sess.run(optimizer, feed_dict={x_input:train_ft_inst, \
                                           y_target:train_label_inst, keep_prob:kp_prob})
    
            # record loss and accuracy every step_size generations
            if (epoch+1)%record_step_size == 0:
                epoch_list.append(epoch)
                train_loss_temp = sess.run(loss, feed_dict={ \
                                           x_input:train_ft_inst, \
                                           y_target:train_label_inst, keep_prob:1})
                train_loss_list.append(train_loss_temp)
                 
                val_pred_score_all = []
                val_label_all = []
                
                for iii in range(n_val_batch):
                    # get batch
                    start_idx = iii*batch_size
                    end_idx = (iii+1)*batch_size
                    cur_val_ft = val_ft_inst[start_idx: end_idx]
                    cur_val_label = val_label_inst[start_idx: end_idx]
                    # pred score
                    cur_val_pred_score = sess.run(pred_score, feed_dict={ \
                                            x_input:cur_val_ft, keep_prob:1})
                    val_pred_score_all.append(cur_val_pred_score.flatten())
                    val_label_all.append(cur_val_label)   
                    
                # calculate auc
                val_pred_score_re = func.list_flatten(val_pred_score_all)
                val_label_re = func.list_flatten(val_label_all)
                val_auc_temp, _, _ = func.cal_auc(val_pred_score_re, val_label_re)
                # record all val results    
                val_avg_auc_list.append(val_auc_temp)
                 
                # record best and save models
                if val_auc_temp > best_val_avg_auc:
                    best_val_avg_auc = val_auc_temp
                    best_n_round = epoch
                    # Save the variables to disk
                    save_path = saver.save(sess, model_saving_addr)
                    print("Model saved in file: %s" % save_path)
                # count of consecutive lower
                if val_auc_temp < best_val_avg_auc:
                     lower_ct += 1
                # once higher or equal, set to 0
                else:
                     lower_ct = 0
                
                if lower_ct >= max_num_lower_ct:
                    early_stop_flag = 1
                
                auc_and_loss = [epoch+1, train_loss_temp, val_auc_temp]
                auc_and_loss = [np.round(xx,4) for xx in auc_and_loss]
                func.print_time() 
                print('Generation # {}. Train Loss: {:.4f}. Val Avg AUC: {:.4f}.'\
                      .format(*auc_and_loss))
            if early_stop_flag == 1:
                break
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    
    # after training
    saver.restore(sess, model_saving_addr)
    print("Model restored.")
     
    # load test data
    test_pred_score_all = []
    test_label_all = []
    test_loss_all = []
    try:
        while True:
            test_ft_inst, test_label_inst = sess.run([test_ft, test_label])
            cur_test_pred_score = sess.run(pred_score, feed_dict={ \
                                    x_input:test_ft_inst, keep_prob:1})
            test_pred_score_all.append(cur_test_pred_score.flatten())
            test_label_all.append(test_label_inst)
            
            cur_test_loss = sess.run(loss, feed_dict={ \
                                    x_input:test_ft_inst, \
                                    y_target: np.transpose([test_label_inst]), keep_prob:1})
            test_loss_all.append(cur_test_loss)

    except tf.errors.OutOfRangeError:
        print('Done loading testing data -- epoch limit reached')    
    finally:
        coord.request_stop()
        
    coord.join(threads) 
         
    # calculate auc
    test_pred_score_re = func.list_flatten(test_pred_score_all)
    test_label_re = func.list_flatten(test_label_all)
    test_auc, _, _ = func.cal_auc(test_pred_score_re, test_label_re)
    test_rmse = func.cal_rmse(test_pred_score_re, test_label_re)
    test_loss = np.mean(test_loss_all)
    
    # rounding
    test_auc = np.round(test_auc, 4)
    test_rmse = np.round(test_rmse, 4)
    test_loss = np.round(test_loss, 5)
    train_loss_list = [np.round(xx,4) for xx in train_loss_list]
    val_avg_auc_list = [np.round(xx,4) for xx in val_avg_auc_list]
      
    print('test_auc = ', test_auc)
    print('test_rmse =', test_rmse)
    print('test_loss =', test_loss)
    print('train_loss_list =', train_loss_list)
    print('val_avg_auc_list =', val_avg_auc_list)
      
    # write output to file
    with open(output_file_name, 'a') as f:
        now = datetime.datetime.now()
        time_str = now.strftime(cfg.time_style)
        f.write(time_str + '\n')
        f.write('train_file_name = ' + train_file_name[0] + '\n')
        f.write('learning_rate = ' + str(eta) + ', n_epoch = ' + str(n_epoch) \
                + ', emb_dize = ' + str(k) + '\n')
        f.write('test_auc = ' + str(test_auc) + '\n')
        f.write('test_rmse = ' + str(test_rmse) + '\n')
        f.write('test_loss = ' + str(test_loss) + '\n')
        f.write('train_loss_list =' + str(train_loss_list) + '\n')
        f.write('val_avg_auc_list =' + str(val_avg_auc_list) + '\n')
        f.write('-'*50 + '\n')

