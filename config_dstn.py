'''
config file
'''
n_one_hot_slot = 25 # num of one-hot slots in target ad
n_mul_hot_slot = 2 # num of mul-hot slots in target ad
max_len_per_slot = 5 # max num of fts per mul-hot ft slot in target ad
num_aux_type = 3 # num of types of aux ads£¬e.g., if the types include ctxt/clk/nonclk£¬then num_aux_type=3
n_one_hot_slot_aux = [25, 25, 25] # nums of one-hot slots in each type of aux ads; nums can be different; len(n_one_hot_slot_aux) should equals num_aux_type
n_mul_hot_slot_aux = [2, 2, 2] # nums of mul-hot slots in each type of aux ads; nums can be different
max_len_per_slot_aux = [5, 5, 5] # max nums of fts per mul-hot ft slot in each type of aux ads, nums can be different
num_aux_inst_in_data = [5, 5, 5] # nums of insts of each type of aux ads (per row) in data£¨fixed; depends on your data
max_num_aux_inst_used = [3, 3, 3] # nums of insts of each type of aux ads actually used in experiments£¨can be tuned for experiments

n_ft = 42301586 # num of unique fts
num_csv_col = 561 # num of cols in the csv file

pre = './data/'
train_file_name = [pre+'day_1.csv', pre+'day_2.csv'] # can contain multiple file names
val_file_name = [pre+'day_3.csv'] # should contain only 1 file name
test_file_name = [pre+'day_4.csv'] # should contain only 1 file name

time_style = '%Y-%m-%d %H:%M:%S'
output_file_name = '0508_1541' # part of file and folder names for recording the output model and result
k = 10 # embedding dim for each ft
eta = 0.01 # learning rate
batch_size = 128 # mini batch size
kp_prob = 1.0 # keep prob in dropout; set to 1.0 if n_epoch = 1
opt_alg = 'Adagrad' # 'Adagrad' or 'Adam'
att_hidden_dim = 128 # att hidden layer dim
max_num_lower_ct = 100 # early stop if the metric does not improve over the validation set after max_num_lower_ct times 
n_epoch = 1 # num of times to loop over the training set
record_step_size = 200 # record auc and loss on the validation set after record_step_size times of mini_batch
layer_dim = [512, 256, 1] # FC layer dimes; the last num must be 1 (the output layer)
