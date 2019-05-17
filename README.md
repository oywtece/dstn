# Deep Spatio-Temporal Neural Network (DSTN)

DSTN is a model for click-through rate (CTR) prediction. DSTN investigates various types of auxiliary ads for improving the CTR prediction of the target ad.

The auxiliary ads are from two viewpoints: one is from the spatial domain, where DSTN considers the contextual ads shown above the target ad on the same page; the other is from the temporal domain, where DSTN considers historically clicked and unclicked ads of the user. The intuitions are that ads shown together may influence each other, clicked ads reflect a user’s possible preferences, and unclicked ads may indicate what a user dislikes to certain extent.

If you use this code, please cite the following paper:
* **Wentao Ouyang, Xiuwu Zhang, Li Li, Heng Zou, Xin Xing, Zhaojie Liu, Yanlong Du. 2019. Deep Spatio-Temporal Neural Networks for Click-Through Rate Prediction. In KDD. ACM, xx-xx.**

#### TensorFlow (TF) version
1.3.0

#### Abbreviation
ft - feature, ctxt - contextual, clk - clicked, unclk - unclicked, slot == field

## Data Preparation
Data is in the "csv" format.
* Each row contains one target ad and several auxiliary ads (e.g., include ctxt/clk/unclk ads).
* Each ad is represented as a set of fts (e.g., include the user/query/ad/context fts).

Assume there are N unique fts. Fts need to be indexed from 1 to N. Use 0 for missing values or for padding.

We categorize fts as i) **one-hot** or **univalent** (e.g., user id, city) and ii) **mul-hot** or **multivalent** (e.g., words in ad title).

Assume we have at most 2 contextual ads, 2 clicked ads and 2 unclicked ads, then one row of the csv data looks like:
* \<label\>\<target one-hot fts\>\<target mul-hot fts\>\<ctxt1 one-hot fts, ctxt2 one-hot fts\>\<ctxt1 multi-hot fts, ctxt2 multi-hot fts\>\<clk1 one-hot fts, clk2 one-hot fts\>\<clk1 multi-hot fts, clk2 multi-hot fts\>\<unclk1 one-hot fts, unclk2 one-hot fts\>\<unclk1 multi-hot fts, unclk2 multi-hot fts\>

We also need to define the max number of features per mul-hot ft slot (through the "max_len_per_slot" parameter) and perform trimming or padding accordingly. Please refer to the following examples for more detail.

### Example 1 (target ad)
1) original fts (ft_name:ft_value)
* label:0, gender:male, age:27, query:apple, title:apple, title:fruit, title:fresh, title:cheap
2) csv fts
* 0, male, 27, apple, 0, 0, apple, fruit, fresh

#### Explanation 1
csv format:\
\<label\>\<target one-hot fts\>\<target mul-hot fts\>

csv format settings:\
n_one_hot_slot = 2 # num of one-hot ft slots (gender, age)\
n_mul_hot_slot = 2 # num of mul-hot ft slots (query, title)\
max_len_per_slot = 3 # max num of fts per mul-hot ft slot

For the mul-hot ft slot "query", we have only 1 ft, which is "apple". Terefore, we pad 2 zeros (because max_len_per_slot = 3). The resulting 3 fts are "apple, 0, 0".

For the mul-hot ft slot "title", we have 4 fts, which are "apple, fruit, fresh, cheap". Therefore, we keep only the first 3 fts (because max_len_per_slot = 3). The resulting 3 fts are "apple, fruit, fresh".

### Example 2 (ctxt ads)
1) original fts (ft_name:ft_value)
* ctxt1 - gender:female, age:31, query:pear, query:orange, title:pear, title:fruit, title:fresh
* ctxt2 - gender:male, age:17, query:cherry, title:cherry, title:fruit
2) csv fts
* female, 31, male, 17, 0, 0, pear, orange, 0, pear, fruit, fresh, cherry, 0, 0, cherry, fruit, 0, 0, 0, 0, 0, 0, 0

#### Explanation 2
csv format:\
\<ctxt1 one-hot fts, ctxt2 one-hot fts, ctxt3 one-hot fts\>\<ctxt1 mul-hot fts, ctxt2 mul-hot fts, ctxt3 mul-hot fts\>

csv format settings:\
n_one_hot_slot_aux = 2 # num of one-hot ft slots (gender, age)\
n_mul_hot_slot_aux = 2 # num of mul-hot ft slots (query, title)\
max_len_per_slot_aux = 3 # max num of fts per mul-hot ft slot\
num_aux_inst_in_data = 3 # max num of ctxt ads per target ad

One-hot fts:\
Because we set "num_ctxt_in_data = 3", but we have only 2 ctxt ads, we then pad n_one_hot_slot_aux zeros for the missing ctxt3.

Mul-hot fts:\
For ctxt1, the resulting 3 fts for the mul-hot ft slot "query" are "pear, orange, 0".
The resulting 3 fts for the mul-hot ft slot "title" are "pear, fruit, fresh". \
For ctxt2, the resulting 3 fts for the mul-hot ft slot "query" are "cherry, 0, 0".
The resulting 3 fts for the mul-hot ft slot "title" are "cherry, fruit, 0,".
Because we set "num_ctxt_in_data = 3", but we have only 2 ctxt ads, we then padding max_len_per_slot_aux zeros for "query" and max_len_per_slot_aux zeros for "title" for the missing ctxt3, resulting in "0, 0, 0, 0, 0, 0".

## Sample Data
In the "data" folder.\
Sampled and reformatted Avito data (csv format with ft index) of 4 days are provided (only for demonstration purpose; the whole dataset is huge). \
Because the amount of data is small, the test results may not be statistically reliable.
* [Full Avito data](https://www.kaggle.com/c/avito-context-ad-clicks/data)

## Source Code
* config_dstn.py -- config file
* ctr_funcs.py -- functions
* dnn.py -- DNN model
* dstn_pooling.py -- DSTN - pooling model
* dstn_self_att.py -- DSTN - self-attention model
* dstn_int_att.py -- DSTN - interactive attention model

## Run the Code
First revise the config file, and then run the code
```bash
nohup python dstn_pooling.py > [output_file_name] 2>&1 &
```
