# Data Processing Pipeline

* Original data: Each line contains only the target ad and the target user

* Processed data: augmented with contextual ads, user clicked ads, and user unclicked ads

## Avito data csv files (processed)
* Each row of the Avito data csv file contains:\
\<target label\>\<target one-hot fts\>\<target mul-hot fts\> \
\<ctxt1 one-hot fts, ctxt2 one-hot fts, ..., ctxt5 one-hot fts\> \
\<ctxt1 multi-hot fts, ctxt2 multi-hot fts, ..., ctxt5 multi-hot fts\> \
\<clk1 one-hot fts, clk2 one-hot fts, ..., clk5 one-hot fts\> \
\<clk1 multi-hot fts, clk2 multi-hot fts, ..., clk5 multi-hot fts\> \
\<unclk1 one-hot fts, unclk2 one-hot fts, ..., unclk5 one-hot fts\> \
\<unclk1 multi-hot fts, unclk2 multi-hot fts, ..., unclk5 multi-hot fts\>

* There are n_one_hot_slot = 25 one-hot fts and n_mul_hot_slot = 2 multi-hot fts (with max_len_per_slot = 5; i.e., each multi-hot ft has 5 values; e.g., the multi-hot ft "search_params" may contain param a, param b, ..., param e) for each ad.
Therefore, there are totoally 1 + (25 + 2\*5)\*(1+5+5+5) = 561 columns in the csv file.

* The 25 one_hot fts are:
bias, ad_id, position, ip_id, user_id, is_user_logged_on, search_query_keyword, search_loc_id, search_loc_level, search_region_id, search_city_id, search_cate_id, search_cate_level, search_par_cate_id, search_sub_cate_id, user_agent_id, user_agent_os_id, user_device_id, user_agent_family_id, ad_title_keyword, ad_cate_id, ad_cate_level, ad_parent_cate_id, ad_sub_cate_id, hist_ctr_bin

* The 2 multi-hot fts are:
search_params, ad_params

## Alternative data formats
You can define your own data formats. For example, \
\<target label\>\<target one-hot fts\>\<target mul-hot fts\> \
\<ctxt1 one-hot fts, ctxt1 multi-hot fts\> \
\<ctxt2 one-hot fts, ctxt2 multi-hot fts\> \
... \
\<ctxt5 one-hot fts, ctxt5 multi-hot fts\> ...

After that, you only need to revise the "data_partition" function in the script.

## Spatial and temporal information
* Spatial information: Each line must contain search_id (or session_id) and ad_position
* Temporal information: Each line must contain user_id and time_stamp
 
### For spatial information: two prefixes should be added to each line: search_id, ad_position
* First sort data by search_id, then by ad_position [e.g., you can use Hadoop]
* Then for each target ad, the ads with the same search_id, but smaller ad_position are its contextual ads

* Example: \
**ori data** \
#search_id, ad_position, label, ad_id \
001, 1, 0, a \
001, 2, 1, b \
001, 3, 0, c \
002, 1, 0, d \
002, 2, 0, a \
**with contextual ads (max 2)** \
001, 1, 0, a, 0, 0 \
001, 2, 1, b, a, 0 \
001, 3, 0, c, b, a \
002, 1, 0, d, 0, 0 \
002, 2, 0, a, d, 0

* **Afterwards, remove (or do not print out) the two prefixes search_id & ad_position**

### For temporal information: two prefixes should be added to each line: user_id, time_stamp
* First sort data by user_id, then by time_stamp [e.g., you can use Hadoop]
* Then for each target ad, the ads with the same user_id, but smaller time_stamp and label=1 are user clicked ads
* The ads with the same user_id, but smaller time_stamp and label=0 are user unclicked ads
* Note: contextual ads will not become clicked / unclicked ads because contextual ads and the target ad have the same search_id and thus the same (not smaller) search time_stamp

* Example: \
**ori data** \
#user_id, time_stamp, label, ad_id \
u1, 201906011201, 0, p \
u1, 201906011202, 1, q \
u1, 201906011203, 0, r \
u2, 201906011201, 1, s \
u2, 201906011202, 0, t \
**with clicked ads (max 2) and unclicked ads (max 2)** \
u1, 201906011201, 0, p, 0, 0, 0, 0 \
u1, 201906011202, 1, q, 0, 0, p, 0 \
u1, 201906011203, 0, r, q, 0, p, 0 \
u2, 201906011201, 1, s, 0, 0, 0, 0 \
u2, 201906011202, 0, t, s, 0, 0, 0

* **Afterwards, remove (or do not print out) the two prefixes user_id & time_stamp**

You may need to use time_stamp for data splitting as well.

## Data Processing Pipeline
ori_data -> \
[search_id, ad_position,] ori_data -> \
[search_id, ad_position,] ori_data, contextual ads -> \
ori_data, contextual ads -> \
[user_id, time_stamp,] ori_data, contextual ads -> \
[user_id, time_stamp,] ori_data, contextual ads, clicked ads, unclicked ads -> \
ori_data, contextual ads, clicked ads, unclicked ads
