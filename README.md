# ARNN
PyTorch implementation of AAAI-20 paper-ARNN: An Attentional Recurrent Neural Networkfor Personalized Next Location Recommendation. [link](https://ojs.aaai.org/index.php/AAAI/article/view/5337)

Note: I'm not among the authors of the paper ARNN. Without source code from the authors of ARNN, I can only reproduct this model based on DeepMove, which is a similar model. So if there are any errors, please contact me.

# Datasets
The foursquare data to evaluate our model can be found in the data folder, which contains 1000+ users and is real-world datasets.

# Requirements
- Python 3.6
- Pytorch 1.7.0
- cudatoolkit 9.2

# Project Structure
- /codes
    - data_pre_with_category.py
    - generate_graph.py
    - main.py
    - model.py
    - train.py
    - utils.py
- /data
    - dataset_TSMC2014_NYC.txt  
    - foursquare_NYC_4input.pkl
    - nyc_4input.pkl   
    - paths_NYC.pkl
    - triple_pc.txt
    - triple_plg.txt
    - triple_ptp.txt
    - triple_utp.txt
    - relation_category_id.txt
    - relation_loc_id.txt
    - relation_td_id.txt
    - relation_time_id.txt
    - entity_category_name_id.txt
    - entity_grid_id.txt
    - entity_loc_dict.txt
    - entity_user_dict.txt
- /resutls
    - /checkpoint
        - ep_i.m

# Usage
1. Prepare the session data and KG data:
> ```python
> python data_pre_with_category.py
> ```

In this part, we conduct the raw data to filter and form check-in sessions and spatial-temproal-category triples.

2. Discover neighbors:
> ```python
> python generate_graph.py
> ```

Using meta-path based random walk method to discover every location's plenty neighbors. You can choose the meta-path you need within the code. You can also determine the length of the paths.

And it may take some time to finish.

Output file: 
paths_NYC.pkl

2. Train a new model:
> ```python
> python main.py
> ```

The parameters are already set in the code.

Other parameters:
- for training: 
    - learning_rate, lr_step, lr_decay, L2, clip, epoch_max, dropout_p
- model definition: 
    - loc_emb_size, uid_emb_size, tim_emb_size, word_emb_size, hidden_size, neighbors_num, rnn_type, attn_type
    - history_mode: avg, avg, whole

# Author
Haoyu Huang

E-mail: haoyuhuang@bjtu.edu.cn

# References
The modle was implemented base on the codes of DeepMove.[code link](https://github.com/vonfeng/DeepMove)