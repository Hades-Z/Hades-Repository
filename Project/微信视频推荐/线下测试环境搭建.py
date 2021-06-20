#%% md

### 环境初始化

#%%

import os
import numpy as np
import pandas as pd


#%%

# 存储数据的根目录
ROOT_PATH = './data'
# 比赛数据集路径
DATASET_PATH = os.path.join(ROOT_PATH, 'wechat_algo_data1')
# 训练集
USER_ACTION = os.path.join(DATASET_PATH, 'user_action.csv')
FEED_INFO = os.path.join(DATASET_PATH, 'feed_info.csv')
FEED_EMBEDDINGS = os.path.join(DATASET_PATH, 'feed_embeddings.csv')
# 测试集
TEST_FILE = os.path.join(DATASET_PATH, 'test_a.csv')


#%%

# 视频的单值特征
feed_simple_features = ['feedid','authorid','videoplayseconds','bgm_song_id','bgm_singer_id']

# 用户行为特征
action_features = ['userid','feedid','device']

# 初赛待预测行为列表
action_lables = ['read_comment', 'like', 'click_avatar','forward']
# 复赛待预测行为列表
# action_lables = ['read_comment', 'like', 'click_avatar','forward','favorite','comment','follow']

# 初赛提交字段
submit_fields = ['userid','feedid','read_comment','like','click_avatar','forward']
# 复赛提交字段
# submit_fields = ['userid','feedid','read_comment','like','click_avatar','forward','favorite','comment','follow']


#%%
# 数据读取
feed_info = pd.read_csv(FEED_INFO, dtype={'feedid': str}); print(feed_info.info(),'\n')
user_action = pd.read_csv(USER_ACTION, dtype={'userid': str,'feedid': str,'device': str}); print(user_action.info(),'\n')
feed_embed = pd.read_csv(FEED_EMBEDDINGS, dtype={'feedid': str}); print(feed_embed.info(),'\n')
test = pd.read_csv(TEST_FILE, dtype={'userid': str,'feedid': str,'device': str}); print(test.info(),'\n')


#%% md
### 线下测试数据集分割

#%%
# 数据准备
user_action['index'] = user_action.userid+user_action.feedid+user_action.device
temp = user_action[action_features+['index']].drop_duplicates()

# 数据集划分
from sklearn.model_selection import train_test_split
train_index, valid_index = train_test_split(temp, test_size=0.2, random_state=42)

train_data = user_action[user_action['index'].isin(train_index['index'])]
valid_data = user_action[user_action['index'].isin(valid_index['index'])]
valid_data = valid_data[action_features].drop_duplicates()
