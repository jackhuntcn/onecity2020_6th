import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd

df_label = pd.read_pickle('df_label.pickle')

# 十折有标题模型, 时间关系只取其中的两折

p0 = np.load('prob_fulltext_0.npy')
p9 = np.load('prob_fulltext_9.npy')

p = (p0 + p9) / 2
predictions = p.argmax(axis=1)

test1 = pd.read_pickle('test_with_title.pickle')
sub1 = test1[['filename']]
sub1['label_n'] = predictions
sub1 = pd.merge(sub1, df_label, on='label_n', how='left')
sub1.drop(['label_n'], axis=1, inplace=True)


# 十折无标题模型

p0 = np.load('prob_content_0.npy')
p1 = np.load('prob_content_1.npy')
p2 = np.load('prob_content_2.npy')
p3 = np.load('prob_content_3.npy')
p4 = np.load('prob_content_4.npy')
p5 = np.load('prob_content_5.npy')
p6 = np.load('prob_content_6.npy')
p7 = np.load('prob_content_7.npy')
p8 = np.load('prob_content_8.npy')
p9 = np.load('prob_content_9.npy')

p = (p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) / 10
predictions = p.argmax(axis=1)

test2 = pd.read_pickle('test_no_title.pickle')
sub2 = test2[['filename']]
sub2['label_n'] = predictions
sub2 = pd.merge(sub2, df_label, on='label_n', how='left')
sub2.drop(['label_n'], axis=1, inplace=True)


# 合并

sub_all = pd.concat([sub1, sub2])
sub = pd.read_csv('raw_data/submit_example_test2.csv')[['filename']]
sub = pd.merge(sub, sub_all, on='filename', how='left')

print(sub[sub.label.isna()].shape)


# 生成提交文件

sub.to_csv('submission_all.csv', index=False)
