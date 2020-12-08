import sys

import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

from simpletransformers.classification import ClassificationModel, ClassificationArgs

fold = int(sys.argv[1])


# 读取之前预处理完的文件

train = pd.read_pickle('train_fulltext_final.pickle')
test = pd.read_pickle('test_no_title.pickle')

df_label = pd.read_pickle('df_label.pickle')


# 清理文本, 只留下中文汉字

def clean_content(x):
    text = re.sub('[^\u4e00-\u9fa5]', ' ', x)
    text = re.sub(' +', ' ', text)
    text = text.strip()

    return text


train['cleaned_content'] = train['cleaned_content'].progress_apply(lambda x: clean_content(x))
test['cleaned_content'] = test['cleaned_content'].progress_apply(lambda x: clean_content(x))


# 生成训练样本和测试样本

train_df = train[['cleaned_content', 'label_n']]
train_df.columns = ['text', 'label']

test_df = test[['filename', 'cleaned_content']]
test_df.columns = ['filename', 'text']


# 去除重复样本和 Label 不一致的样本

train_df = train_df.drop_duplicates(subset=['text', 'label'])

df_tmp = train_df.copy()
df_tmp['count'] = df_tmp.groupby(['text'])['text'].transform('count')
train_df = df_tmp[df_tmp['count'] == 1]


# 截断

train_df['text'] = train_df['text'].progress_apply(lambda x: x[:512])
test_df['text'] = test_df['text'].progress_apply(lambda x: x[:512])


# 保存文件

train_df.to_pickle('train_content_final.pickle')
test_df.to_pickle('test_content_final.pickle')
