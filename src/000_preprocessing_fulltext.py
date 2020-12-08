import os
import re
import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd

from tqdm import tqdm
tqdm.pandas()

import xlrd


# 读取原始文件

train = pd.read_csv('raw_data/answer_train.csv')
test = pd.read_csv('raw_data/submit_example_test2.csv')  # 复赛测试集


# label 标签化

df_label = pd.DataFrame({'label': train.label.value_counts(normalize=True).index.tolist(),
                         'label_n': [i for i in range(train.label.nunique())]})
train = train.merge(df_label, on='label', how='left')
df_label.to_pickle('df_label.pickle')


# 使用 Linux find file 生成文件类型

os.system('find raw_data/train -name "*" -exec file -F "###" {} \; > ./train_filetype.txt')
os.system('find raw_data/test2 -name "*" -exec file -F "###" {} \; > ./test2_filetype.txt')

train_filetype = pd.read_csv('train_filetype.txt', sep='### ')
test_filetype = pd.read_csv('test2_filetype.txt', sep='### ')
train_filetype.columns = ['filename', 'filetype']
test_filetype.columns = ['filename', 'filetype']

train_filetype['filename'] = train_filetype['filename'].apply(lambda x: re.sub('raw_data/', '', x))
test_filetype['filename'] = test_filetype['filename'].apply(lambda x: re.sub('raw_data/', '', x))
train_filetype['filetype0'] = train_filetype['filetype'].apply(lambda x: x.split(',')[0])
test_filetype['filetype0'] = test_filetype['filetype'].apply(lambda x: x.split(',')[0])

train = pd.merge(train, train_filetype[['filename', 'filetype0']], on='filename', how='left')
test = pd.merge(test, test_filetype[['filename', 'filetype0']], on='filename', how='left')

# 除了 excel 文件类型外, 其余的直接使用 open().read() 读取

def plain_read(filename):
    try:
        text = open(f'raw_data/{filename}', 'rb').read().decode('utf-8')
    except:
        text = "READ_FAILED"

    return text


def xlrd_read(filename):
    text = ''
    try:
        xl_workbook = xlrd.open_workbook(f'raw_data/{filename}')
        sheet_names = xl_workbook.sheet_names()
        for idx in range(len(sheet_names)):
            xl_sheet = xl_workbook.sheet_by_index(idx)
            num_cols = xl_sheet.ncols
            for row_idx in range(0, xl_sheet.nrows):
                for col_idx in range(0, num_cols):
                    cell_obj = xl_sheet.cell(row_idx, col_idx)
                    cell_type_str = xlrd.sheet.ctype_text.get(cell_obj.ctype, 'unknown type')
                    if cell_type_str == 'text':
                        text += " " + cell_obj.value
    except:
        text = "READ_FAILED"

    return text


def read_content(row):
    try:
        if row['filetype0'] in ['CDFV2 Microsoft Excel', 'Composite Document File V2 Document']:
            text = xlrd_read(row['filename'])
        else:
            text = plain_read(row['filename'])
    except:
        text = 'READ_FAILED'

    return text


train['content'] = train.progress_apply(lambda row: read_content(row), axis=1)
test['content'] = test.progress_apply(lambda row: read_content(row), axis=1)


# 文本清理, 只留下中文字符

def clean_content(x):
    text = re.sub('[^\u4e00-\u9fa5]', ' ', x)
    text = re.sub(' +', ' ', text)

    return text


def clean_filename(x, is_test=False):
    if is_test:
        text = x.replace('test2/', '').replace('.xls', '').replace('.csv', '').replace('_', ' ')
    else:
        text = x.replace('train/', '').replace('.xls', '').replace('.csv', '').replace('_', ' ')
    return text


train['filename_cleaned'] = train['filename'].progress_apply(lambda x: clean_filename(x))
test['filename_cleaned'] = test['filename'].progress_apply(lambda x: clean_filename(x, is_test=True))

train['text'] = train['filename_cleaned'] + ' ' + train['content']
test['text'] = test['filename_cleaned'] + ' ' + test['content']

train['text'] = train['text'].progress_apply(lambda x: clean_content(x))
test['text'] = test['text'].progress_apply(lambda x: clean_content(x))


# 将测试集分割为两种类型, 有标题(13727) 和 无标题(11712)

def find_no_title(x):
    text = x.replace('test2/', '').replace('.xls', '').replace('.csv', '')
    if re.search(r'^[0-9a-z]+$', text):
        return 1
    else:
        return 0

test['no_title'] = test['filename'].apply(lambda x: find_no_title(x))


# 保存文件

test[test.no_title == 1].to_pickle('test_no_title.pickle')
test[test.no_title == 0].to_pickle('test_with_title.pickle')
train.to_pickle('train_fulltext_final.pickle')
test.to_pickle('test_fulltext_final.pickle')
