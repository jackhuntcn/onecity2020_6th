# 后处理

# 对 Content 进行清理后, 发现训练集和测试集有相当多的重复项
# 所以可以直接对测试集里与训练集中内容相同的部分直接打标签

train = pd.read_pickle('train_fulltext_final.pickle')
test = pd.read_pickle('test_fulltext_final.pickle')

train_content = train[['filename', 'content', 'label_n']]
test_content = test[['filename', 'content']]

# 对 Content 做文本清理, 只留下中文汉字

def clean_content(x):
    text = re.sub('[^\u4e00-\u9fa5]', ' ', x)
    text = re.sub(' +', ' ', text)
    text = text.strip()

    return text


train_content['cleaned_content'] = train_content['content'].progress_apply(lambda x: clean_content(x))
test_content['cleaned_content'] = test_content['content'].progress_apply(lambda x: clean_content(x))


# 去掉训练集中 "无权限访问" 的样本 (无意义的噪声)

train_tmp = train_content[train_content['cleaned_content'] != '无访问权限']
train_tmp = train_tmp[['cleaned_content', 'label_n']].drop_duplicates(subset=['cleaned_content'])

test_tmp = test_content.copy()


# 取出训练集和测试集中相同的文本

df = pd.merge(test_tmp[['filename', 'cleaned_content']],
              train_tmp,
              on='cleaned_content',
              how='left')

df_ = df[df.label_n.notna()]
df_['label_n'] = df_['label_n'].astype(int)
df_ = pd.merge(df_, df_label, on='label_n', how='left')
df_.columns = ['filename', 'cleaned_content', 'label_n', 'label_from_train']


# 读取 label

df_label = pd.read_pickle('df_label.pickle')


# 处理提交文件

sub = pd.read_csv('submission_all.csv')
new_sub = pd.merge(sub, 
                   df_[['filename', 'label_from_train']],
                   on='filename',
                   how='left')

to_check = new_sub[new_sub.label_from_train.notna()]
print(to_check.shape)

n = 0
for i, row in to_check.iterrows():
    if row['label'] != row['label_from_train']:
        # print(row['filename'])
        n += 1

print(n)


new_sub['label_from_train'] = new_sub['label_from_train'].fillna('')

pred = list()

for i, row in tqdm(new_sub.iterrows()):
    if row['label_from_train'] == '':
        pred.append(row['label'])
    else:
        pred.append(row['label_from_train'])

new_sub['new_label'] = pred
sub1 = new_sub[['filename', 'new_label']]
sub1.columns = ['filename', 'label']


# 生成最终提交文件

sub1.to_csv('submission_final.csv', index=False)
