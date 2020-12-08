import sys

import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

from simpletransformers.classification import ClassificationModel, ClassificationArgs

fold = int(sys.argv[1])


# 读取预处理完的文件

train = pd.read_pickle('train_content_final.pickle')
test = pd.read_pickle('test_content_final.pickle')

df_label = pd.read_pickle('df_label.pickle')

train_df = train.copy()


# 获取当前的 fold 的数据

df = train_df.sample(frac=1., random_state=1024)

train_df = df[df.index % 10 != fold]
eval_df = df[df.index % 10 == fold]

print(train_df.shape, eval_df.shape)


# 设置参数, 加载预训练模型

model_args = ClassificationArgs()

model_args.output_dir = f"RoBERTa_content_512seq_fold_{fold}"
model_args.max_seq_length = 512
model_args.sliding_window = True
model_args.stride = 0.375
model_args.train_batch_size = 8       # V100 可以改成 16, 应该可以在半小时内完成
model_args.num_train_epochs = 3
model_args.fp16 = False
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
model_args.save_model_every_epoch = False
model_args.overwrite_output_dir = True

model = ClassificationModel("bert",
                            "hfl/chinese-roberta-wwm-ext",
                            num_labels=len(df_label),
                            args=model_args)


# Fine Tune

model.train_model(train_df, eval_df=eval_df)

# local CV
# result, _, _ = model.eval_model(eval_df, acc=accuracy_score)
# print(result['acc'])

# test 推断

data = []
for i, row in test.iterrows():
    data.append(row['text'])

predictions, raw_outputs = model.predict(data)

# 保存概率

np.save(f'prob_content_{fold}', raw_outputs)
