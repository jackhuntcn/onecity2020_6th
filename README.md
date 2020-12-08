# DC OneCity 2020 文本多分类 6th 代码分享

### 运行环境:

1. 系统: Ubuntu 16.04
2. 显卡: RTX 1080Ti
3. 主要软件环境: CUDA 10.1, Pytorch==1.7.0, transformers==3.5.1, simpletransformers==0.49.3

### 复赛应对思路:

1. 针对复赛中有接近一半的文件名使用随机的数字和英文字母替代的情况, 先将复赛测试集分为两个部分: 有标题和无标题
2. 有标题的部分使用 RoBERTa-wwm-ext seq_len 128 模型 (因为标题很准所以只跑其中的两折)
3. 无标题的部分训练集重新处理清洗到较小样本再使用 RoBERTa-wwm-ext seq_len 512 sliding_window 0.375 模型 (跑十折)
4. 最后拼接起来进行后处理 (比对训练集中存在的 Content 相同的部分) 得到最后的提交结果

### 代码结构说明

1. 新建一个 raw_data 目录, 将数据通过 unzip -O cp936 解压在此目录后, 执行 bash run.sh 即可
2. 在 RTX 1080Ti 上运行大概是 11 个小时, 如果是 V100 应该能在 8 小时内完成 (可以适当改大 batch_size)

```
├── run.sh
├── src
│   ├── 000_preprocessing_fulltext.py
│   ├── 001_train_folds_fulltext.py
│   ├── 002_preprocessing_content.py
│   ├── 003_train_folds_content.py
│   ├── 004_merge_folds.py
│   └── 005_postprocessing.py
└── README.md
```

### 榜上分数说明

1. 两模型单折不加后处理线上分数: 0.926
2. 后处理大概能加 0.005
3. 两模型多折加后处理最终得分: 0.939
