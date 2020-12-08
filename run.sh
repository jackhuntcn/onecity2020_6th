#!/bin/bash

# 预处理, 读取文档内容
python3 ./src/000_preprocessing_fulltext.py

# FullText 模型, 只跑其中两折
for fold in 0 9; do python3 ./src/001_train_folds_fulltext.py ${fold}; done

# 生成 Content 训练集和测试集
python3 ./src/002_preprocessing_content.py

# Content 模型, 跑十折
for fold in `seq 0 9`; do python3 ./src/003_train_folds_content.py ${fold}; done

# 合并两个模型各折的结果
python3 ./src/004_merge_folds.py

# 后处理并生成最终的提交文件
python3 ./src/005_postprocessing.py
