import pandas as pd

# 读取数据
data = pd.read_csv("data/preprocess_train_multiclassification.csv")
# print(data)

# 使用 sample() 函数打乱 DataFrame 行顺序
data_shuffled = data.sample(frac=1)
# print(data_shuffled)

# 按照6：2：2划分训练集、验证集、测试集
n_samples = len(data_shuffled)
train_idx = int(0.8 * n_samples)

train_set = data_shuffled[:train_idx]
test_set = data_shuffled[train_idx:]

# 打印数据集大小
print(f'Training set size: {len(train_set)}')
print(f'Test set size: {len(test_set)}')

train_set.to_csv('data/train.csv', index=False)
test_set.to_csv('data/test.csv', index=False)
