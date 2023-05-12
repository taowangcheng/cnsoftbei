import pandas as pd
from sklearn.impute import KNNImputer

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# 使用均值填充缺失值
train_mean = train.fillna(train.mean())
test_mean = test.fillna(test.mean())

train_mean.to_csv('data/train_mean.csv', index=False)
test_mean.to_csv('data/test_mean.csv', index=False)

# 使用中位数填充缺失值
train_median = train.fillna(train.median())
test_median = test.fillna(test.median())

train_median.to_csv('data/train_median.csv', index=False)
test_median.to_csv('data/test_median.csv', index=False)

# 使用众数填充缺失值
train_mode = train.fillna(train.mode().iloc[0])
test_mode = test.fillna(test.mode().iloc[0])

train_mode.to_csv('data/train_mode.csv', index=False)
test_mode.to_csv('data/test_mode.csv', index=False)

# 使用线性插值填充缺失值
train_interpolation = train.interpolate()
test_interpolation = test.interpolate()

train_interpolation.to_csv('data/train_interpolation.csv', index=False)
test_interpolation.to_csv('data/test_interpolation.csv', index=False)

# 使用KNN填充缺失值
imputer = KNNImputer(n_neighbors=7)
train_knn = pd.DataFrame(imputer.fit_transform(train), columns=train.columns)
test_knn = pd.DataFrame(imputer.fit_transform(test), columns=test.columns)

train_knn.to_csv('data/train_knn.csv', index=False)
test_knn.to_csv('data/test_knn.csv', index=False)
