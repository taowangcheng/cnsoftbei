import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

# 加载数据集
data = pd.read_csv('NULL_processing.csv')
data_all = pd.read_csv('preprocess_train_multiclassification.csv')
X = data.drop(['Unnamed: 0'], axis=1)
y = data_all['label']

# 删除常数特征
constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(X)
X = X.loc[:, constant_filter.get_support()]
f1_max = 0

# 特征选择
k = 104
selector = SelectKBest(score_func=f_classif, k=k)
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()
# print('Selected features:', selected_features)

# 设置参数
params = {
    'max_depth': 15,
    'task': 'train',

    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_leaves': 50,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'num_class': 6,
    'force_col_wise': True
}

# 十折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracy_scores = []

for train_idx, test_idx in kf.split(X_new):
    X_train, y_train = X_new[train_idx], y.iloc[train_idx]
    X_test, y_test = X_new[test_idx], y.iloc[test_idx]

    # 构建LightGBM数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)

    # 训练模型
    num_round = 100
    model = lgb.train(params, train_data, num_round)

    # 预测测试集
    y_pred = model.predict(X_test)
    y_pred = [list(value).index(max(value)) for value in y_pred]

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# 计算F1分数
f1 = f1_score(y_test, y_pred, average='macro')

# # 输出准确率的平均值
print("k:", k)
print('Average accuracy:', sum(accuracy_scores) / len(accuracy_scores))
print("F1 score:", f1)
print("")
