import numpy as np
import pandas as pd
from sklearn import svm
import lightgbm as lgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    StackingClassifier, VotingClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

# 读取数据
data = pd.read_csv("data/train_mode.csv")
data = data.drop(['sample_id', 'feature57', 'feature77', 'feature100'], axis=1)

# 对数据进行 Min-Max 归一化
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[data.columns[:-1]])
data_scaled = pd.DataFrame(data_scaled, columns=data.columns[:-1])
data_scaled['label'] = data['label']

# 定义模型
models = [
    # ("MultinomialNB", MultinomialNB()),
    # ("GaussianNB", GaussianNB()),
    # ("BernoulliNB", BernoulliNB()),
    # ("ComplementNB", ComplementNB()),
    # ("RandomForestClassifier", RandomForestClassifier(random_state=0, n_jobs=-1)),
    # ("DecisionTreeClassifier", DecisionTreeClassifier()),
    # ("svm.SVC", svm.SVC()),
    # ("KNeighborsClassifier", KNeighborsClassifier(n_jobs=-1)),
    # ("LogisticRegression", LogisticRegression(max_iter=5000, n_jobs=-1)),
    # ("LinearDiscriminantAnalysis", LinearDiscriminantAnalysis()),
    #
    # ("BaggingClassifier(MultinomialNB(), n_jobs=-1)", BaggingClassifier(MultinomialNB(), n_jobs=-1)),
    # ("BaggingClassifier(GaussianNB(), n_jobs=-1)", BaggingClassifier(GaussianNB(), n_jobs=-1)),
    # ("BaggingClassifier(BernoulliNB(), n_jobs=-1)", BaggingClassifier(BernoulliNB(), n_jobs=-1)),
    # ("BaggingClassifier(ComplementNB(), n_jobs=-1)", BaggingClassifier(ComplementNB(), n_jobs=-1)),
    # ("BaggingClassifier(RandomForestClassifier(), n_jobs=-1)", BaggingClassifier(RandomForestClassifier(), n_jobs=-1)),
    # ("BaggingClassifier(DecisionTreeClassifier(), n_jobs=-1)", BaggingClassifier(DecisionTreeClassifier(), n_jobs=-1)),
    # ("BaggingClassifier(svm.SVC(), n_jobs=-1)", BaggingClassifier(svm.SVC(), n_jobs=-1)),
    # ("BaggingClassifier(KNeighborsClassifier(), n_jobs=-1)", BaggingClassifier(KNeighborsClassifier(), n_jobs=-1)),
    # ("BaggingClassifier(LogisticRegression(max_iter=5000, n_jobs=-1), n_jobs=-1)",
    #  BaggingClassifier(LogisticRegression(max_iter=5000, n_jobs=-1), n_jobs=-1)),
    # ("BaggingClassifier(LinearDiscriminantAnalysis(), n_jobs=-1)",
    #  BaggingClassifier(LinearDiscriminantAnalysis(), n_jobs=-1)),
    #
    # ("GradientBoostingClassifier", GradientBoostingClassifier(random_state=0)),
    # ("AdaBoostClassifier", AdaBoostClassifier(random_state=0)),
    # ("lgb.LGBMClassifier", lgb.LGBMClassifier(random_state=0, n_jobs=-1)),
    ("StackingClassifier", StackingClassifier(estimators=[
        ('rf', RandomForestClassifier()),
        ('dt', DecisionTreeClassifier()),
        ('lgbm', lgb.LGBMClassifier(random_state=0, n_jobs=-1)),
        ('gdbt', GradientBoostingClassifier())], n_jobs=-1)),
    # ("VotingClassifier", VotingClassifier(estimators=[
    #     ('rf', RandomForestClassifier()), ('dt', DecisionTreeClassifier()), ('gdbt', GradientBoostingClassifier())], n_jobs=-1)),
]

# 定义 KFold 交叉验证
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

# 定义特征选择器
selector = SelectKBest(f_classif, k=104)

# 对每个模型进行交叉验证
for model_name, model in models:
    print(f"Training model: {model_name}")
    results = {}
    y_pred = None

    # 交叉验证
    for i, (train_index, test_index) in enumerate(kf.split(data_scaled)):
        print(f"Fold {i + 1}")
        X_train, X_test = data_scaled.iloc[train_index][data_scaled.columns[:-1]], data_scaled.iloc[test_index][
            data_scaled.columns[:-1]]
        y_train, y_test = data_scaled.iloc[train_index]['label'], data_scaled.iloc[test_index]['label']

        X_train_tree, X_test_tree = data.iloc[train_index][data.columns[:-1]], data.iloc[test_index][
            data.columns[:-1]]
        y_train_tree, y_test_tree = data.iloc[train_index]['label'], data.iloc[test_index]['label']

        # 特征选择
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

        X_train_tree = selector.fit_transform(X_train_tree, y_train_tree)
        X_test_tree = selector.transform(X_test_tree)

        # 训练模型
        if model_name == 'RandomForestmodel_name(random_state=0, n_jobs=-1)' or \
                model_name == 'DecisionTreemodel_name()' or \
                model_name == 'Baggingmodel_name(RandomForestmodel_name(), n_jobs=-1)' or \
                model_name == 'Baggingmodel_name(DecisionTreemodel_name(), n_jobs=-1)' or \
                model_name == 'GradientBoostingmodel_name()' or model_name == 'AdaBoostmodel_name()' or \
                model_name == 'StackingClassifier':
            model.fit(X_train_tree, y_train_tree)

            # 预测测试集
            y_pred_fold = model.predict(X_test_tree)

            # 计算测试集上的精度和 F1 分数
            accuracy_fold = accuracy_score(y_test_tree, y_pred_fold)
            f1_score_fold = f1_score(y_test_tree, y_pred_fold, average='macro')

            # 计算每个类别的分类精度
            accuracy_by_class_fold = {}
            classes = sorted(list(set(y_test_tree)))
            for cls in classes:
                mask = (y_test_tree == cls)
                accuracy = accuracy_score(y_test_tree[mask], y_pred_fold[mask])
                accuracy_by_class_fold[f"class_{cls}"] = accuracy

            # 记录测试结果
            results[f"Fold {i + 1} accuracy"] = accuracy_fold
            results[f"Fold {i + 1} f1-score"] = f1_score_fold
            results.update(accuracy_by_class_fold)

            # 合并所有折叠的预测结果
            if y_pred is None:
                y_pred = y_pred_fold
            else:
                y_pred = np.concatenate([y_pred, y_pred_fold], axis=0)
        else:
            model.fit(X_train, y_train)

            # 预测测试集
            y_pred_fold = model.predict(X_test)

            # 计算测试集上的精度和 F1 分数
            accuracy_fold = accuracy_score(y_test, y_pred_fold)
            f1_score_fold = f1_score(y_test, y_pred_fold, average='macro')

            # 计算每个类别的分类精度
            accuracy_by_class_fold = {}
            classes = sorted(list(set(y_test)))
            for cls in classes:
                mask = (y_test == cls)
                accuracy = accuracy_score(y_test[mask], y_pred_fold[mask])
                accuracy_by_class_fold[f"class_{cls}"] = accuracy

            # 记录测试结果
            results[f"Fold {i + 1} accuracy"] = accuracy_fold
            results[f"Fold {i + 1} f1-score"] = f1_score_fold
            results.update(accuracy_by_class_fold)

            # 合并所有折叠的预测结果
            if y_pred is None:
                y_pred = y_pred_fold
            else:
                y_pred = np.concatenate([y_pred, y_pred_fold], axis=0)

    # 计算平均精度和 F1 分数
    accuracy_mean = np.mean([results[f"Fold {i + 1} accuracy"] for i in range(n_splits)])
    f1_score_mean = np.mean([results[f"Fold {i + 1} f1-score"] for i in range(n_splits)])

    # 计算每个类别的平均分类精度
    accuracy_by_class_mean = {}
    for cls in classes:
        accuracy_by_class_mean[f"class_{cls}"] = np.mean([results[f"class_{cls}"] for i in range(n_splits)])

    # 输出结果
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy_mean:.4f}")
    print(f"F1-score: {f1_score_mean:.4f}")
    print("Accuracy by class:")
    for cls, accuracy in accuracy_by_class_mean.items():
        print(f"{cls}: {accuracy:.4f}")

    # 保存结果到文件
    with open(f"./precision_per_label/{model_name}(rf dt ldbm gbdt).txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {accuracy_mean:.4f}\n")
        f.write(f"F1-score: {f1_score_mean:.4f}\n")
        f.write("Accuracy by class:\n")
        for cls, accuracy in accuracy_by_class_mean.items():
            f.write(f"{cls}: {accuracy:.4f}\n")