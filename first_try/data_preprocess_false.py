import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
import preprocessor as preprocessor
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier,\
    StackingClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, make_scorer, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('preprocess_train_multiclassification.csv')  # 每列是一个字典
df_0 = df.drop(['sample_id'], axis=1)  # 去除了sample_id列
df_1 = df.drop(['sample_id', 'label'], axis=1)  # 去除了sample_id、label两列

# 作图代码 ##############################################################################################################
# x = list(range(6295))
# for column in df_1:
#     plt.figure(figsize=(10, 8), dpi=200)
#     plt.scatter(x, df_1[column][1:], s=0.1)
#     plt.ylabel(column)
#     route = "./" + column + ".jpg"
#     plt.savefig(route)
#     plt.close()
#     # plt.show()

#   out是插补后的文件
out = 'NULL_processing.csv'

# 均值插补代码 ###########################################################################################################
# for column in df_1:
#     df_1[column] = df_1[column].replace(np.NaN, df_1[column].mean(skipna=True))
# df_1.to_csv(out)

# 中位数插补代码 ##########################################################################################################
# for column in df_1:
#     df_1[column] = df_1[column].replace(np.NaN, df_1[column].median(skipna=True))
# df_1.to_csv(out)

# 众数+均值插补代码 #######################################################################################################
# for column in df_1:
#     df_1[column] = df_1[column].replace(np.NaN, statistics.mode(df_1[column]))
#     df_1[column] = df_1[column].replace(np.NaN, df_1[column].mean(skipna=True))
# df_1.to_csv(out)

# 众数+中位数插补代码 #####################################################################################################
# for column in df_1:
#     df_1[column] = df_1[column].replace(np.NaN, statistics.mode(df_1[column]))
#     df_1[column] = df_1[column].replace(np.NaN, df_1[column].median(skipna=True))
# df_1.to_csv(out)

# 众数插补代码 ###########################################################################################################
df_1 = df_1.fillna(df.mode().iloc[0])
df_1.to_csv(out)

#   out_normal 是归一化后的文件
out_normal = 'normal.csv'
df_2 = pd.read_csv('NULL_processing.csv')  # 每列是一个字典
df_2 = df_2.drop(['Unnamed: 0'], axis=1)
# min_max 归一化代码 #####################################################################################################
scaler = MinMaxScaler(feature_range=(0, 1))
df_2 = pd.DataFrame(scaler.fit_transform(df_2), columns=df_2.columns)
df_2.to_csv(out_normal)

# # log 归一化代码 #########################################################################################################
# df_log_norm = df_2.apply(lambda x: np.log(x) - np.log(x.min() + 1))
# df_log_norm.to_csv(out_normal)


# df_3 = pd.read_csv('normal.csv')  # 每列是一个字典
# df_3 = df_3.drop(['Unnamed: 0'], axis=1)
# 卡方特征选择（要求必须是非负数）############################################################################################
# y = df_0['label']
# X = df_3.dropna(axis=1)
# X_new = SelectKBest(chi2, k=50).fit_transform(X, y)
# print(X_new.shape)


# 训练集划分 #############################################################################################################
df_all = pd.read_csv("normal.csv")  # 有空列
df_all = df_all.drop(['Unnamed: 0'], axis=1)
df_all = df_all.dropna(axis=1)
df_all['label'] = df_0['label']
y = df_all['label']
X = df_all.drop(['label'], axis=1).copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

df_tree = pd.read_csv('NULL_processing.csv')
df_tree = df_tree.drop(['Unnamed: 0'], axis=1)
df_tree = df_tree.dropna(axis=1)
df_tree['label'] = df_0['label']
y_tree = df_tree['label']
X_tree = df_tree.drop(['label'], axis=1).copy()

clf = [
    MultinomialNB(),
    GaussianNB(),
    BernoulliNB(),
    ComplementNB(),
    RandomForestClassifier(random_state=0, n_jobs=10),
    DecisionTreeClassifier(),
    svm.SVC(),
    KNeighborsClassifier(n_jobs=10),
    LogisticRegression(max_iter=5000, n_jobs=10),
    LinearDiscriminantAnalysis(),

    BaggingClassifier(MultinomialNB(), n_jobs=10),
    BaggingClassifier(GaussianNB(), n_jobs=10),
    BaggingClassifier(BernoulliNB(), n_jobs=10),
    BaggingClassifier(ComplementNB(), n_jobs=10),
    BaggingClassifier(RandomForestClassifier(), n_jobs=10),
    BaggingClassifier(DecisionTreeClassifier(), n_jobs=10),
    BaggingClassifier(svm.SVC(), n_jobs=10),
    BaggingClassifier(KNeighborsClassifier(), n_jobs=10),
    BaggingClassifier(LogisticRegression(max_iter=5000, ), n_jobs=10),
    BaggingClassifier(LinearDiscriminantAnalysis(), n_jobs=10),
    GradientBoostingClassifier(),
    AdaBoostClassifier(),
    StackingClassifier(estimators=[
        ('mnb', MultinomialNB()), ('gnb', GaussianNB()), ('bnb', BernoulliNB()), ('cnb', ComplementNB()),
        ('rf', RandomForestClassifier()), ('dt', DecisionTreeClassifier()), ('svc', svm.SVC()),
        ('lr', LogisticRegression(max_iter=100000000)), ('knn', KNeighborsClassifier()),
        ('lda', LinearDiscriminantAnalysis())], n_jobs=10),
    VotingClassifier(estimators=[
        ('mnb', MultinomialNB()), ('gnb', GaussianNB()), ('bnb', BernoulliNB()), ('cnb', ComplementNB()),
        ('rf', RandomForestClassifier()), ('dt', DecisionTreeClassifier()), ('svc', svm.SVC(probability=True)),
        ('lr', LogisticRegression(max_iter=10000)), ('knn', KNeighborsClassifier()),
        ('lda', LinearDiscriminantAnalysis())], n_jobs=10),
]

clf_name = [
    'MultinomialNB Naive Bayes', 'GaussianNB', 'BernoulliNB', 'ComplementNB', 'Random forest', 'Decision Tree',
    'SVC', 'KNN', 'Logistic Regression', 'Linear Discriminant Analysis',
    'Bagging Classifier mnb', 'Bagging Classifier gnb', 'Bagging Classifier bnb', 'Bagging Classifier cnb',
    'Bagging Classifier rf', 'Bagging Classifier dt', 'Bagging Classifier svm', 'Bagging Classifier knn',
    'Bagging Classifier lr', 'Bagging Classifier lda', 'GradientBoostingClassifier', 'AdaBoostClassifier',
    'StackingClassifier', 'Vote'
]

f = open(file='5fold_result_micro.txt', mode="w", encoding="utf-8")
_scoring = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro']  # 设置评分项
for classifier in clf:
    if classifier == 'RandomForestClassifier(random_state=0, n_jobs=10)' or classifier == 'DecisionTreeClassifier()' or\
        classifier == 'BaggingClassifier(RandomForestClassifier(), n_jobs=10)' or \
        classifier == 'BaggingClassifier(DecisionTreeClassifier(), n_jobs=10)' or \
        classifier == 'GradientBoostingClassifier()' or classifier == 'AdaBoostClassifier()':
            results = cross_validate(estimator=classifier,
                                     X=X_tree,
                                     y=y_tree,
                                     cv=5,
                                     scoring=_scoring,
                                     return_train_score=True)
    else:
        results = cross_validate(estimator=classifier,
                                 X=X,
                                 y=y,
                                 cv=5,
                                 scoring=_scoring,
                                 return_train_score=True)
    print("########################################################################################")
    print(classifier)
    print("")
    print("Training Accuracy scores")
    print(results['train_accuracy'])
    print("Mean Training Accuracy")
    print(results['train_accuracy'].mean() * 100)
    print('')
    print("Training Precision scores")
    print(results['train_precision_micro'])
    print("Mean Training Precision")
    print(results['train_precision_micro'].mean())
    print('')
    print("Training Recall scores")
    print(results['train_recall_micro'])
    print("Mean Training Recall")
    print(results['train_recall_micro'].mean())
    print('')
    print("Training F1 scores")
    print(results['train_f1_micro'])
    print("Mean Training F1 Score")
    print(results['train_f1_micro'].mean())
    print('')
    print("Validation Accuracy scores")
    print(results['test_accuracy'])
    print("Mean Validation Accuracy")
    print(results['test_accuracy'].mean() * 100)
    print('')
    print("Validation Precision scores")
    print(results['test_precision_micro'])
    print("Mean Validation Precision")
    print(results['test_precision_micro'].mean())
    print('')
    print("Validation Recall scores")
    print(results['test_recall_micro'])
    print("Mean Validation Recall")
    print(results['test_recall_micro'].mean())
    print('')
    print("Validation F1 scores")
    print(results['test_f1_micro'])
    print("Mean Validation F1 Score")
    print(results['test_f1_micro'].mean())
    print('')
    print("########################################################################################")

    f.write("#################################################"+"\n")
    content = str(classifier) + '\n'
    f.write(content)
    f.write("\n")
    f.write("Training Accuracy scores"+"\n")
    content = str(results['train_accuracy']) + '\n'
    f.write(content)
    f.write("Mean Training Accuracy"+"\n")
    content = str(results['train_accuracy'].mean()*100) + '\n'
    f.write(content)
    f.write("\n")
    f.write("Training Precision scores"+"\n")
    content = str(results['train_precision_micro']) + '\n'
    f.write(content)
    f.write("Mean Training Precision"+"\n")
    content = str(results['train_precision_micro'].mean()) + '\n'
    f.write(content)
    f.write("\n")
    f.write("Training Recall scores")
    content = str(results['train_recall_micro']) + '\n'
    f.write(content)
    f.write("Mean Training Recall")
    content = str(results['train_recall_micro'].mean()) + '\n'
    f.write(content)
    f.write("\n")
    f.write("Training F1 scores")
    content = str(results['train_f1_micro']) + '\n'
    f.write(content)
    f.write("Mean Training F1 Score")
    content = str(results['train_f1_micro'].mean()) + '\n'
    f.write(content)
    f.write("\n")
    f.write("Validation Accuracy scores")
    content = str(results['test_accuracy']) + '\n'
    f.write(content)
    f.write("Mean Validation Accuracy")
    content = str(results['test_accuracy'].mean() * 100) + '\n'
    f.write(content)
    f.write("\n")
    f.write("Validation Precision scores")
    content = str(results['test_precision_micro']) + '\n'
    f.write(content)
    f.write("Mean Validation Precision")
    content = str(results['test_precision_micro'].mean()) + '\n'
    f.write(content)
    f.write("\n")
    f.write("Validation Recall scores")
    content = str(results['test_recall_micro']) + '\n'
    f.write(content)
    f.write("Mean Validation Recall")
    content = str(results['test_recall_micro'].mean()) + '\n'
    f.write(content)
    f.write("\n")
    f.write("Validation F1 scores")
    content = str(results['test_f1_micro']) + '\n'
    f.write(content)
    f.write("Mean Validation F1 Score")
    content = str(results['test_f1_micro'].mean()) + '\n'
    f.write(content)
    f.write("\n")
    f.write('\n')
f.close()
