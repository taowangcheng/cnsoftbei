import pandas as pd
from sklearn import svm
import lightgbm as lgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    StackingClassifier, VotingClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

# 读取数据
data = pd.read_csv("data/train_mode.csv")
data = data.drop(['sample_id'], axis=1)
# 对数据进行 Min-Max 归一化
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[data.columns[:-1]])
data_scaled = pd.DataFrame(data_scaled, columns=data.columns[:-1])
data_scaled['label'] = data['label']

# 定义模型
models = [
    ("MultinomialNB", MultinomialNB()),
    ("GaussianNB", GaussianNB()),
    ("BernoulliNB", BernoulliNB()),
    ("ComplementNB", ComplementNB()),
    ("RandomForestClassifier", RandomForestClassifier(random_state=0, n_jobs=-1)),
    ("DecisionTreeClassifier", DecisionTreeClassifier()),
    ("svm.SVC", svm.SVC()),
    ("KNeighborsClassifier", KNeighborsClassifier(n_jobs=-1)),
    ("LogisticRegression", LogisticRegression(max_iter=5000, n_jobs=-1)),
    ("LinearDiscriminantAnalysis", LinearDiscriminantAnalysis()),

    ("BaggingClassifier(MultinomialNB(), n_jobs=-1)", BaggingClassifier(MultinomialNB(), n_jobs=-1)),
    ("BaggingClassifier(GaussianNB(), n_jobs=-1)", BaggingClassifier(GaussianNB(), n_jobs=-1)),
    ("BaggingClassifier(BernoulliNB(), n_jobs=-1)", BaggingClassifier(BernoulliNB(), n_jobs=-1)),
    ("BaggingClassifier(ComplementNB(), n_jobs=-1)", BaggingClassifier(ComplementNB(), n_jobs=-1)),
    ("BaggingClassifier(RandomForestClassifier(), n_jobs=-1)", BaggingClassifier(RandomForestClassifier(), n_jobs=-1)),
    ("BaggingClassifier(DecisionTreeClassifier(), n_jobs=-1)", BaggingClassifier(DecisionTreeClassifier(), n_jobs=-1)),
    ("BaggingClassifier(svm.SVC(), n_jobs=-1)", BaggingClassifier(svm.SVC(), n_jobs=-1)),
    ("BaggingClassifier(KNeighborsClassifier(), n_jobs=-1)", BaggingClassifier(KNeighborsClassifier(), n_jobs=-1)),
    ("BaggingClassifier(LogisticRegression(max_iter=5000, ), n_jobs=-1)", BaggingClassifier(LogisticRegression(max_iter=5000, ), n_jobs=-1)),
    ("BaggingClassifier(LinearDiscriminantAnalysis(), n_jobs=-1)", BaggingClassifier(LinearDiscriminantAnalysis(), n_jobs=-1)),
    ("GradientBoostingClassifier", GradientBoostingClassifier()),
    ("AdaBoostClassifier", AdaBoostClassifier()),
    ("StackingClassifier", StackingClassifier(estimators=[
        ('rf', RandomForestClassifier()), ('dt', DecisionTreeClassifier()), ('gdbt', GradientBoostingClassifier())], n_jobs=-1)),
    ("VotingClassifier", VotingClassifier(estimators=[
        ('rf', RandomForestClassifier()), ('dt', DecisionTreeClassifier()), ('gdbt', GradientBoostingClassifier())], n_jobs=-1)),
]

# 定义评估指标
scoring = ['precision_macro', 'recall_macro', 'f1_macro']

# 创建一个空的 DataFrame 对象，用于保存评估结果
results_df = pd.DataFrame(columns=['Model', 'Precision', 'Recall', 'F1 Score'])

# 进行十折交叉验证，并计算模型的精度、F1 分数和召回率
for model_name, model in models:
    print(f"Evaluating {model_name}...")
    if model_name == 'RandomForestmodel_name(random_state=0, n_jobs=-1)' or model_name == 'DecisionTreemodel_name()' or\
            model_name == 'Baggingmodel_name(RandomForestmodel_name(), n_jobs=-1)' or \
            model_name == 'Baggingmodel_name(DecisionTreemodel_name(), n_jobs=-1)' or \
            model_name == 'GradientBoostingmodel_name()' or model_name == 'AdaBoostmodel_name()':
        # 使用原始数据进行模型训练
        results = cross_validate(model, data.drop('label', axis=1), data['label'], cv=KFold(n_splits=10),
                                 scoring=scoring, n_jobs=-1, error_score=0, return_train_score=False, verbose=1)
    else:
        # 使用归一化后的数据进行模型训练
        results = cross_validate(model, data_scaled.drop('label', axis=1), data_scaled['label'], cv=KFold(n_splits=10),
                                 scoring=scoring, n_jobs=-1, error_score=0, return_train_score=False, verbose=1)
    precision = results['test_precision_macro'].mean()
    recall = results['test_recall_macro'].mean()
    f1 = results['test_f1_macro'].mean()

    # 将评估结果保存到 DataFrame 对象中
    tmp = {'Model': model_name, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
    tmp = pd.DataFrame(tmp, index=[0], columns=results_df.columns[:])
    results_df = pd.concat([results_df, tmp], ignore_index=False)

# 将评估结果写入到 CSV 文件中
results_df.to_csv("model_evaluation_results.csv", index=False)