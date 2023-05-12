import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

# 读取数据
data = pd.read_csv("preprocess_train_multiclassification.csv")

# 定义模型
models = [
    ("Decision Tree", DecisionTreeClassifier()),
    ("Random Forest", RandomForestClassifier())
]

# 定义十折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 定义结果列表
results = []

# 循环每一折交叉验证
for train_index, test_index in kf.split(data):
    # 将数据集分为训练集和测试集
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]

    # 计算各列的中位数
    median_values = train_data.median()

    # 对训练集和测试集中的空值进行插补
    train_data = train_data.fillna(median_values)
    test_data = test_data.fillna(median_values)

    # 循环每种模型
    for model_name, model in models:
        # 训练模型
        model.fit(train_data.drop("label", axis=1), train_data["label"])

        # 预测测试集
        y_pred = model.predict(test_data.drop("label", axis=1))

        # 计算精度、召回率和f1分数
        accuracy = accuracy_score(test_data["label"], y_pred)
        recall = recall_score(test_data["label"], y_pred, average="macro")
        f1 = f1_score(test_data["label"], y_pred, average="macro")

        # 将结果添加到结果列表中
        results.append((model_name, accuracy, recall, f1))

# 打印结果
for model_name, accuracy, recall, f1 in results:
    print(f"{model_name}: Accuracy={accuracy:.4f}, Recall={recall:.4f}, F1={f1:.4f}")