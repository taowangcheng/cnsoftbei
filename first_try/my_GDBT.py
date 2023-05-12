import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

df = pd.read_csv('preprocess_train_multiclassification.csv')  # 每列是一个字典
df_0 = df.drop(['sample_id'], axis=1)  # 去除了sample_id列
df_1 = df.drop(['sample_id', 'label'], axis=1)  # 去除了sample_id、label两列

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imp = SimpleImputer(strategy='most_frequent')
X_train_imputed = imp.fit_transform(X_train)
X_test_imputed = imp.transform(X_test)  # 使用测试集中的众数来填充测试集中的空值

params = {
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7]
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

clf = GridSearchCV(GradientBoostingClassifier(random_state=42), params, cv=cv)
clf.fit(X_train_imputed, y_train)

score = clf.score(X_test_imputed, y_test)
print(f'Test accuracy: {score:.4f}')
y_pred = clf.predict(X_test_imputed)
f1 = f1_score(y_test, y_pred, average='macro')
print(f'F1 score: {f1:.4f}')
