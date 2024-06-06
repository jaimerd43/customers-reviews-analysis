# %% [markdown]
# ### SUPPORT VECTOR MACHINE

# %% [markdown]
# **Features Selection**

# %%
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold


X_sub, _, y_sub, _ = train_test_split(X, y, test_size=0.8, stratify=y, random_state=42)


X_train_sub_scaled = scaler.fit_transform(X_sub)
X_sub_scaled = scaler.transform(X_sub)

clf = svm.SVC(kernel="linear")

rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(5), scoring='accuracy')

rfecv.fit(X_sub_scaled, y_sub)


print("Optimal number of features: %d" % rfecv.n_features_)


print('Selected features:', rfecv.support_)

# %%
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.fit_transform(X)

clf = svm.SVC()

clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)

print("Default hyperparameter values:")
print(clf.get_params())

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %% [markdown]
# **k-fold cross-validation**

# %%
scores1 = cross_val_score(clf, X_scaled, y, cv=10)

print("Cross-Validation Scores:", scores1)
print("Average Score:", scores1.mean())
print("Standard Deviation:", scores1.std())

# %% [markdown]
# **hyperparameter tunning**

# %%
from sklearn.model_selection import GridSearchCV

param_grid = {
    'kernel': [ 'linear', 'rbf', 'poly', 'sigmoid'],
    'C': [0.01,  1, 5, 10],
    'gamma': [0.1, 0.01,'auto', 'scale']
}

svc = SVC()

grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print("Mejores hiperparámetros encontrados:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

accuracy = best_model.score(X_test_scaled, y_test)
print("Precisión del mejor modelo en el conjunto de prueba:", accuracy)

# %%
best_kernel = 'linear'
best_C = 0.01


best_svc = SVC(kernel=best_kernel, C=best_C, random_state=404, probability=True)

best_svc.fit(X_train_scaled, y_train)

y_pred_4 = best_svc.predict(X_test_scaled)

print(classification_report(y_test, y_pred_4))

# %% [markdown]
# **k-fold cross-validation**

# %%
scores12 = cross_val_score(best_svc, X_scaled, y, cv=10)

print("Cross-Validation Scores:", scores12)
print("Average Score:", scores12.mean())
print("Standard Deviation:", scores12.std())

