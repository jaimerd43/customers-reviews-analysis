
# %% [markdown]
# ### RANDOM FOREST

# %%
randomforest = RandomForestClassifier(random_state = 12, oob_score=True)

y_pred_1 = randomforest.fit(X_train, y_train).predict(X_test)
report_1 = classification_report(y_test, y_pred_1)
print("Performance Metrics Before Hyperparameter Tuning:")
print(report_1)


# %% [markdown]
# **k-fold cross-validation**

# %%
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

scores = cross_val_score(randomforest, X, y, cv=10)

print("Cross-Validation Scores:", scores)
print("Average Score:", scores.mean())
print("Standard Deviation:", scores.std())


# %% [markdown]
# **Hyperparameter tunning**

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('randomforest', RandomForestClassifier())
])


param_grid = {
    'randomforest__n_estimators': [25, 50, 100],
    'randomforest__max_depth': [2, 5, 10],
    'randomforest__min_samples_split': [2, 4, 6],
    'randomforest__min_samples_leaf': [2, 4, 6]
}

grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)

print("Mejores hiperparámetros:", grid_search.best_params_)

best_model = grid_search.best_estimator_


# %%

random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=6,
    random_state=78
)

random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
report2 = classification_report(y_test, y_pred_rf)
print("\nPerformance Metrics After Hyperparameter Tuning:")
print(report2)

# %% [markdown]
# **k-fold Cross-validation after hyperparameter**

# %%
scores = cross_val_score(random_forest, X, y, cv=10)

print("Cross-Validation Scores:", scores)
print("Average Score:", scores.mean())
print("Standard Deviation:", scores.std())

