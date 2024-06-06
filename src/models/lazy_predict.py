# %%
import lazypredict

from lazypredict.Supervised import LazyClassifier

from sklearn.model_selection import train_test_split

# %%
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.80,test_size=0.20,random_state=404)

# %%
clflazy = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clflazy.fit(X_train, X_test, y_train, y_test)

print(models)


