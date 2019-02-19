import numpy as np

def mean_partial_dependence(model, X, y, col, values, method='predict'):
    """
    model - An sklearn-style estimator
    X - Dataframe
    y - Array-like
    col - Index in X of the target variable
    values - Values of col for which we want to compute the partial dependency
    """
    results = []
    for value in values:
        X_copy = X.copy()
        X_copy[col] = value
        f = getattr(model, method) 
        if method == 'predict_proba':
          y_mean = np.mean(f(X_copy), axis=0)
        else:
          y_mean = np.mean(f(X_copy))
        results.append(y_mean)
    return np.array(results)
