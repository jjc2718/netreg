"""
Quick script to benchmark scikit-learn logistic regression pipeline

"""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_predict
)

def train_sklearn_model(X_train, X_test, y_train, alphas, l1_ratios,
                        seed=0, n_folds=5, max_iter=1000):

    # Setup the classifier parameters
    clf_parameters = {
        "classify__loss": ["log"],
        "classify__penalty": ["elasticnet"],
        "classify__alpha": alphas,
        "classify__l1_ratio": l1_ratios,
    }

    estimator = Pipeline(
        steps=[
            (
                "classify",
                SGDClassifier(
                    random_state=seed,
                    class_weight="balanced",
                    loss="log",
                    max_iter=max_iter,
                    tol=1e-3,
                ),
            )
        ]
    )

    cv_pipeline = GridSearchCV(
        estimator=estimator,
        param_grid=clf_parameters,
        n_jobs=-1,
        cv=n_folds,
        scoring="roc_auc",
        iid=True,
        return_train_score=True,
    )

    # Fit the model
    cv_pipeline.fit(X=X_train, y=y_train)

    # Obtain cross validation results
    y_cv = cross_val_predict(
        cv_pipeline.best_estimator_,
        X=X_train,
        y=y_train,
        cv=n_folds,
        method="decision_function",
    )

    # Get all performance results
    y_pred_train = cv_pipeline.decision_function(X_train)
    y_pred_test = cv_pipeline.decision_function(X_test)

    y_pred_bn_train = cv_pipeline.predict(X_train)
    y_pred_bn_test = cv_pipeline.predict(X_test)

    return y_pred_train, y_pred_test, y_pred_bn_train, y_pred_bn_test
