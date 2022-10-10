import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

import data


def main():   
    X_unlabeled = data.get_unlabeled_data()
    X_train, y_train = data.get_training_data(outliers="nan")
    X_test = data.get_test_data()

    transformer = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(), data.categorical_features),
            ("numerical", StandardScaler(), data.numerical_features)
        ]
    )

    model = Pipeline([
        ("transform", data.append_unlabeled_input(transformer, X_unlabeled)),
        ("impute", KNNImputer()),
        ("pca", data.remove_unlabeled_output(PCA(), len(X_unlabeled))),
        ("svm", SVC(kernel='rbf'))
    ])

    param_grid = {
        "pca__n_components": range(8, 22),
        "svm__C": np.linspace(0.5, 3, 10),
    }

    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid,
        scoring="accuracy",
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10),
        n_jobs=-1,
        verbose=1
    )
    
    tuned_model = grid_search.fit(X_train, y_train)
    y_pred = tuned_model.predict(X_test)

    print("-" * 15, "Result", "-" * 15)
    print("Best parameters:", tuned_model.best_params_)
    print("Best score:", tuned_model.best_score_)
    print("Predictions:", "".join(y_pred.astype(str)))


if __name__ == "__main__":
    main()