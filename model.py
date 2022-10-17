import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score

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
        ("select", SelectKBest()),
        ("pca", data.remove_unlabeled_output(PCA(), len(X_unlabeled))),
        ("svm", SVC(kernel='rbf')),
        #("knn", KNeighborsClassifier())
        #("qda", QDA())
    ])

    param_grid = [{
        "select__k": range(16, 22),
        "pca__n_components": range(8, 15),
        #"svm__C": np.linspace(0.5, 2, 10),
        #"svm__gamma": np.linspace(0.01, 0.4, 10),
        #"knn__n_neighbors": range(5, 15)
        #"qda__reg_param": np.linspace(0, 0.3, 5)
    }]

    search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid,
        scoring="accuracy",
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0),
        n_jobs=-1,
        verbose=2
    )
    
    tuned_model = search.fit(X_train, y_train)
    y_pred = tuned_model.predict(X_test)

    print("-" * 15, "Result", "-" * 15)
    print("Best parameters:", tuned_model.best_params_)
    print("Best score:", tuned_model.best_score_)
    print("Predictions:", "".join(y_pred.astype(str)))

    scores = cross_val_score(
        search.best_estimator_, 
        X_train, 
        y_train, 
        scoring="accuracy",
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0),
        n_jobs=-1
    )

    plt.boxplot(scores)
    plt.show()



if __name__ == "__main__":
    main()

