"""
Conventional (non-spiking) decoders as baselines.

Implements:
1. Logistic Regression - Wang et al. (2025), multinomial with PCA
2. Linear SVM - Donati et al. (2026), linear kernel
3. MLP - Donati et al. (2026), single hidden layer

All decoders follow a common interface:
    .fit(X_train, y_train) -> self
    .predict(X_test) -> y_pred
    .score(X_test, y_test) -> accuracy
"""

import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class LogisticRegressionDecoder:
    """
    PCA + Multinomial Logistic Regression decoder (Wang et al.).

    Pipeline:
        Raw features -> StandardScaler -> PCA -> Logistic Regression

    Wang et al. used MATLAB's mnrfit (iteratively reweighted least squares).
    sklearn's LogisticRegression with lbfgs solver is equivalent.
    """

    def __init__(self, n_components=0.95, max_iter=1000, random_state=42):
        """
        Args:
            n_components: PCA variance ratio to retain (0.95 = 95%).
            max_iter: Max iterations for LR solver.
            random_state: For reproducibility.
        """
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("lr", SklearnLR(
                solver="lbfgs",
                max_iter=max_iter,
                random_state=random_state,
            )),
        ])

    def fit(self, X, y):
        """
        Fit the decoder.

        Args:
            X: (n_samples, n_features) feature matrix.
            y: (n_samples,) integer labels.
        """
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def score(self, X, y):
        return self.pipeline.score(X, y)

    def get_pca_info(self):
        """Return explained variance ratio after fitting."""
        pca = self.pipeline.named_steps["pca"]
        return {
            "n_components": pca.n_components_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "total_explained": np.sum(pca.explained_variance_ratio_),
        }


class SVMDecoder:
    """
    Linear SVM decoder (Donati et al. baseline).

    Donati et al. tested with various features (RMS, power, spike rate).
    RMS and power achieved best performance.
    """

    def __init__(self, kernel="linear", C=1.0, random_state=42):
        """
        Args:
            kernel: SVM kernel type. Donati: linear outperformed RBF.
            C: Regularization parameter.
            random_state: For reproducibility.
        """
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel=kernel,
                C=C,
                random_state=random_state,
            )),
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def score(self, X, y):
        return self.pipeline.score(X, y)


class MLPDecoder:
    """
    Multi-Layer Perceptron decoder (Donati et al. baseline).

    Donati et al.: 112 input -> 236 hidden (210% of input) -> N_class output.
    Uses ReLU activation and Adam optimizer.
    """

    def __init__(
        self,
        hidden_ratio=2.1,
        max_iter=500,
        lr=1e-3,
        random_state=42,
    ):
        """
        Args:
            hidden_ratio: Hidden layer size as ratio of input features.
                          Donati: 236/112 ≈ 2.1.
            max_iter: Training iterations.
            lr: Learning rate.
            random_state: For reproducibility.
        """
        self.hidden_ratio = hidden_ratio
        self.max_iter = max_iter
        self.lr = lr
        self.random_state = random_state
        self._model = None

    def fit(self, X, y):
        n_features = X.shape[1]
        hidden_size = int(n_features * self.hidden_ratio)

        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(hidden_size,),
                activation="relu",
                solver="adam",
                learning_rate_init=self.lr,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )),
        ])
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def score(self, X, y):
        return self._model.score(X, y)
