import mlx.core as mx
import numpy as np
from typing import Optional, Union
from ..base.base_estimator import BaseEstimator


class GaussianNB(BaseEstimator):
    """
    Gaussian Naive Bayes (GaussianNB).

    Parameters:
        var_smoothing: Portion of the largest variance of all features that is added to
            variances for calculation stability.
        priors: Prior probabilities of the classes. If specified, the priors are not
            adjusted according to the data.
    """
    def __init__(
        self,
        var_smoothing: float = 1e-9,
        priors: Optional[Union[np.array, mx.array]] = None
    ):
        super().__init__()
        self.var_smoothing = var_smoothing
        self.priors = priors

        self.classes_: Optional[np.array] = None
        self.n_classes_: Optional[int] = None
        self.class_prior_: Optional[np.array] = None
        self.theta_: Optional[np.array] = None  # mean of each feature per class
        self.var_: Optional[np.array] = None   # variance of each feature per class
        self.epsilon_: Optional[float] = None

    def fit(self, X: Union[mx.array, np.array], y: Union[mx.array, np.array]) -> "GaussianNB":
        """
        Fit Gaussian Naive Bayes according to X, y.

        Parameters:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            Fitted estimator
        """
        X, y = self._validate_data(X, y)
        X_np = np.array(X)
        y_np = np.array(y)

        self.classes_ = np.unique(y_np)
        self.n_classes_ = len(self.classes_)
        n_features = X_np.shape[1]

        # Initialize parameters
        self.theta_ = np.zeros((self.n_classes_, n_features), dtype=np.float64)
        self.var_ = np.zeros((self.n_classes_, n_features), dtype=np.float64)
        self.class_prior_ = np.zeros(self.n_classes_, dtype=np.float64)

        # Compute epsilon for variance smoothing
        self.epsilon_ = self.var_smoothing * np.var(X_np, axis=0).max()

        for i, c in enumerate(self.classes_):
            X_c = X_np[y_np == c]
            self.theta_[i] = np.mean(X_c, axis=0)
            self.var_[i] = np.var(X_c, axis=0) + self.epsilon_

            if self.priors is None:
                self.class_prior_[i] = len(X_c) / len(X_np)
            else:
                self.class_prior_[i] = np.array(self.priors)[i]

        return self

    def _joint_log_likelihood(self, X: np.array) -> np.array:
        """Compute the unnormalized posterior log probability of X."""
        joint_log_likelihood = []
        for i in range(self.n_classes_):
            prior = np.log(self.class_prior_[i])
            n_ij = -0.5 * np.sum(np.log(2. * np.pi * self.var_[i]))
            n_ij -= 0.5 * np.sum(((X - self.theta_[i]) ** 2) / self.var_[i], axis=1)
            joint_log_likelihood.append(prior + n_ij)

        return np.array(joint_log_likelihood).T

    def predict(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Perform classification on an array of test vectors X.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Predicted classes of shape (n_samples,)
        """
        if self.theta_ is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        X_np = np.array(X)

        jll = self._joint_log_likelihood(X_np)
        predictions = self.classes_[np.argmax(jll, axis=1)]

        return mx.array(predictions.astype(np.int32))

    def predict_proba(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Return probability estimates for the test vectors X.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Probability of the sample for each class in the model, of shape (n_samples, n_classes)
        """
        if self.theta_ is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        X_np = np.array(X)

        jll = self._joint_log_likelihood(X_np)
        # Normalize log probabilities
        log_prob_x = np.logaddexp.reduce(jll, axis=1, keepdims=True)
        probas = np.exp(jll - log_prob_x)

        return mx.array(probas.astype(np.float32))


class MultinomialNB(BaseEstimator):
    """
    Naive Bayes classifier for multinomial models.

    Parameters:
        alpha: Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
        fit_prior: Whether to learn class prior probabilities or not. If False, a uniform prior will be used.
        class_prior: Prior probabilities of the classes. If specified, the priors are not adjusted according to the data.
    """
    def __init__(
        self,
        alpha: float = 1.0,
        fit_prior: bool = True,
        class_prior: Optional[Union[np.array, mx.array]] = None
    ):
        super().__init__()
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

        self.classes_: Optional[np.array] = None
        self.n_classes_: Optional[int] = None
        self.feature_count_: Optional[np.array] = None
        self.class_count_: Optional[np.array] = None
        self.feature_log_prob_: Optional[np.array] = None
        self.class_log_prior_: Optional[np.array] = None

    def fit(self, X: Union[mx.array, np.array], y: Union[mx.array, np.array]) -> "MultinomialNB":
        """
        Fit Multinomial Naive Bayes according to X, y.

        Parameters:
            X: Training data of shape (n_samples, n_features), expected to be non-negative counts
            y: Target values of shape (n_samples,)

        Returns:
            Fitted estimator
        """
        X, y = self._validate_data(X, y)
        X_np = np.array(X)
        y_np = np.array(y)

        # Ensure non-negative values
        if np.any(X_np < 0):
            raise ValueError("Input X must be non-negative for MultinomialNB")

        self.classes_ = np.unique(y_np)
        self.n_classes_ = len(self.classes_)
        n_features = X_np.shape[1]

        self.feature_count_ = np.zeros((self.n_classes_, n_features), dtype=np.float64)
        self.class_count_ = np.zeros(self.n_classes_, dtype=np.float64)

        for i, c in enumerate(self.classes_):
            X_c = X_np[y_np == c]
            self.feature_count_[i] = np.sum(X_c, axis=0)
            self.class_count_[i] = len(X_c)

        # Apply smoothing
        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = np.sum(smoothed_fc, axis=1, keepdims=True)

        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)

        # Compute class priors
        if self.class_prior is not None:
            self.class_log_prior_ = np.log(np.array(self.class_prior))
        elif self.fit_prior:
            self.class_log_prior_ = np.log(self.class_count_ / np.sum(self.class_count_))
        else:
            self.class_log_prior_ = np.full(self.n_classes_, -np.log(self.n_classes_))

        return self

    def predict(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Perform classification on an array of test vectors X.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Predicted classes of shape (n_samples,)
        """
        if self.feature_log_prob_ is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        X_np = np.array(X)

        # Compute log likelihood
        jll = X_np @ self.feature_log_prob_.T + self.class_log_prior_
        predictions = self.classes_[np.argmax(jll, axis=1)]

        return mx.array(predictions.astype(np.int32))

    def predict_proba(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Return probability estimates for the test vectors X.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Probability of the sample for each class in the model, of shape (n_samples, n_classes)
        """
        if self.feature_log_prob_ is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        X_np = np.array(X)

        jll = X_np @ self.feature_log_prob_.T + self.class_log_prior_
        log_prob_x = np.logaddexp.reduce(jll, axis=1, keepdims=True)
        probas = np.exp(jll - log_prob_x)

        return mx.array(probas.astype(np.float32))


class BernoulliNB(BaseEstimator):
    """
    Naive Bayes classifier for multivariate Bernoulli models.

    Parameters:
        alpha: Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
        binarize: Threshold for binarizing (mapping to booleans) of sample features. If None,
            input is presumed to already consist of binary vectors.
        fit_prior: Whether to learn class prior probabilities or not. If False, a uniform prior will be used.
        class_prior: Prior probabilities of the classes. If specified, the priors are not adjusted according to the data.
    """
    def __init__(
        self,
        alpha: float = 1.0,
        binarize: Optional[float] = 0.0,
        fit_prior: bool = True,
        class_prior: Optional[Union[np.array, mx.array]] = None
    ):
        super().__init__()
        self.alpha = alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
        self.class_prior = class_prior

        self.classes_: Optional[np.array] = None
        self.n_classes_: Optional[int] = None
        self.feature_count_: Optional[np.array] = None
        self.class_count_: Optional[np.array] = None
        self.feature_log_prob_: Optional[np.array] = None
        self.class_log_prior_: Optional[np.array] = None

    def fit(self, X: Union[mx.array, np.array], y: Union[mx.array, np.array]) -> "BernoulliNB":
        """
        Fit Bernoulli Naive Bayes according to X, y.

        Parameters:
            X: Training data of shape (n_samples, n_features). Will be binarized if binarize is not None.
            y: Target values of shape (n_samples,)

        Returns:
            Fitted estimator
        """
        X, y = self._validate_data(X, y)
        X_np = np.array(X)
        y_np = np.array(y)

        # Binarize features if needed
        if self.binarize is not None:
            X_np = (X_np > self.binarize).astype(np.float64)

        self.classes_ = np.unique(y_np)
        self.n_classes_ = len(self.classes_)
        n_features = X_np.shape[1]

        self.feature_count_ = np.zeros((self.n_classes_, n_features), dtype=np.float64)
        self.class_count_ = np.zeros(self.n_classes_, dtype=np.float64)

        for i, c in enumerate(self.classes_):
            X_c = X_np[y_np == c]
            self.feature_count_[i] = np.sum(X_c, axis=0)
            self.class_count_[i] = len(X_c)

        # Apply smoothing
        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = self.class_count_ + 2 * self.alpha  # 2 outcomes for each feature

        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1))

        # Compute class priors
        if self.class_prior is not None:
            self.class_log_prior_ = np.log(np.array(self.class_prior))
        elif self.fit_prior:
            self.class_log_prior_ = np.log(self.class_count_ / np.sum(self.class_count_))
        else:
            self.class_log_prior_ = np.full(self.n_classes_, -np.log(self.n_classes_))

        return self

    def predict(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Perform classification on an array of test vectors X.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Predicted classes of shape (n_samples,)
        """
        if self.feature_log_prob_ is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        X_np = np.array(X)

        # Binarize features if needed
        if self.binarize is not None:
            X_np = (X_np > self.binarize).astype(np.float64)

        # Compute log likelihood (with negation for 0 features)
        neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
        jll = X_np @ (self.feature_log_prob_ - neg_prob).T + np.sum(neg_prob, axis=1) + self.class_log_prior_
        predictions = self.classes_[np.argmax(jll, axis=1)]

        return mx.array(predictions.astype(np.int32))

    def predict_proba(self, X: Union[mx.array, np.array]) -> mx.array:
        """
        Return probability estimates for the test vectors X.

        Parameters:
            X: Samples of shape (n_samples, n_features)

        Returns:
            Probability of the sample for each class in the model, of shape (n_samples, n_classes)
        """
        if self.feature_log_prob_ is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        X, _ = self._validate_data(X)
        X_np = np.array(X)

        # Binarize features if needed
        if self.binarize is not None:
            X_np = (X_np > self.binarize).astype(np.float64)

        neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
        jll = X_np @ (self.feature_log_prob_ - neg_prob).T + np.sum(neg_prob, axis=1) + self.class_log_prior_
        log_prob_x = np.logaddexp.reduce(jll, axis=1, keepdims=True)
        probas = np.exp(jll - log_prob_x)

        return mx.array(probas.astype(np.float32))
