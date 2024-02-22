# %%
import numpy as np
import random
from scipy import sparse
from typing import Any


# TODO: implement the PCA with numpy
# Note that you are not allowed to use any existing PCA implementation from sklearn or other libraries.
class PrincipalComponentAnalysis:
    def __init__(self, n_components: int) -> None:
        """_summary_

        Parameters
        ----------
        n_components : int
            The number of principal components to be computed. This value should be less than or equal to the number of features in the dataset.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    # TODO: implement the fit method
    def fit(self, X: np.ndarray):
        """
        Fit the model with X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        means = []
        standard_deviations = []
        for column in range(X.shape[1]):
            mean = X[:, column].sum() / X.shape[0]
            means.append(mean)
            standard_deviations.append(X[:, column].std())
        self.mean = np.array(means)
        self.std = np.array(standard_deviations)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted from a training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.
        """
        X = X - X.mean(axis=0)
        X = X / X.std(axis=0)

        covariance_matrix = np.cov(X)
        eigenvalues, eigenvectors = sparse.linalg.eigs(
            covariance_matrix, k=self.n_components, which="LM"
        )
        self.components = eigenvectors
        eigenvalues[0 : self.n_components]
        U = eigenvectors[:, 0 : self.n_components]

        X_new = X @ U
        return X_new


# TODO: implement the LDA with numpy
# Note that you are not allowed to use any existing LDA implementation from sklearn or other libraries.
class LinearDiscriminantAnalysis:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.

        Hint:
        -----
        To implement LDA with numpy, follow these steps:
        1. Compute the mean vectors for each class.
        2. Compute the within-class scatter matrix.
        3. Compute the between-class scatter matrix.
        4. Compute the eigenvectors and corresponding eigenvalues for the scatter matrices.
        5. Sort the eigenvectors by decreasing eigenvalues and choose k eigenvectors with the largest eigenvalues to form a d×k dimensional matrix W.
        6. Use this d×k eigenvector matrix to transform the samples onto the new subspace.
        """

        means = []
        standard_deviations = []
        for column in range(X.shape[1]):
            mean = X[:, column].sum() / X.shape[0]
            means.append(mean)
            standard_deviations.append(X[:, column].std())
        self.mean = np.array(means)
        self.std = np.array(standard_deviations)

        data_dict = {}
        mean_vector = {}
        categories = np.unique(y)
        for category in categories:
            data_dict[category] = X[np.where(y == category), :]
            mean_vector[category] = data_dict[category].mean(axis=0)
        mean_vector["Global"] = X.mean(axis=0)

        self.components = categories

        Sw = None
        for category in categories:
            if Sw is None:
                Sw = np.cov(data_dict[category], rowvar=False)
            else:
                Sw = Sw + np.cov(data_dict[category], rowvar=False)

        Sb = None
        for category in categories:
            centralized_category_mean = mean_vector[category] - mean_vector["Global"]
            nk = data_dict[category].shape[0]
            if Sw is None:
                Sw = nk * np.outer(
                    (centralized_category_mean, centralized_category_mean.T)
                )
            else:
                Sw = Sw + (
                    nk
                    * np.outer((centralized_category_mean, centralized_category_mean.T))
                )

        self.Sw = Sw
        self.Sb = Sb

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted from a training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.
        """
        Sw_inv = np.linalg.inv(self.Sw)
        eigenvalues, eigenvectors = sparse.linalg.eigs(
            Sw_inv @ self.Sb, k=len(self.components) - 1, which="LM"
        )
        X_new = X @ eigenvectors

        return X_new


# TODO: Generating adversarial examples for PCA.
# We will generate adversarial examples for PCA. The adversarial examples are generated by creating two well-separated clusters in a 2D space. Then, we will apply PCA to the data and check if the clusters are still well-separated in the transformed space.
# Your task is to generate adversarial examples for PCA, in which
# the clusters are well-separated in the original space, but not in the PCA space. The separabilit of the clusters will be measured by the K-means clustering algorithm in the test script.
#
# Hint:
# - You can place the two clusters wherever you want in a 2D space.
# - For example, you can use `np.random.multivariate_normal` to generate the samples in a cluster. Repeat this process for both clusters and concatenate the samples to create a single dataset.
# - You can set any covariance matrix, mean, and number of samples for the clusters.
class AdversarialExamples:
    def __init__(self) -> None:
        pass

    def pca_adversarial_data(self, n_samples, n_features):
        """Generate adversarial examples for PCA

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.
        n_features : int
            The number of features.

        Returns
        -------
        X: ndarray of shape (n_samples, n_features)
            Transformed values.

        y: ndarray of shape (n_samples,)
            Cluster IDs. y[i] is the cluster ID of the i-th sample.

        """
        X = np.zeros([n_samples, n_features])
        y = np.zeros(n_samples)

        for sample_number in range(int(n_samples / 2)):
            y[sample_number] = 1
            X[sample_number, 0:2] = [1 + np.random.normal(), 7 * np.random.normal()]
        for sample_number in range(int(n_samples / 2), n_samples):
            y[sample_number] = 2
            X[sample_number, 0:2] = [
                10 + np.random.normal(),
                1 + 8 * np.random.normal(),
            ]

        X = X - X.mean(axis=0)
        X = X / X.std(axis=0)

        return X, y


# %%
