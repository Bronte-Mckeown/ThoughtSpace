import numpy as np
import pandas as pd
from factor_analyzer import Rotator, calculate_bartlett_sphericity, calculate_kmo
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple


class basePCA(TransformerMixin, BaseEstimator):
    def __init__(self, n_components="infer"):
        self.n_components = n_components

    def check_stats(self, df: pd.DataFrame) -> None:
        """
        This function checks the KMO and Bartlett Sphericity of the dataframe.

        Args:
            df: The dataframe to check.

        Returns:
            None
        """
        bart = calculate_bartlett_sphericity(df)
        if bart[1] < 0.05:
            print("Bartlett Sphericity is acceptable. The p-value is %.3f" % bart[1])
        else:
            print(
                "Bartlett Sphericity is unacceptable. Something is very wrong with your data. The p-value is %.3f"
                % bart[1]
            )
        kmo = calculate_kmo(df)
        k = kmo[1]
        if k < 0.5:
            print(
                "KMO score is unacceptable. The value is %.3f, you should not trust your data."
                % k
            )
        if 0.6 > k > 0.5:
            print(
                "KMO score is miserable. The value is %.3f, you should consider resampling or continuing data collection."
                % k
            )
        if 0.7 > k > 0.6:
            print(
                "KMO score is mediocre. The value is %.3f, you should consider continuing data collection, or use the data as is."
                % k
            )
        if 0.8 > k > 0.7:
            print(
                "KMO score is middling. The value is %.3f, your data is perfectly acceptable, but could benefit from more sampling."
                % k
            )
        if 0.9 > k > 0.8:
            print(
                "KMO score is meritous. The value is %.3f, your data is perfectly acceptable."
                % k
            )
        if k > 0.9:
            print(
                "KMO score is marvelous. The value is %.3f, what demon have you sold your soul to to collect this data? Please email me."
                % k
            )

    def check_inputs(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Check the inputs of the function.

        Args:
            df: The input dataframe.
            fit: Whether the function is in fit mode.

        Returns:
            The processed dataframe.
        """
        if fit:
            self.extra_columns = df.copy()
        dtypes = df.dtypes
        for col in dtypes.index:
            if fit and dtypes[col] in [np.int64, np.float64, np.int32, np.float32]:
                self.extra_columns.drop(col, axis=1, inplace=True)
            if dtypes[col] not in [np.int64, np.float64, np.int32, np.float32]:
                df.drop(col, axis=1, inplace=True)
        if fit:
            self.items = df.columns.tolist()
        return df

    def z_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This function returns the z-score of the dataframe.

        Args:
            df: The dataframe to be scaled.

        Returns:
            The z-score of the dataframe.
        """
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(df)

    def naive_pca(self, df: pd.DataFrame) -> Tuple[PCA, pd.DataFrame]:
        """
        This is a multi-line Google style docstring.

        Args:
            df (pd.DataFrame): The dataframe to be used for PCA.

        Returns:
            Tuple[PCA, pd.DataFrame]: The PCA object and the loadings dataframe.
        """
        if self.n_components == "infer":
            pca = PCA().fit(df)
            self.n_components = len([x for x in pca.explained_variance_ if x >= 1])
            print(f"Inferred number of components: {self.n_components}")
        pca = PCA(n_components=self.n_components).fit(df)
        loadings = Rotator().fit_transform(pca.components_.T)
        loadings = pd.DataFrame(
            loadings,
            index=self.items,
            columns=[f"PC{x}" for x in range(self.n_components)],
        )
        return pca, loadings

    def fit(self, df: pd.DataFrame, y=None, **kwargs) -> "PCA":
        """
        Fit the PCA model.

        Args:
            df: The input dataframe.
            y: The target variable.
            **kwargs: The keyword arguments.

        Returns:
            The fitted PCA model.
        """
        df = self.check_inputs(df, fit=True)
        self.check_stats(df)
        df_z = self.z_score(df)
        self.pca, self.loadings = self.naive_pca(df_z)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.check_inputs(df)
        zdf = self.scaler.transform(df)
        output_ = np.dot(zdf, self.loadings).T
        for x in range(self.n_components):
            self.extra_columns[f"PCA_{x}"] = output_[x, :]
        return self.extra_columns.copy()

    def project(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)

    def fit_project(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).project(df)
