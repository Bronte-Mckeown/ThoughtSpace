from typing import Tuple
import numpy as np
import pandas as pd
from factor_analyzer import Rotator, calculate_bartlett_sphericity, calculate_kmo
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ThoughtSpace.plotting import save_wordclouds, plot_scree
from ThoughtSpace.utils import setupanalysis
import os




class basePCA(TransformerMixin, BaseEstimator):
    def __init__(self, n_components="infer",verbosity=1):
        self.n_components = n_components
        self.verbosity = verbosity
        self.path = None
        self.ogdf = None
    def check_stats(self, df: pd.DataFrame) -> None:
        """
        This function checks the KMO and Bartlett Sphericity of the dataframe.
        Args:
            df: The dataframe to check.
        Returns:
            None
        """
        if self.verbosity > 0:
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

    def check_inputs(
        self, df: pd.DataFrame, fit: bool = False, project: bool = False
    ) -> pd.DataFrame:
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
        if project:
            self.project_columns = df.copy()
        if isinstance(df, pd.DataFrame):
            dtypes = df.dtypes
            for col in dtypes.index:
                if fit and dtypes[col] in [np.int64, np.float64, np.int32, np.float32]:
                    self.extra_columns.drop(col, axis=1, inplace=True)
                if project and dtypes[col] in [np.int64, np.float64, np.int32, np.float32]:
                    self.project_columns.drop(col, axis=1, inplace=True)
                if dtypes[col] not in [np.int64, np.float64, np.int32, np.float32]:
                    df.drop(col, axis=1, inplace=True)
            if fit:
                self.items = df.columns.tolist()
        else:
            self.items = [f"item_{x}" for x in range(df.shape[1])]
            self.extra_columns = pd.DataFrame()
        return df

    def z_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        This function returns the z-score of the dataframe.
        Args:
            df: The dataframe to be scaled.
        Returns:
            The z-score of the dataframe.
        """
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(df)

    def naive_pca(self, df: pd.DataFrame) -> Tuple[PCA, pd.DataFrame]:  # type: ignore
        """
        This is a multi-line Google style docstring.
        Args:
            df (pd.DataFrame): The dataframe to be used for PCA.
        Returns:
            Tuple[PCA, pd.DataFrame]: The PCA object and the loadings dataframe.
        """
        if self.n_components == "infer":
            self.fullpca = PCA().fit(df)
            self.n_components = len([x for x in self.fullpca.explained_variance_ if x >= 1])
            if self.verbosity > 0:
                print(f"Inferred number of components: {self.n_components}")
        else:
            self.fullpca = PCA().fit(df)
        pca = PCA(n_components=self.n_components).fit(df)
        loadings = Rotator().fit_transform(pca.components_.T)
        loadings = pd.DataFrame(
            loadings,
            index=self.items,
            columns=[f"PC{x+1}" for x in range(self.n_components)],
        )
        averages = loadings.mean(axis=0).to_dict()
        for col in averages:
            if averages[col] < 0:
                print(f"Component {col} has mostly negative loadings, flipping component")
                loadings[col] = loadings[col] * -1
        return loadings

    def fit(self, df: pd.DataFrame, y=None, scale: bool = True, **kwargs) -> "PCA":
        """
        Fit the PCA model.
        Args:
            df: The input dataframe.
            y: The target variable.
            **kwargs: The keyword arguments.
        Returns:
            The fitted PCA model.
        """
        _df = df.copy()
        if self.ogdf is None:
            self.ogdf = _df.copy()
        _df = self.check_inputs(_df, fit=True)
        self.check_stats(_df)
        if scale:
            _df = self.z_score(_df)
        self.loadings = self.naive_pca(_df)
        return self

    def transform(
        self,
        df: pd.DataFrame,
        scale=True,
    ) -> pd.DataFrame:
        df = self.check_inputs(df,project=True)
        
        if scale:
            df = self.scaler.transform(df)
        output_ = np.dot(df, self.loadings).T
        for x in range(self.n_components):
            self.project_columns[f"PCA_{x}"] = output_[x, :]
        return self.project_columns.copy()

    def project(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df.copy())

    def fit_project(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).project(df)
    
    def save(self,group=None,path=None,pathprefix="analysis",includetime=True) -> None:
        if self.path is None:

            self.path = setupanalysis(path,pathprefix,includetime)
        if group is None:
            os.makedirs(os.path.join(self.path,"wordclouds"),exist_ok=True)
            os.makedirs(os.path.join(self.path,"csvdata"),exist_ok=True)
            os.makedirs(os.path.join(self.path,"screeplots"),exist_ok=True)
            
            save_wordclouds(self.loadings,os.path.join(self.path,"wordclouds"))
            self.extra_columns.to_csv(os.path.join(self.path,"csvdata","pca_scores.csv"))
            self.loadings.to_csv(os.path.join(self.path,"csvdata","pca_loadings.csv"))
            pd.concat([self.ogdf, self.check_inputs(self.extra_columns)], axis=1).to_csv(os.path.join(self.path,"csvdata","pca_scores_original_format.csv"))
            plot_scree(self.fullpca,os.path.join(self.path, "screeplots", "scree"))
        
        else:
            os.makedirs(os.path.join(self.path,f"wordclouds_{group}"),exist_ok=True)
            os.makedirs(os.path.join(self.path,f"csvdata_{group}"),exist_ok=True)
            os.makedirs(os.path.join(self.path,"screeplots"),exist_ok=True)

            save_wordclouds(self.loadings,os.path.join(self.path,f"wordclouds_{group}"))
            self.extra_columns.to_csv(os.path.join(self.path,f"csvdata_{group}","pca_scores.csv"))
            self.loadings.to_csv(os.path.join(self.path,f"csvdata_{group}","pca_loadings.csv"))
            pd.concat([self.ogdf, self.check_inputs(self.extra_columns)], axis=1).to_csv(os.path.join(self.path,f"csvdata_{group}","pca_scores_original_format.csv"))
            plot_scree(self.fullpca,os.path.join(self.path, "screeplots", f"scree_{group}"))
        
        print(f"Saving done. Results have been saved to {self.path}")


class groupedPCA(basePCA):
    def __init__(self, grouping_col=None, n_components="infer", **kwargs):
        """
        Initialize the class.

        Args:
            grouping_col: The column to group by.
            n_components: The number of components to use.
            kwargs: Additional keyword arguments.
        """
        super().__init__(n_components)
        self.grouping_col = grouping_col
        if grouping_col is None:
            raise ValueError("Must specify a grouping column.")

    def z_score_byitem(self, df_dict) -> pd.DataFrame:
        """
        This function is used to calculate the z-score of the dataframe.

        Args:
            df_dict (dict): Dictionary of dataframes.

        Returns:
            pd.DataFrame: Dataframe with z-score.
        """
        self.scalerdict = {}
        outdict = []
        for key, value in df_dict.items():
            scaler = StandardScaler()
            value_ = self.check_inputs(value, fit=True)
            value_scaled = scaler.fit_transform(value_)
            extcol = self.extra_columns.copy().assign(
                **dict(zip(self.items, value_scaled.T))
            )
            self.scalerdict[key] = scaler
            outdict.append(extcol)
            
        return pd.concat(outdict, axis=0)

    def z_score_byitem_project(self, df_dict) -> pd.DataFrame:
        """
        This function takes a dictionary of dataframes and returns a dataframe with z-scored values.

        Args:
            df_dict (dict): A dictionary of dataframes.

        Returns:
            pd.DataFrame: A dataframe with z-scored values.
        """
        outdict = []
        for key, value in df_dict.items():
            value_ = self.check_inputs(value, project=True)
            try:
                scaler = self.scalerdict[key]
            except Exception:
                print(
                    f"Encountered a group in the data that wasn't seen while fitting: {key}. New group will be zscored individually."
                )
                scaler = StandardScaler()
                scaler.fit(value_)
            value_scaled = scaler.transform(value_)
            extcol = self.project_columns.copy().assign(
                **dict(zip(self.items, value_scaled.T))
            )
            outdict.append(extcol)
        return pd.concat(outdict, axis=0)

    def fit(self, df: pd.DataFrame, y=None, **kwargs):
        """
        This is a multi-line Google style docstring.

        Args:
            df (pd.DataFrame): The dataframe to fit.
            y (pd.Series): The target variable.
            **kwargs: Additional keyword arguments.

        Returns:
            self
        """
        self.ogdf = df.copy()
        d = dict(tuple(df.groupby(self.grouping_col)))
        zdf = self.z_score_byitem(d)
        super().fit(zdf, y=y, scale=False, **kwargs)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = dict(tuple(df.groupby(self.grouping_col)))
        zdf = self.z_score_byitem_project(d)
        return super().transform(zdf, scale=False)
    
    def save(self,savebygroup=False,path=None,pathprefix="analysis",includetime=True):
        self.path = setupanalysis(path,"grouped_"+pathprefix,includetime)
        if savebygroup:
            raise NotImplementedError("Saving by group is not yet implemented.")
            
        else:
            super().save()