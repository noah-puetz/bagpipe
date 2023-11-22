from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from torch.utils.data import ConcatDataset
import pandas as pd
import numpy as np


class ApplyThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, by, threshold, seq_length=1, threshold_as_upper_bound=False):
        self.by = by
        self.threshold = threshold
        self.seq_length = seq_length
        self.threshold_as_upper_bound = threshold_as_upper_bound

    def fit(self, X, y=None):
        return self

    def _threshold_condition(self, df):
        if self.threshold_as_upper_bound:
            return df[self.by] < self.threshold
        else:
            return df[self.by] > self.threshold

    def _process_group(self, group):
        if len(group) > self.seq_length:
            return group

    def transform(self, dflist):
        new_dflist = []
        for df in dflist:
            df_mask = self._threshold_condition(df)
            groups = df[df_mask].groupby((~df_mask).cumsum())
            new_data = [
                self._process_group(group)
                for _, group in groups
                if self._process_group(group) is not None
            ]
            new_dflist.extend(new_data)

        return new_dflist


class CreateConcatDataset(BaseEstimator, TransformerMixin):
    def __init__(self, dataset_class, **kwargs):
        self.dataset_class = dataset_class
        self.dataset_args = kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, dflist):
        dslist = []
        for df in dflist:
            dslist.append(self.dataset_class(df, **self.dataset_args))
        datasets = ConcatDataset(dslist)
        return datasets


class SkScalerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, scaler, columns=None):
        if hasattr(scaler, "partial_fit"):
            self.scaler = scaler.set_output(transform="pandas")
        else:
            raise ValueError(
                "scaler must have partial_fit method to be used with the bagpipe package"
            )
        if columns is not None and not isinstance(columns, list):
            columns = [columns]
        self.columns = columns

    def fit(self, dflist, y=None):
        self.scaler._reset()
        for df in dflist:
            if self.columns is None:
                self.scaler.partial_fit(df)
            else:
                self.scaler.partial_fit(df[self.columns])
        return self

    def transform(self, dflist):
        new_dflist = []
        for df in dflist:
            df_copy = df.copy()
            if self.columns is not None:
                df_copy[self.columns] = self.scaler.transform(df[self.columns])
            else:
                df_copy = self.scaler.transform(df)
            new_dflist.append(df_copy)
        return new_dflist


class _ConcatDataFrames(BaseEstimator, TransformerMixin):
    def fit(self, dflist, y=None):
        return self

    def transform(self, dflist):
        return pd.concat(dflist, keys=np.arange(0, len(dflist), 1))


class _SeparateDataFrames(BaseEstimator, TransformerMixin):
    def fit(self, dflist, y=None):
        return self

    def transform(self, df):
        return [df.xs(i) for i in df.index.get_level_values(0).unique().to_list()]
