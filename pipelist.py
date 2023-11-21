from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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


class ConcatDataFrames(BaseEstimator, TransformerMixin):
    def fit(self, dflist, y=None):
        return self

    def transform(self, dflist):
        return pd.concat(dflist, keys=np.arange(0, len(dflist), 1))


class SeparateDataFrames(BaseEstimator, TransformerMixin):
    def fit(self, dflist, y=None):
        return self

    def transform(self, df):
        return [df.xs(i) for i in df.index.get_level_values(0).unique().to_list()]


# EXPERIMENTAL
class PL_StandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler().set_output(transform="pandas")
        return None

    def fit(self, dflist, y=None):
        self.scaler._reset()
        for df in dflist:
            self.scaler.partial_fit(df)
        return self

    def transform(self, dflist):
        return [self.scaler.transform(df) for df in dflist]


# EXPERIMENTAL
class PL_MinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = MinMaxScaler().set_output(transform="pandas")
        return None

    def fit(self, dflist, y=None):
        self.scaler._reset()
        for df in dflist:
            self.scaler.partial_fit(df)
        return self

    def transform(self, dflist):
        return [self.scaler.transform(df) for df in dflist]
