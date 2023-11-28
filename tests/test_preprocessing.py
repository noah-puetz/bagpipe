import unittest
import pandas as pd
from bagpipe.preprocessing import (
    ApplyThreshold,
    _ConcatDataFrames,
    _SeparateDataFrames,
    SkScalerWrapper,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


df1 = pd.DataFrame(
    {
        "value_1": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "value_2": [10, 20, 30, 40, 50, 60, 70, 80, 90],
        "value_3": [1, 10, 10, 5, 10, 10, 3, 10, 10],
    },
)

df2 = pd.DataFrame(
    {
        "value_1": [11, 12, 13, 14, 15, 16, 17, 18, 19],
        "value_2": [20, 30, 40, 50, 60, 70, 80, 90, 99],
        "value_3": [1, 10, 10, 5, 10, 10, 3, 10, 10],
    },
)


class Test_ApplyThreshold(unittest.TestCase):
    def setUp(self):
        self.transformer_v1 = ApplyThreshold(by="value_1", threshold=5)
        self.transformer_v2 = ApplyThreshold(by="value_3", threshold=5)

    def test_threshold_condition(self):
        expected_result = pd.Series(
            [False, False, False, False, False, True, True, True, True]
        )
        result = self.transformer_v1._threshold_condition(df1)
        pd.testing.assert_series_equal(result, expected_result, check_names=False)

    def test_process_group(self):
        expected_result = df1
        result = self.transformer_v1._process_group(df1)
        pd.testing.assert_frame_equal(result, expected_result)

    def test_transform(self):
        dflist = [df1, df2]
        expected_result = [df1[df1["value_1"] > 5], df2]
        result = self.transformer_v1.transform(dflist)
        for res, exp in zip(result, expected_result):
            pd.testing.assert_frame_equal(res, exp)

    def test_threshold_cutting(self):
        dflist = [df1, df2]

        result = self.transformer_v2.transform(dflist)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 6)

        for res in result:
            self.assertIsInstance(res, pd.DataFrame)
            self.assertEqual(len(res), 2)


class Test_BagpipePipline(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "minmax_value": [1, 10, 1, 1, 10, 1, 5, 5, 5],
                "stand_value": [-4, -3, -2, -1, 0, 1, 2, 3, 4],
                "remainder": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            }
        )

    def test_column_transformer_pipeline(self):
        ct = ColumnTransformer(
            [
                ("stand", StandardScaler(), ["stand_value"]),
                ("minmax", MinMaxScaler(), ["minmax_value"]),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

        pipeline = Pipeline(
            [
                ("concat", _ConcatDataFrames()),
                ("column transformer", ct.set_output(transform="pandas")),
                ("separate", _SeparateDataFrames()),
            ]
        )

        concat_df = pipeline.fit_transform([self.df, self.df])

        self._test_pipline_results(concat_df)

    def test_scaler_wrapper_pipeline(self):
        pipline = Pipeline(
            [
                ("stand scaler", SkScalerWrapper(StandardScaler(), ["stand_value"])),
                ("minmax scaler", SkScalerWrapper(MinMaxScaler(), ["minmax_value"])),
            ]
        )

        concat_df = pipline.fit_transform([self.df, self.df])

        self._test_pipline_results(concat_df)

    def _test_pipline_results(self, result_df_ls):
        self.assertIsInstance(result_df_ls, list)
        for res in result_df_ls:
            self.assertIsInstance(res, pd.DataFrame)

        self.assertEqual(len(result_df_ls), 2)
        for res in result_df_ls:
            self.assertEqual(res.shape, self.df.shape)

        stand_scaler = StandardScaler()
        minmax_scaler = MinMaxScaler()
        expected_stand_values = stand_scaler.fit_transform(self.df[["stand_value"]])
        expected_minmax_values = minmax_scaler.fit_transform(self.df[["minmax_value"]])

        for pd_df in result_df_ls:
            self.assertTrue("stand_value" in pd_df.columns)
            self.assertTrue("minmax_value" in pd_df.columns)
            self.assertTrue("remainder" in pd_df.columns)

            pd.testing.assert_series_equal(
                pd_df["stand_value"],
                pd.Series(expected_stand_values.flatten()),
                check_names=False,
            )

            pd.testing.assert_series_equal(
                pd_df["minmax_value"],
                pd.Series(expected_minmax_values.flatten()),
                check_names=False,
            )

            pd.testing.assert_series_equal(
                pd_df["remainder"],
                self.df["remainder"],
                check_names=False,
            )


class Test_Concat_Seperate(unittest.TestCase):
    def setUp(self):
        self.dflist = [df1, df2]

    def test_concat_seperate(self):
        concat = _ConcatDataFrames()
        result = concat.transform(self.dflist)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 18)

        seperate = _SeparateDataFrames()
        result = seperate.transform(result)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for i, res in enumerate(result):
            self.assertIsInstance(res, pd.DataFrame)
            self.assertEqual(len(res), 9)
            pd.testing.assert_frame_equal(res, self.dflist[i])


if __name__ == "__main__":
    unittest.main()
