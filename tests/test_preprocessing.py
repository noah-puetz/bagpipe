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


class Test_ApplyThreshold(unittest.TestCase):
    def setUp(self):
        self.transformer = ApplyThreshold(by="value", threshold=5)

    def test_threshold_condition(self):
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
        expected_result = pd.Series(
            [False, False, False, False, False, True, True, True, True]
        )
        result = self.transformer._threshold_condition(df)
        pd.testing.assert_series_equal(result, expected_result, check_names=False)

    def test_process_group(self):
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
        expected_result = df
        result = self.transformer._process_group(df)
        pd.testing.assert_frame_equal(result, expected_result)

    def test_transform(self):
        df1 = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
        df2 = pd.DataFrame({"value": [10, 20, 30, 40, 50, 60, 70, 80, 90]})
        dflist = [df1, df2]
        expected_result = [df1[df1["value"] > 5], df2]
        result = self.transformer.transform(dflist)
        for res, exp in zip(result, expected_result):
            pd.testing.assert_frame_equal(res, exp)

    def test_threshold_cutting(self):
        df1 = pd.DataFrame({"value": [1, 10, 10, 5, 10, 10, 3, 10, 10]})
        df2 = pd.DataFrame({"value": [1, 10, 10, 5, 10, 10, 3, 10, 10]})
        dflist = [df1, df2]

        result = self.transformer.transform(dflist)

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
        self.assertIsInstance(result_df_ls[0], pd.DataFrame)

        self.assertEqual(len(result_df_ls), 2)
        self.assertEqual(result_df_ls[0].shape, self.df.shape)

        stand_scaler = StandardScaler()
        minmax_scaler = MinMaxScaler()
        expected_stand_values = stand_scaler.fit_transform(self.df[["stand_value"]])
        expected_minmax_values = minmax_scaler.fit_transform(self.df[["minmax_value"]])
        pd.testing.assert_series_equal(
            result_df_ls[0]["stand_value"],
            pd.Series(expected_stand_values.flatten()),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result_df_ls[0]["minmax_value"],
            pd.Series(expected_minmax_values.flatten()),
            check_names=False,
        )


class Test_Concat_Seperate(unittest.TestCase):
    def setUp(self):
        self.df1 = pd.DataFrame(
            {
                "value": [1, 10, 1, 1, 10, 1, 5, 5, 5],
            }
        )
        self.df2 = pd.DataFrame(
            {
                "value": [10, 1, 2, 4, 10, 3, 5, 6, 9],
            }
        )
        self.dflist = [self.df1, self.df2]

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
