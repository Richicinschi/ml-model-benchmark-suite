"""Unit tests for the preprocessing pipeline."""

import unittest

import numpy as np
import pandas as pd

from benchmark.preprocessing import PreprocessingPipeline, build_preprocessing_pipeline


class TestPreprocessingPipeline(unittest.TestCase):
    """Tests for PreprocessingPipeline build, fit_transform, and target encoding."""

    def _make_mixed_df(self):
        return pd.DataFrame({
            "num_a": [1.0, 2.0, np.nan, 4.0],
            "num_b": [10, 20, 30, 40],
            "cat_a": ["x", "y", "x", np.nan],
            "cat_b": ["foo", "foo", "bar", "bar"],
        })

    def test_build_numeric_only_standard_scaling(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        pipeline = PreprocessingPipeline({"scale": "standard"}).build(df)
        Xt = pipeline.fit_transform(df)
        self.assertEqual(Xt.shape, (3, 2))
        # StandardScaler produces ~zero mean and unit variance
        self.assertAlmostEqual(Xt["a"].mean(), 0.0, places=5)
        self.assertAlmostEqual(Xt["a"].std(ddof=0), 1.0, places=5)

    def test_build_numeric_only_minmax_scaling(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        pipeline = PreprocessingPipeline({"scale": "minmax"}).build(df)
        Xt = pipeline.fit_transform(df)
        self.assertEqual(Xt.shape, (3, 2))
        self.assertAlmostEqual(Xt["a"].min(), 0.0, places=5)
        self.assertAlmostEqual(Xt["a"].max(), 1.0, places=5)

    def test_build_no_scaling(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        pipeline = PreprocessingPipeline({"scale": "none", "impute": False}).build(df)
        Xt = pipeline.fit_transform(df)
        pd.testing.assert_frame_equal(Xt, df)

    def test_boolean_scale_true_false(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        pipeline_true = PreprocessingPipeline({"scale": True, "impute": False}).build(df)
        Xt_true = pipeline_true.fit_transform(df)
        self.assertAlmostEqual(Xt_true["a"].mean(), 0.0, places=5)

        pipeline_false = PreprocessingPipeline({"scale": False, "impute": False}).build(df)
        Xt_false = pipeline_false.fit_transform(df)
        pd.testing.assert_frame_equal(Xt_false, df)

    def test_imputation_numeric_median(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        pipeline = PreprocessingPipeline({"impute": True, "scale": "none"}).build(df)
        Xt = pipeline.fit_transform(df)
        self.assertFalse(Xt.isnull().values.any())
        # Median of [1.0, 2.0, 4.0] is 2.0
        self.assertAlmostEqual(Xt["a"].iloc[2], 2.0, places=5)

    def test_imputation_categorical_most_frequent(self):
        df = pd.DataFrame({"c": ["x", "y", "x", np.nan]})
        pipeline = PreprocessingPipeline({"impute": True, "encode": False}).build(df)
        Xt = pipeline.fit_transform(df)
        self.assertFalse(Xt.isnull().values.any())
        self.assertEqual(Xt["c"].iloc[3], "x")

    def test_categorical_one_hot_encoding(self):
        df = pd.DataFrame({"c": ["a", "b", "a"]})
        pipeline = PreprocessingPipeline({"impute": False, "encode": True}).build(df)
        Xt = pipeline.fit_transform(df)
        # OneHotEncoder with sparse_output=False yields 2 columns for 2 categories
        self.assertEqual(Xt.shape, (3, 2))
        # Should contain 0 and 1 only
        self.assertTrue(set(Xt.values.flatten()).issubset({0, 1}))

    def test_mixed_numeric_and_categorical(self):
        df = self._make_mixed_df()
        pipeline = PreprocessingPipeline({"scale": "standard", "impute": True, "encode": True}).build(df)
        Xt = pipeline.fit_transform(df)
        self.assertEqual(len(Xt), 4)
        self.assertFalse(Xt.isnull().values.any())

    def test_transform_before_fit_raises(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        pipeline = PreprocessingPipeline({"scale": "standard"}).build(df)
        with self.assertRaises(RuntimeError):
            pipeline.transform(df)

    def test_fit_transform_then_transform(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        pipeline = PreprocessingPipeline({"scale": "standard", "impute": False}).build(df)
        pipeline.fit_transform(df)
        Xt2 = pipeline.transform(df)
        self.assertEqual(Xt2.shape, (3, 1))

    def test_target_label_encoding(self):
        y = pd.Series(["cat", "dog", "cat", "dog"], name="animal")
        pipeline = PreprocessingPipeline({"encode_target": True}).build(pd.DataFrame())
        y_enc = pipeline.fit_transform_target(y)
        self.assertEqual(sorted(y_enc.unique().tolist()), [0, 1])

    def test_target_inverse_transform(self):
        y = pd.Series(["cat", "dog", "cat"], name="animal")
        pipeline = PreprocessingPipeline({"encode_target": True}).build(pd.DataFrame())
        y_enc = pipeline.fit_transform_target(y)
        y_dec = pipeline.inverse_transform_target(y_enc)
        pd.testing.assert_series_equal(y_dec, y)

    def test_target_no_encoding(self):
        y = pd.Series([1, 2, 3])
        pipeline = PreprocessingPipeline({"encode_target": False}).build(pd.DataFrame())
        y_out = pipeline.fit_transform_target(y)
        pd.testing.assert_series_equal(y_out, y)

    def test_factory_function(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        pipeline, Xt = build_preprocessing_pipeline(df, {"scale": "minmax", "impute": False})
        self.assertIsInstance(pipeline, PreprocessingPipeline)
        self.assertAlmostEqual(Xt["a"].min(), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
