"""Regression guards for the active classification workflows."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import torch

import preprocessing_restratified as workflow
import train_wine_random_forest as random_forest
import testing


class NeuralClassifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(7)
        self.data = pd.DataFrame(rng.normal(size=(180, len(workflow.CHEMICALS))),
                                 columns=workflow.CHEMICALS)
        self.data['wine_type'] = np.where(np.arange(180) % 2, 'red', 'white')
        self.data['quality_class'] = np.asarray(workflow.CLASSES)[np.arange(180) % 3]
        self.csv = self.root / 'classes_only.csv'
        self.data.to_csv(self.csv, index=False)

    def test_numerical_quality_does_not_affect_targets_features_or_splits(self):
        original = workflow.prepare_data(self.csv)
        altered = self.data.copy()
        altered['quality'] = ['unused'] * len(altered)
        # A score-only difference must not retain a duplicate training sample.
        duplicate = altered.iloc[[0]].assign(quality='different unused score')
        altered = pd.concat([altered, duplicate], ignore_index=True)
        altered.to_csv(self.root / 'with_score.csv', index=False)
        with_score = workflow.prepare_data(self.root / 'with_score.csv')
        pd.testing.assert_frame_equal(original[1], with_score[1])
        pd.testing.assert_frame_equal(original[2], with_score[2])
        for a, b in zip(original[3:], with_score[3:]):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(np.asarray(workflow.CLASSES)[original[3]], self.data.quality_class)
        self.assertNotIn('quality', original[1])
        self.assertNotIn('quality_class', original[2])

    def test_classifier_outputs_and_saved_model_with_class_only_input(self):
        args = argparse.Namespace(csv=self.csv, output=self.root / 'outputs',
                                  epochs=2, patience=1, folds=2)
        with patch.object(workflow, 'CANDIDATES', [(32, 16)]), contextlib.redirect_stdout(io.StringIO()):
            workflow.run_neural_network(args)
        predictions = pd.read_csv(args.output / 'predictions.csv')
        probabilities = predictions[workflow.PROBABILITY_COLUMNS].to_numpy()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
        np.testing.assert_array_equal(predictions.predicted_class,
                                      np.asarray(workflow.CLASSES)[probabilities.argmax(axis=1)])
        self.assertTrue((probabilities >= 0).all() and (probabilities <= 1).all())
        self.assertFalse(any('quality' in column or 'rounded' in column for column in predictions))
        summary = json.loads((args.output / 'metrics.json').read_text())
        self.assertEqual(summary['target'], 'quality_class')
        self.assertFalse(summary['test_evaluated'])
        self.assertEqual(len(predictions), summary['split_sizes']['development'])
        partitions = pd.read_csv(args.output / 'partition_assignments.csv')
        reserved = set(partitions.loc[partitions.split == 'test_reserved', 'source_row'])
        self.assertFalse(set(predictions.source_row) & reserved)
        checkpoint = torch.load(args.output / 'model.pt', weights_only=True)
        self.assertEqual(checkpoint['classes'], workflow.CLASSES)
        self.assertNotIn('target_mean', checkpoint)
        model = workflow.network(checkpoint['hidden_layers'])
        model.load_state_dict(checkpoint['state_dict'])
        scaler = joblib.load(args.output / 'scaler.joblib')
        X = workflow.feature_matrix(self.data.iloc[:3])
        reloaded = workflow.predict(model, scaler, X)
        self.assertEqual(reloaded.shape, (3, 3))
        np.testing.assert_allclose(reloaded.sum(axis=1), 1, atol=1e-6)
        self.assertFalse((args.output / 'validation_metrics_by_quality.csv').exists())
        counts = pd.read_csv(args.output / 'training_rows.csv')
        self.assertTrue(counts.repeated_source_rows.eq(0).all())

        # The forest must reuse saved neural splits without changing neural artifacts.
        forest_args = argparse.Namespace(csv=self.csv, neural_dir=args.output,
                                         output=self.root / 'forest', trees=8, jobs=1)
        with patch.object(testing, 'run_test', side_effect=AssertionError('Training must not score the test set')), \
                contextlib.redirect_stdout(io.StringIO()):
            forest_summary = random_forest.run(forest_args)
        forest_predictions = pd.read_csv(forest_args.output / 'predictions.csv')
        self.assertFalse(forest_summary['test_evaluated'])
        self.assertNotIn('test_metrics', forest_summary)
        self.assertFalse((forest_args.output / 'testing').exists())
        test_args = argparse.Namespace(model='random-forest', csv=self.csv,
                                       model_dir=forest_args.output, output=forest_args.output / 'testing')
        with patch.object(random_forest.RandomForestClassifier, 'fit', side_effect=AssertionError('Testing must not retrain')), \
                patch.object(workflow.StandardScaler, 'fit', side_effect=AssertionError('Testing must not refit scaling')), \
                contextlib.redirect_stdout(io.StringIO()):
            forest_test_summary = testing.run_test(test_args)
        self.assertEqual(forest_test_summary['evaluator'], 'testing.py')
        forest_test = pd.read_csv(forest_args.output / 'testing' / 'test_predictions.csv')
        self.assertEqual(set(forest_predictions.source_row), set(predictions.source_row))
        self.assertEqual(set(forest_test.source_row), reserved)
        self.assertFalse(set(forest_test.source_row) & set(forest_predictions.source_row))
        original_folds = pd.read_csv(args.output / 'fold_assignments.csv').set_index('source_row').fold.sort_index()
        forest_folds = forest_predictions.set_index('source_row').fold.sort_index()
        pd.testing.assert_series_equal(forest_folds, original_folds)
        self.assertEqual(forest_summary['parameters']['n_estimators'], 8)
        self.assertFalse(forest_summary['parameters']['bootstrap'])
        self.assertNotIn('quality', forest_summary['features'])
        forest_pipeline = joblib.load(forest_args.output / 'model.joblib')
        self.assertIsInstance(forest_pipeline.named_steps['classifier'], random_forest.RandomForestClassifier)
        self.assertEqual(len(forest_pipeline.named_steps['classifier'].estimators_), 8)
        forest_X = workflow.feature_matrix(pd.read_csv(self.csv).iloc[forest_test.source_row])
        np.testing.assert_allclose(random_forest.aligned_probabilities(forest_pipeline, forest_X),
                                   forest_test[workflow.PROBABILITY_COLUMNS].to_numpy(), atol=1e-12)

    def test_knn_keeps_the_requested_three_neighbours(self):
        args = argparse.Namespace(csv=self.csv, output=self.root / 'knn', folds=2)
        with contextlib.redirect_stdout(io.StringIO()):
            workflow.run_knn(args)
        summary = json.loads((args.output / 'metrics.json').read_text())
        self.assertEqual(summary['n_neighbors'], 3)
        self.assertEqual(summary['k_selection'], 'fixed by user')
        for filename in ['predictions.csv', 'fold_metrics.csv']:
            self.assertEqual(set(pd.read_csv(args.output / filename).k), {3})
        self.assertFalse((args.output / 'tuning.csv').exists())

    def test_roc_metrics_preserve_tied_forest_probabilities(self):
        actual = np.array([0, 1, 2, 0, 1, 2])
        probabilities = np.tile([0.1, 0.2, 0.7], (6, 1))
        probabilities[0, 2] += 1e-15
        expected = workflow.classification_roc_auc(np.asarray(workflow.CLASSES)[actual], probabilities)
        scores = workflow.metrics(actual, probabilities)
        self.assertAlmostEqual(scores['roc_auc_macro'], expected['macro'], places=15)


if __name__ == '__main__':
    unittest.main()
