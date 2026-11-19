"""Checks that test evaluation uses only the saved holdout and never refits."""
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

import preprocessing_restratified as training
import testing


class HeldoutEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(8)
        data = pd.DataFrame(rng.normal(size=(180, len(training.CHEMICALS))), columns=training.CHEMICALS)
        data['wine_type'] = 'red'
        data['quality_class'] = np.asarray(training.CLASSES)[np.arange(180) % 3]
        self.csv = self.root / 'classes_only.csv'
        data.to_csv(self.csv, index=False)
        _, data, X, _, _, development, reserved = training.prepare_data(self.csv)
        self.reserved = set(data.iloc[reserved].source_row)
        self.model_dir = self.root / 'model'
        self.model_dir.mkdir()
        self.output = self.root / 'test_outputs'
        source = training.data_source(self.csv)
        hidden = (8, 4)
        checkpoint = {'state_dict': training.network(hidden).state_dict(), 'hidden_layers': hidden,
                      'task': 'multiclass_classification', 'target': 'quality_class',
                      'classes': training.CLASSES, 'probability_class_order': training.CLASSES,
                      'features': training.FEATURES, 'data_source': source, 'training_rows': len(development),
                      'training_class_counts': {label: int(data.iloc[development].quality_class.eq(label).sum())
                                                for label in training.CLASSES}}
        torch.save(checkpoint, self.model_dir / 'model.pt')
        joblib.dump(training.StandardScaler().fit(X.iloc[development]), self.model_dir / 'scaler.joblib')
        metadata = {'data_source': source, 'split_sizes': {'development': len(development), 'test_reserved': len(reserved)}}
        (self.model_dir / 'metrics.json').write_text(json.dumps(metadata))
        pd.DataFrame([{'stage': 'holdout', 'split': split, 'source_row': int(data.iloc[i].source_row)}
                      for split, indices in [('development', development), ('test_reserved', reserved)]
                      for i in indices]).to_csv(self.model_dir / 'partition_assignments.csv', index=False)

    def test_only_saved_holdout_is_scored_without_refitting(self):
        hashes = {name: testing.file_hash(self.model_dir / name) for name in ['model.pt', 'scaler.joblib']}
        args = argparse.Namespace(csv=self.csv, model_dir=self.model_dir, output=self.output)
        with patch.object(training, 'fit_network', side_effect=AssertionError('Model must not be retrained')), \
                patch.object(training.StandardScaler, 'fit', side_effect=AssertionError('Scaler must not be refitted')), \
                contextlib.redirect_stdout(io.StringIO()):
            summary = testing.run_test(args)
        predictions = pd.read_csv(self.output / 'test_predictions.csv')
        self.assertEqual(set(predictions.source_row), self.reserved)
        self.assertFalse(predictions.source_row.duplicated().any())
        self.assertEqual(summary['test_metrics']['n'], len(self.reserved))
        self.assertEqual(set(predictions.split), {'test'})
        self.assertFalse(any('quality' in column for column in predictions))
        np.testing.assert_allclose(predictions[training.PROBABILITY_COLUMNS].sum(axis=1), 1, atol=1e-6)
        for name, digest in hashes.items():
            self.assertEqual(testing.file_hash(self.model_dir / name), digest)
        for name in ['test_accuracy.png', 'test_roc_curves.png', 'test_metrics.json']:
            self.assertTrue((self.output / name).is_file())

    def test_changed_source_is_rejected(self):
        data = pd.read_csv(self.csv)
        data.loc[0, 'alcohol'] += 1
        data.to_csv(self.csv, index=False)
        with self.assertRaisesRegex(ValueError, 'does not match the source'):
            testing.load_test_data(self.csv, self.model_dir)

    def test_training_and_test_overlap_is_rejected(self):
        path = self.model_dir / 'partition_assignments.csv'
        partitions = pd.read_csv(path)
        first_training_row = partitions.loc[partitions.split == 'development', 'source_row'].iloc[0]
        first_test_index = partitions.index[partitions.split == 'test_reserved'][0]
        partitions.loc[first_test_index, 'source_row'] = first_training_row
        partitions.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            testing.load_test_data(self.csv, self.model_dir)


if __name__ == '__main__':
    unittest.main()
