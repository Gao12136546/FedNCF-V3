import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from data import SampleGenerator
from utils import compute_metrics


def make_generator(test_neg=2, train_neg=2):
    # Each user's held-out item belongs to another user's training set.
    frame = pd.DataFrame(
        [(0, 0, 5, 1), (0, 1, 4, 2),
         (1, 1, 3, 1), (1, 2, 5, 2),
         (2, 3, 2, 1), (2, 4, 5, 2)],
        columns=['userId', 'itemId', 'rating', 'timestamp'])
    return SampleGenerator(
        {'dataset': 'synthetic', 'NUM_NEG': test_neg, 'num_negative': train_neg}, frame)


class NegativeSamplingTests(unittest.TestCase):
    def setUp(self):
        self.random_state = random.getstate()
        random.seed(42)

    def tearDown(self):
        random.setstate(self.random_state)

    def test_all_observed_items_excluded_per_user(self):
        gen = make_generator()
        observed = gen.preprocess_ratings.groupby('userId')['itemId'].apply(set)
        for row in gen.negatives.itertuples():
            self.assertEqual(set(row.negative_items), gen.item_pool - observed[row.userId])
            for samples in (row.val_negative_samples, row.tst_negative_samples):
                self.assertEqual(len(samples), 2)
                self.assertEqual(len(set(samples)), 2)
                self.assertFalse(set(samples) & observed[row.userId])
        # An item observed by one user remains a valid negative for another.
        self.assertIn(0, gen.negatives.set_index('userId').loc[1, 'negative_items'])

    def test_training_and_pseudo_negatives_exclude_held_out(self):
        gen = make_generator()
        observed = gen.preprocess_ratings.groupby('userId')['itemId'].apply(set)
        users, items, labels = gen.store_all_train_data(2)
        for us, its, ys in zip(users, items, labels):
            self.assertEqual(len(ys), 3)
            for user, item, label in zip(us, its, ys):
                if label == 0:
                    self.assertNotIn(item, observed[user])
        gen.pseudo_item_embeddings = {u: torch.zeros(2, 3) for u in gen.user_pool}
        for user, batches in gen.store_pseudo_train_data(2).items():
            for batch in batches:
                self.assertEqual(len(batch), 2)
                self.assertFalse(set(batch) & observed[user])

    def test_candidates_and_metrics(self):
        gen = make_generator()
        data = gen.test_data
        self.assertTrue(all(t.device.type == 'cpu' for t in data))
        for user, positive in zip(data[0].tolist(), data[1].tolist()):
            negatives = data[3][data[2] == user].tolist()
            self.assertNotIn(positive, negatives)
            self.assertEqual(len(set(negatives + [positive])), 3)
        recall, ndcg = compute_metrics(data, torch.ones(3), torch.zeros(6), [1, 3])
        self.assertEqual(recall, [1.0, 1.0])
        self.assertEqual(ndcg, [1.0, 1.0])

    def test_insufficient_pool_fails_without_reintroducing_positives(self):
        for test_neg, train_neg in [(4, 2), (2, 4)]:
            with self.subTest(test_neg=test_neg, train_neg=train_neg):
                with self.assertRaisesRegex(ValueError, 'User 0 has 3 unobserved items; need 4'):
                    make_generator(test_neg, train_neg)
        gen = make_generator(3, 3)  # Exact pool size is supported.
        self.assertEqual(len(gen.test_data[3]), 9)
        self.assertEqual(len(gen.store_all_train_data(3)[0][0]), 4)

    def test_existing_cache_not_overwritten(self):
        gen = make_generator()
        with tempfile.TemporaryDirectory() as directory:
            gen.dataset_dir = directory + '/'
            paths = [Path(directory) / name for name in ('train.npy', 'val.npy', 'test.npy')]
            for path in paths:
                path.write_bytes(b'historical-cache')
            with patch('data.torch.load', side_effect=['old-val', 'old-test', 'old-train']), \
                    patch('data.torch.save') as save:
                self.assertEqual(gen.get_data(), ('old-train', 'old-val', 'old-test'))
                save.assert_not_called()
            self.assertTrue(all(p.read_bytes() == b'historical-cache' for p in paths))


if __name__ == '__main__':
    unittest.main()
