import unittest

import torch

from torcheeg.models import EEGNetAttentionDANN


class TestEEGNetAttentionDANN(unittest.TestCase):

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(0)

    def test_forward_returns_expected_shape(self):
        model = EEGNetAttentionDANN(chunk_size=128,
                                    num_electrodes=32,
                                    num_classes=2).eval()
        eeg = torch.randn(2, 1, 32, 128)
        with torch.no_grad():
            pred = model(eeg)
        self.assertEqual(tuple(pred.shape), (2, 2))
        self.assertTrue(torch.isfinite(pred).all())

    def test_features_and_domain_heads(self):
        model = EEGNetAttentionDANN(chunk_size=128,
                                    num_electrodes=32,
                                    num_classes=3).eval()
        eeg = torch.randn(2, 1, 32, 128)
        with torch.no_grad():
            features = model.extract_features(eeg)
            domain_pred = model.predict_domain(features)
        self.assertEqual(tuple(features.shape), (2, 16))
        self.assertEqual(tuple(domain_pred.shape), (2, 2))
        self.assertTrue(torch.isfinite(features).all())
        self.assertTrue(torch.isfinite(domain_pred).all())

    def test_backward_produces_finite_gradients(self):
        model = EEGNetAttentionDANN(chunk_size=128, num_electrodes=8)
        eeg = torch.randn(2, 1, 8, 128, requires_grad=True)
        features = model.extract_features(eeg)
        loss = model.classifier(features).square().mean()
        loss = loss + model.predict_domain(features).square().mean()
        loss.backward()
        self.assertIsNotNone(eeg.grad)
        self.assertTrue(torch.isfinite(eeg.grad).all())
        self.assertTrue(eeg.grad.abs().sum() > 0)
        gradients = [
            parameter.grad for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all()
                            for gradient in gradients))

    def test_attention_only_ablation(self):
        model = EEGNetAttentionDANN(chunk_size=128,
                                    num_electrodes=32,
                                    use_attention=True,
                                    use_dann=False).eval()
        eeg = torch.randn(2, 1, 32, 128)
        with torch.no_grad():
            features = model.extract_features(eeg)
            pred = model(eeg)
        self.assertIsNone(model.domain_classifier)
        self.assertEqual(tuple(features.shape), (2, 16))
        self.assertEqual(tuple(pred.shape), (2, 2))
        with self.assertRaisesRegex(ValueError, 'use_dann'):
            model.predict_domain(features)

    def test_dann_only_ablation(self):
        model = EEGNetAttentionDANN(chunk_size=128,
                                    num_electrodes=32,
                                    use_attention=False,
                                    use_dann=True).eval()
        eeg = torch.randn(2, 1, 32, 128)
        with torch.no_grad():
            features = model.extract_features(eeg)
            domain_pred = model.predict_domain(features)
        self.assertEqual(tuple(features.shape), (2, 64))
        self.assertEqual(tuple(domain_pred.shape), (2, 2))

    def test_invalid_heads_raises(self):
        with self.assertRaisesRegex(ValueError, 'divisible'):
            EEGNetAttentionDANN(chunk_size=128, num_electrodes=32, heads=3)


if __name__ == '__main__':
    unittest.main()
