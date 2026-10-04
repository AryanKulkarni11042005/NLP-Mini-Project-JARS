import math
import importlib.util
import tempfile
import unittest

import pandas as pd

from evaluation.classification_metrics import classification_scores
from evaluation.robustness_metrics import (
    exploratory_correlations,
    robustness_table,
    summarize_replicates,
)
from evaluation.plots import (
    plot_classification_robustness,
    plot_normalization_recovery,
    plot_quality_correlations,
    plot_text_quality,
)
from evaluation.text_metrics import (
    calculate_bleu,
    calculate_bleu_n,
    calculate_chrf,
    calculate_meteor,
    calculate_rouge,
    calculate_ter,
    calculate_text_metrics,
    evaluate_transformation,
    tokenize_text,
)
from evaluation.transforms import add_noise, fit_normalizer, to_hinglish


class TextMetricTests(unittest.TestCase):
    def test_hindi_and_hinglish_exact_and_degraded_pairs(self):
        hindi = ["यह फोन बहुत अच्छा है।", "कैमरा साफ़ है"]
        hinglish = ["yah phone bahut achha hai", "camera saaf hai"]
        self.assertEqual(calculate_bleu(hindi, hindi), 100.0)
        self.assertEqual(calculate_bleu(hinglish, hinglish), 100.0)
        for n in range(1, 5):
            self.assertEqual(calculate_bleu_n(hinglish, hinglish, n), 100.0)
        self.assertEqual(calculate_chrf(hinglish, hinglish), 100.0)
        self.assertEqual(calculate_ter(hinglish, hinglish), 0.0)
        self.assertEqual(calculate_meteor(hinglish, hinglish), 100.0)
        self.assertTrue(all(v == 100.0 for v in calculate_rouge(hinglish, hinglish).values()))
        noisy = ["yah phon bahut achha hai", "camera saaf hai"]
        scores = calculate_text_metrics(hinglish, noisy)
        self.assertTrue(all(math.isfinite(v) for v in scores.values()))
        self.assertLess(scores["chrf"], 100.0)
        self.assertGreater(scores["ter"], 0.0)

    def test_empty_unicode_and_alignment(self):
        scores = calculate_text_metrics(["", "नमस्ते"], ["", "नमस्ते"])
        self.assertTrue(all(math.isfinite(v) for v in scores.values()))
        self.assertEqual(scores["bleu"], 100.0)
        self.assertEqual(scores["ter"], 0.0)
        self.assertEqual(tokenize_text("नमस्ते, phone!"), ["नमस्ते", ",", "phone", "!"])
        with self.assertRaises(ValueError):
            calculate_bleu(["one"], [])
        with self.assertRaises(ValueError):
            calculate_bleu([], [])

    def test_row_metadata(self):
        row = evaluate_transformation(
            ["phone achha hai"], ["phone achha hai"], condition="clean",
            noise_level="none", normalization=False, seed="clean", dataset="toy",
        )
        self.assertEqual(row["n_samples"], 1)
        self.assertEqual(row["script"], "Latin")
        self.assertEqual(row["bleu"], 100.0)


class ClassificationAndRobustnessTests(unittest.TestCase):
    def test_classification_metrics_and_alignment(self):
        scores = classification_scores(["a", "a", "b", "b"], ["a", "b", "b", "b"])
        self.assertEqual(scores["accuracy"], 0.75)
        self.assertAlmostEqual(scores["macro_f1"], 0.7333333333)
        self.assertAlmostEqual(scores["weighted_f1"], 0.7333333333)
        with self.assertRaises(ValueError):
            classification_scores(["a"], [])

    def test_noise_normalizer_and_seed_summary(self):
        if importlib.util.find_spec("indic_transliteration"):
            self.assertEqual(to_hinglish("यह फोन अच्छा है"), "yah phon achcha hai")
        self.assertEqual(add_noise("bahut achha phone", 0.6, 7), add_noise("bahut achha phone", 0.6, 7))
        normalize, _ = fit_normalizer(["phone bahut achha", "phone bahut achha"])
        self.assertEqual(normalize("phne bahut achha"), "phone bahut achha")
        frame = pd.DataFrame({"condition": ["high"] * 5, "macro_f1": [0.5, 0.6, 0.7, 0.8, 0.9]})
        summary = summarize_replicates(frame, ["condition"], ["macro_f1"])
        self.assertAlmostEqual(summary.loc[0, "macro_f1_mean"], 0.7)
        self.assertGreater(summary.loc[0, "macro_f1_std"], 0)

    def test_robustness_and_exploratory_correlations(self):
        rows = [{"model": "toy", "condition": "Clean Hinglish", "noise_level": "none",
                 "normalization": False, "seed": "clean", "macro_f1": 0.805}]
        quality = []
        for level_i, level in enumerate(("Low", "Medium", "High")):
            for seed in range(1, 6):
                noisy_f1 = 0.8 - 0.1 * level_i + seed / 1000
                quality_score = 90 - level_i * 15 + seed / 10
                for norm, condition, f1, q in (
                    (False, f"{level} noisy", noisy_f1, quality_score),
                    (True, f"{level} normalized", noisy_f1 + 0.08, quality_score + 8),
                ):
                    rows.append({"model": "toy", "condition": condition, "noise_level": level,
                                 "normalization": norm, "seed": seed, "macro_f1": f1})
                    quality.append({"condition": condition, "noise_level": level,
                                    "normalization": norm, "seed": seed, "chrf": q,
                                    "bleu": q, "rougeL": q})
        cls = pd.DataFrame(rows)
        robustness = robustness_table(cls)
        self.assertAlmostEqual(robustness.loc[0, "F1 drop (clean - high noise)"], 0.202)
        corr, paired = exploratory_correlations(pd.DataFrame(quality), cls)
        self.assertEqual(len(paired), 30)
        self.assertEqual(set(corr["text_metric"]), {"chrf", "bleu", "rougeL"})
        cls_summary = summarize_replicates(cls, ["model", "condition"], ["macro_f1"])
        text_summary = summarize_replicates(
            pd.DataFrame(quality), ["condition", "noise_level", "normalization"],
            ["chrf", "bleu", "rougeL"],
        )
        with tempfile.TemporaryDirectory() as output_dir:
            self.assertIsNotNone(plot_classification_robustness(cls_summary, output_dir))
            self.assertIsNotNone(plot_normalization_recovery(cls, output_dir))
            self.assertTrue(plot_text_quality(text_summary, output_dir))
            self.assertTrue(plot_quality_correlations(paired, output_dir))


if __name__ == "__main__":
    unittest.main()
