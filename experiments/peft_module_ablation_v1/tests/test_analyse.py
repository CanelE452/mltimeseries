import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.peft_module_ablation_v1 import analyse


class ModuleAblationAnalysisTests(unittest.TestCase):
    def test_effect_ratio_and_classification_rules(self):
        weights = np.ones((5, 4), dtype=np.float64) / 4
        denominator = np.ones((3, 4), dtype=np.float64)

        harm = analyse.effect_ratio(np.full((3, 4), 0.02), denominator, weights)
        self.assertEqual(analyse.classify_effect(harm), "repeated_practical_harm")

        improvement = analyse.effect_ratio(np.full((3, 4), -0.02), denominator, weights)
        self.assertEqual(analyse.classify_effect(improvement), "practical_improvement")

        equivalent = analyse.effect_ratio(np.full((3, 4), 0.002), denominator, weights)
        self.assertEqual(analyse.classify_effect(equivalent), "conditional_practical_equivalence")

        mixed = analyse.effect_ratio(
            np.array([[0.02, 0.02, 0.02, 0.02], [-0.02, -0.02, -0.02, -0.02], [0.0, 0.0, 0.0, 0.0]]),
            denominator,
            weights,
        )
        self.assertEqual(analyse.classify_effect(mixed), "conditional_practical_equivalence")

    def test_episode_loss_matches_raw_mean_two_pinball_contract(self):
        quantiles = np.array([0.1, 0.5, 0.9])
        target = np.array([[1.0, 2.0]])
        prediction = np.array([[[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]])
        loss = analyse.episode_loss(prediction, target, quantiles)
        expected = np.array([((0.2 + 0.2) + 0.0 + (0.2 + 0.2)) / 6])
        self.assertTrue(np.allclose(loss, expected))

    def test_score_summary_uses_evaluation_only_and_reports_crossing(self):
        quantiles = np.array([0.1, 0.5, 0.9])
        target = np.array([[1.0, 2.0], [3.0, 4.0]])
        prediction = np.stack(
            (
                target - 1.0,
                target,
                target + 1.0,
            ),
            axis=1,
        )
        oracle_mean = target + 0.5
        design = np.column_stack((np.ones(target.size), np.ones((target.size, 3))))
        summary = analyse.summarize_scores(prediction, target, quantiles, oracle_mean, design)
        self.assertEqual(summary["median_mse"], 0.0)
        self.assertEqual(summary["coverage80"], 1.0)
        self.assertEqual(summary["width80"], 2.0)
        self.assertEqual(summary["crossing"], 0.0)
        self.assertIn("self=", summary["exploratoryself/U/Vcoeffs"])

    def test_expected_new_trial_keys_are_the_fixed_twelve_grid(self):
        expected = {
            (corpus, method, lr)
            for corpus in range(3)
            for method in ("OUT_ONLY", "ATTN_ONLY")
            for lr in (3e-5, 1e-4)
        }
        self.assertEqual(analyse.expected_new_trial_keys(), expected)

    def test_resource_summary_keeps_new_and_reused_samples_separate(self):
        sample = {
            "timestamp": "2026-09-08T00:00:00Z",
            "available_ram_gib": 10.0,
            "available_commit_gib": 12.0,
            "child_tree_rss_gib": 1.0,
            "git_process_count": 0,
            "gpus": [{"memory_used_mib": 1000, "temperature_c": 40}],
        }
        new = analyse.summarize_resources([sample], "new guards only", "new")
        reused = analyse.summarize_resources([], "reused references only", "reused")
        self.assertEqual(new["origin"], "new")
        self.assertEqual(new["sample_count"], 1)
        self.assertEqual(new["max_gpu_memory_used_mib"], 1000)
        self.assertEqual(reused["origin"], "reused")
        self.assertEqual(reused["sample_count"], 0)
        self.assertIsNone(reused["max_gpu_memory_used_mib"])

    def test_resource_samples_ignore_guard_finish_metadata_rows(self):
        sample = {
            "timestamp": "2026-09-08T00:00:00Z",
            "available_ram_gib": 10.0,
            "available_commit_gib": 12.0,
            "child_tree_rss_gib": 1.0,
            "git_process_count": 0,
            "gpus": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            trial = Path(directory) / "trial"
            log = trial / "guard" / "resource_log.jsonl"
            log.parent.mkdir(parents=True)
            log.write_text(json.dumps(sample) + "\n" + json.dumps({"event": "finish"}) + "\n")
            self.assertEqual(analyse.resource_samples([trial]), [sample])

    def test_checkpoint_validation_uses_each_trial_own_saved_file(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            first.mkdir()
            second.mkdir()
            (first / "best_adaptation.pt").write_bytes(b"first learned weights")
            (second / "best_adaptation.pt").write_bytes(b"second learned weights")
            first_hash = hashlib.sha256((first / "best_adaptation.pt").read_bytes()).hexdigest()
            second_hash = hashlib.sha256((second / "best_adaptation.pt").read_bytes()).hexdigest()
            self.assertNotEqual(first_hash, second_hash)
            analyse.validate_adaptation_checkpoint_file(first, {"checkpoint_sha256": first_hash})
            analyse.validate_adaptation_checkpoint_file(second, {"checkpoint_sha256": second_hash})

    def test_prediction_archive_accepts_string_episode_ids_without_casting(self):
        quantiles = np.array([0.1, 0.5, 0.9], dtype=np.float64)
        target_val = np.array([[1.0, 2.0]], dtype=np.float64)
        target_eval = np.array([[3.0, 4.0]], dtype=np.float64)
        prediction_val = np.repeat(target_val[:, None, :], len(quantiles), axis=1)
        prediction_eval = np.repeat(target_eval[:, None, :], len(quantiles), axis=1)
        data = {
            "target_val": target_val,
            "target_eval": target_eval,
            "episode_ids_val": np.array(["val_c00_e00000"]),
            "episode_ids_eval": np.array(["eval_c00_e00000"]),
            "quantiles": quantiles,
        }
        with tempfile.TemporaryDirectory() as directory:
            trial = Path(directory) / "trial"
            trial.mkdir()
            np.savez_compressed(
                trial / "predictions.npz",
                val_predictions=prediction_val,
                eval_predictions=prediction_eval,
                val_target=target_val,
                eval_target=target_eval,
                val_episode_losses=analyse.episode_loss(prediction_val, target_val, quantiles),
                eval_episode_losses=analyse.episode_loss(prediction_eval, target_eval, quantiles),
                val_episode_ids=data["episode_ids_val"],
                eval_episode_ids=data["episode_ids_eval"],
                quantiles=quantiles,
            )
            archive = analyse.validate_prediction_archive(trial, data)
            self.assertEqual(archive["eval_predictions"].shape, (1, 3, 2))

    def test_smoke_reproduction_requires_exact_three_method_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = root / "runs" / analyse.STUDY
            study.mkdir(parents=True)
            trials = []
            for method in ("OUT_ONLY", "ATTN_ONLY", "OFF_LORA"):
                path = study / "smoke" / method
                status = path / "guard" / "status.json"
                status.parent.mkdir(parents=True)
                status.write_text(json.dumps({"completed": True, "returncode": 0, "reasons": []}))
                trials.append({"method": method, "path": str(path.relative_to(root)).replace("\\", "/")})
            (study / "smoke_completed.json").write_text(json.dumps({
                "completed": True,
                "both_reproduces_parent_s0": True,
                "trials": trials,
            }))
            smoke = analyse.validate_smoke_reproduction(root, study)
            self.assertTrue(smoke["both_reproduces_parent_s0"])


if __name__ == "__main__":
    unittest.main()
