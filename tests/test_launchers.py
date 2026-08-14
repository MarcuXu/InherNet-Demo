from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from experiment_registry import DATASET_REGISTRY


class LauncherContractTests(unittest.TestCase):
    def _staged_project(self, directory: str) -> Path:
        root = Path(directory) / "project"
        root.mkdir()
        for source in PROJECT_DIR.glob("*.py"):
            shutil.copy2(source, root / source.name)
        shutil.copytree(PROJECT_DIR / "configs", root / "configs")
        shutil.copytree(
            PROJECT_DIR / "scripts",
            root / "scripts",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        return root

    def _dry_run(self, root: Path, script: str, *args: str, **environment: str) -> str:
        env = os.environ.copy()
        env.update(
            {
                "PYTHON_BIN": sys.executable,
                "DRY_RUN": "1",
                "BACKGROUND": "0",
                **environment,
            }
        )
        completed = subprocess.run(
            [str(root / "scripts" / script), *args],
            cwd=root,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return completed.stdout

    def test_formal_dry_run_is_read_only_and_uses_paper_inhernet_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._staged_project(directory)
            output = self._dry_run(
                root,
                "formal.sh",
                "cifar100",
                "resnet56_to_resnet20",
                SEEDS="7",
                FORMAL_RUN_ID="unit_formal",
            )
            self.assertFalse((root / "logs").exists())
            self.assertFalse((root / "checkpoints").exists())
            small = next(
                line for line in output.splitlines()
                if "--method inhernet --size small" in line
            )
            large = next(line for line in output.splitlines() if "--method inhernet --size large" in line)
            dkd = next(line for line in output.splitlines() if "--method student_dkd" in line)
            standardized_kd = next(
                line
                for line in output.splitlines()
                if "--method student_kd_logit_standardized" in line
            )
            ctkd = next(line for line in output.splitlines() if "--method student_ctkd" in line)
            for method in ("student_catkd", "student_simkd", "student_reviewkd", "student_crd"):
                self.assertIn(f"--method {method}", output)
            self.assertIn("--compressed-train-mode distillation", small)
            self.assertIn("--compressed-train-mode supervised", large)
            self.assertNotIn("formal_inhernet_small_supervised", output)
            self.assertNotIn("formal_svd_inheritance_reference", output)
            self.assertNotIn("--head-num 1", output)
            self.assertIn("checkpoints/formal/unit_formal/cifar100", output)
            self.assertIn("teacher_seed_7.pt", dkd)
            self.assertIn("teacher_seed_7.pt", standardized_kd)
            self.assertIn("teacher_seed_7.pt", ctkd)

    def test_distillation_controls_match_inheract_objective_and_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._staged_project(directory)
            stsb = self._dry_run(
                root,
                "formal.sh",
                "glue_stsb",
                "bert4_to_bert2",
                SEEDS="7",
                FORMAL_RUN_ID="unit_formal",
            )
            pets = self._dry_run(
                root,
                "formal.sh",
                "oxford_pets",
                "resnet34_to_resnet18",
                SEEDS="7",
                FORMAL_RUN_ID="unit_formal",
            )
            stsb_control = next(
                line
                for line in stsb.splitlines()
                if "formal_inhernet_large_matched_inheract_objective_optimizer" in line
            )
            pets_control = next(
                line
                for line in pets.splitlines()
                if "formal_inhernet_large_matched_inheract_objective_optimizer" in line
            )
            self.assertIn("--method inhernet --size large", stsb_control)
            self.assertIn("--compressed-train-mode distillation", stsb_control)
            self.assertIn("--kd-temperature 2.0", stsb_control)
            self.assertIn("--kd-fraction 0.25", stsb_control)
            self.assertIn("--lr-scale 2.0", stsb_control)
            self.assertIn("--kd-fraction 0.50", pets_control)
            self.assertIn("--lr-scale 1.0", pets_control)

    def test_ablation_dry_run_is_read_only_and_uses_paired_formal_teachers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._staged_project(directory)
            output = self._dry_run(
                root,
                "ablation.sh",
                "cifar100",
                "resnet32_to_resnet8",
                ABLATION_SEEDS="7,17,27",
                FORMAL_RUN_ID="unit_formal",
            )
            self.assertFalse((root / "logs").exists())
            self.assertFalse((root / "checkpoints").exists())
            for seed in (7, 17, 27):
                checkpoint = (
                    "checkpoints/formal/unit_formal/cifar100/"
                    f"resnet32_to_resnet8/teacher_seed_{seed}.pt"
                )
                self.assertIn(checkpoint, output)
            self.assertEqual(output.count("=== InherAct full ==="), 3)
            self.assertEqual(output.count("=== Direct SVD inheritance control (one head) ==="), 3)
            direct_svd = next(
                line for line in output.splitlines()
                if "ablation_direct_svd" in line
            )
            self.assertIn("--method inhernet", direct_svd)
            self.assertIn("--size large", direct_svd)
            self.assertIn("--head-num 1", direct_svd)
            self.assertIn("--no-final-test", output)

    def test_glue_formal_and_ablation_select_on_training_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._staged_project(directory)
            formal = self._dry_run(
                root,
                "formal.sh",
                "glue_sst2",
                "bert4_to_bert2",
                SEEDS="7",
            )
            pets = self._dry_run(
                root,
                "formal.sh",
                "oxford_pets",
                "resnet34_to_resnet18",
                SEEDS="7",
            )
            output = self._dry_run(
                root,
                "ablation.sh",
                "glue_sst2",
                "bert4_to_bert2",
                ABLATION_SEEDS="7",
                FORMAL_RUN_ID="unit_formal",
            )
            self.assertIn("--search-validation", formal)
            self.assertNotIn("--no-final-test", formal)
            self.assertNotIn("formal_svd_inheritance_reference", formal)
            self.assertNotIn("formal_svd_inheritance_reference", pets)
            self.assertIn("--search-validation", output)
            self.assertIn("--no-final-test", output)
            self.assertIn(
                "checkpoints/formal/unit_formal/glue_sst2/bert4_to_bert2/teacher_seed_7.pt",
                output,
            )

    def test_formal_all_covers_every_registered_target_sequentially(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._staged_project(directory)
            output = self._dry_run(
                root,
                "formal_all.sh",
                "all",
                "--download",
                "--num-workers",
                "4",
                SEEDS="7",
                FORMAL_RUN_ID="unit_formal_all",
            )
            self.assertFalse((root / "logs").exists())
            self.assertFalse((root / "checkpoints").exists())
            target_prefix = "######## Formal target: "
            observed_targets = {
                line.removeprefix(target_prefix).removesuffix(" ########")
                for line in output.splitlines()
                if line.startswith(target_prefix)
            }
            expected_targets = {
                f"{dataset} / {pair}"
                for dataset, dataset_spec in DATASET_REGISTRY.items()
                for pair in dataset_spec.pair_registry
            }
            self.assertSetEqual(observed_targets, expected_targets)
            self.assertEqual(len(observed_targets), 18)
            self.assertEqual(output.count("=== teacher seed=7 ==="), 18)
            self.assertEqual(output.count("--method student_dkd"), 7)
            self.assertEqual(output.count("--method student_kd_logit_standardized"), 7)
            self.assertEqual(output.count("--method student_ctkd"), 7)
            self.assertEqual(output.count("--method student_catkd"), 6)
            self.assertEqual(output.count("--method student_simkd"), 7)
            self.assertEqual(output.count("--method student_reviewkd"), 5)
            self.assertEqual(output.count("--method student_crd"), 7)
            run_commands = [
                line
                for line in output.splitlines()
                if line.startswith(str(root / "scripts" / "run.sh"))
            ]
            self.assertEqual(len(run_commands), 158)
            self.assertNotIn("Direct SVD inheritance reference", output)
            self.assertNotIn("InherNet small supervised control", output)
            self.assertEqual(output.count("=== InherAct recipe=screen_selected seed=7 ==="), 18)
            glue_commands = [
                line
                for line in output.splitlines()
                if "--dataset glue_" in line and line.startswith(str(root / "scripts" / "run.sh"))
            ]
            self.assertTrue(glue_commands)
            self.assertTrue(all("--search-validation" in line for line in glue_commands))

    def test_formal_resume_requires_explicit_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._staged_project(directory)
            env = os.environ.copy()
            env.update(
                {
                    "PYTHON_BIN": sys.executable,
                    "DRY_RUN": "1",
                    "BACKGROUND": "0",
                    "RESUME": "1",
                }
            )
            completed = subprocess.run(
                [str(root / "scripts" / "formal.sh"), "cifar100", "resnet56_to_resnet20"],
                cwd=root,
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("FORMAL_RUN_ID", completed.stderr)

    def test_formal_rejects_teacher_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._staged_project(directory)
            env = os.environ.copy()
            env.update(
                {
                    "PYTHON_BIN": sys.executable,
                    "DRY_RUN": "1",
                    "BACKGROUND": "0",
                    "OVERWRITE_TEACHER": "1",
                }
            )
            completed = subprocess.run(
                [str(root / "scripts" / "formal.sh"), "cifar100", "resnet56_to_resnet20"],
                cwd=root,
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("never overwrite teachers", completed.stderr)


if __name__ == "__main__":
    unittest.main()
