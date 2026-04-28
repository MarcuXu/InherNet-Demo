from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from demo_code import build_argparser
from experiment_registry import (
    DATASET_REGISTRY,
    build_method_tag,
    build_pair_model,
    get_pair_spec,
    get_role_name,
    resolve_compressed_source,
    resolve_compressed_train_mode,
    resolve_train_settings,
)
from model_wrappers import DecoupledGatedSVDLinear, GenericHeteroNet, GenericInherNet
from training_utils import count_parameters


class RankOneConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=1, bias=True)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([[[[3.0]], [[4.0]]], [[[6.0]], [[8.0]]]]))
            self.conv.bias.copy_(torch.tensor([0.1, -0.2]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TinyConvLinearNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=1, bias=True)
        self.fc = nn.Linear(18, 3, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.fc(torch.flatten(x, 1))


class RegistryTests(unittest.TestCase):
    def test_get_role_name_supports_both_pair_registry_formats(self) -> None:
        cifar10_pair = get_pair_spec("cifar10", "resnet50_to_resnet18")
        cifar100_pair = get_pair_spec("cifar100", "resnet56_to_resnet20")
        cifar10_org_pair = get_pair_spec("cifar10", "resnet50_to_resnet18_org")

        self.assertEqual(get_role_name(cifar10_pair, "teacher"), "resnet50")
        self.assertEqual(get_role_name(cifar10_pair, "student"), "resnet18")
        self.assertEqual(get_role_name(cifar10_org_pair, "teacher"), "resnet50_org")
        self.assertEqual(get_role_name(cifar10_org_pair, "student"), "resnet18_org")
        self.assertEqual(get_role_name(cifar100_pair, "teacher"), "resnet56")
        self.assertEqual(get_role_name(cifar100_pair, "student"), "resnet20")

    def test_cifar10_org_pair_uses_demo_code_org_stem(self) -> None:
        cifar_stem_model = build_pair_model("cifar10", "resnet50_to_resnet18", "student", 10)
        org_stem_model = build_pair_model("cifar10", "resnet50_to_resnet18_org", "student", 10)

        self.assertEqual(tuple(cifar_stem_model.conv1.kernel_size), (3, 3))
        self.assertEqual(tuple(cifar_stem_model.conv1.stride), (1, 1))
        self.assertIsInstance(cifar_stem_model.maxpool, nn.Identity)
        self.assertEqual(tuple(org_stem_model.conv1.kernel_size), (7, 7))
        self.assertEqual(tuple(org_stem_model.conv1.stride), (2, 2))
        self.assertIsInstance(org_stem_model.maxpool, nn.MaxPool2d)
        self.assertEqual(count_parameters(org_stem_model) - count_parameters(cifar_stem_model), 7680)

    def test_original_compatibility_pair_defaults_match_demo_code_org(self) -> None:
        parser = build_argparser()
        args = parser.parse_args(
            [
                "--dataset",
                "cifar10",
                "--pair",
                "resnet50_to_resnet18_org",
                "--method",
                "inhernet",
            ]
        )
        pair = get_pair_spec(args.dataset, args.pair)
        settings = resolve_train_settings(DATASET_REGISTRY[args.dataset], args, pair)

        self.assertTrue(settings.legacy_eval_sticky)
        self.assertEqual(settings.optimizer_name, "adam")
        self.assertEqual(settings.batch_size, 256)
        self.assertEqual(settings.epochs, 100)
        self.assertEqual(settings.lr, 0.001)
        self.assertEqual(settings.weight_decay, 0.0)
        self.assertEqual(settings.lr_milestones, ())
        self.assertEqual(settings.kd_temperature, 7.0)
        self.assertEqual(settings.kd_loss_weight, 0.7)
        self.assertEqual(settings.ce_loss_weight, 0.3)
        self.assertEqual(resolve_compressed_source(args, pair), "student")
        self.assertEqual(resolve_compressed_train_mode(args, pair), "supervised")
        self.assertEqual(build_method_tag("inhernet", args, pair, settings), "student_source_small_rank_32_heads_3_supervised")
        self.assertIn("student_source_", build_method_tag("hetero", args, pair, settings))

    def test_cifar100_pair_defaults_remain_paper_style_teacher_kd(self) -> None:
        parser = build_argparser()
        args = parser.parse_args(
            [
                "--dataset",
                "cifar100",
                "--pair",
                "resnet56_to_resnet20",
                "--method",
                "inhernet",
            ]
        )
        pair = get_pair_spec(args.dataset, args.pair)
        settings = resolve_train_settings(DATASET_REGISTRY[args.dataset], args, pair)

        self.assertFalse(settings.legacy_eval_sticky)
        self.assertEqual(settings.optimizer_name, "sgd")
        self.assertEqual(settings.batch_size, 64)
        self.assertEqual(settings.epochs, 240)
        self.assertEqual(settings.lr_milestones, (150, 180, 210))
        self.assertEqual(resolve_compressed_source(args, pair), "teacher")
        self.assertEqual(resolve_compressed_train_mode(args, pair), "distillation")
        self.assertEqual(build_method_tag("inhernet", args, pair, settings), "small_rank_8_heads_3")


class WrapperTests(unittest.TestCase):
    def test_inhernet_rank_one_conv_preserves_rank_one_weight(self) -> None:
        torch.manual_seed(7)
        dense_model = RankOneConvNet().eval()
        sample = torch.randn(4, 2, 5, 5)

        inhernet = GenericInherNet(copy.deepcopy(dense_model)).eval()
        inhernet.load_dense_state_dict(dense_model.state_dict())
        inhernet.apply_svd(rank=1, head_num=3, svd_backend="cpu")

        with torch.no_grad():
            expected = dense_model(sample)
            actual = inhernet(sample)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_inhernet_multi_head_gate_breaks_expert_gradient_symmetry(self) -> None:
        torch.manual_seed(13)
        dense_model = RankOneConvNet().train()
        sample = torch.randn(6, 2, 5, 5)

        inhernet = GenericInherNet(copy.deepcopy(dense_model)).train()
        inhernet.load_dense_state_dict(dense_model.state_dict())
        inhernet.apply_svd(rank=1, head_num=3, svd_backend="cpu")

        output = inhernet(sample)
        output.square().mean().backward()
        gated_conv = next(module for module in inhernet.modules() if hasattr(module, "conv_list"))
        expert_grads = [expert.weight.grad.detach().clone() for expert in gated_conv.conv_list]

        self.assertFalse(torch.allclose(expert_grads[0], expert_grads[1]))

    def test_hetero_can_optionally_compress_linear_layers(self) -> None:
        torch.manual_seed(11)
        dense_model = TinyConvLinearNet().eval()
        sample = torch.randn(5, 2, 3, 3)
        calib_inputs = torch.randn(12, 2, 3, 3)
        calib_loader = DataLoader(
            TensorDataset(calib_inputs, torch.zeros(12, dtype=torch.long)),
            batch_size=4,
            shuffle=False,
        )

        hetero = GenericHeteroNet(copy.deepcopy(dense_model)).eval()
        hetero.load_dense_state_dict(dense_model.state_dict())
        rank_map, backend = hetero.apply_hetero_svd(
            calib_loader=calib_loader,
            head_num=3,
            budget_ratio=1.0,
            min_rank=1,
            compress_threshold=1,
            max_calib_batches=3,
            svd_backend="cpu",
            expert_noise_scale=0.01,
            compress_linear=True,
        )

        with torch.no_grad():
            expected = dense_model(sample)
            actual = hetero(sample)

        self.assertEqual(backend, "cpu")
        self.assertIn("fc", rank_map)
        self.assertTrue(any(isinstance(module, DecoupledGatedSVDLinear) for module in hetero.modules()))
        self.assertTrue(torch.allclose(actual, expected, atol=2e-4, rtol=2e-4))


if __name__ == "__main__":
    unittest.main()
