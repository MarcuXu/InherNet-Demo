from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from demo_code import build_argparser, pretrained_model_cache_dir
from experiment_registry import DATASET_REGISTRY, build_pair_model
from glue_models import SMALL_BERT_TEACHER, build_model


class _FakeConfig:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _FakeEncoder:
    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict, strict: bool = True) -> None:
        self.loaded_state_dict = state_dict
        self.strict = strict


class _FakePretrainingModel:
    def __init__(self) -> None:
        self.config = _FakeConfig()
        self.bert = _FakeEncoder()


class _FakePretraining:
    calls: list[tuple[str, dict[str, object]]] = []

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        cls.calls.append((model_name, kwargs))
        return _FakePretrainingModel()


class _FakeSequenceClassifier:
    def __init__(self, config) -> None:
        self.config = config
        self.bert = _FakeEncoder()


def _fake_transformers_module() -> types.ModuleType:
    module = types.ModuleType("transformers")
    module.BertConfig = _FakeConfig
    module.BertForPreTraining = _FakePretraining
    module.BertForSequenceClassification = _FakeSequenceClassifier
    return module


class GlueCacheTests(unittest.TestCase):
    def test_pretrained_glue_model_uses_requested_cache_and_offline_build_does_not(self) -> None:
        cache_dir = Path("/tmp/inheract_glue_cache")
        _FakePretraining.calls.clear()
        with patch.dict(sys.modules, {"transformers": _fake_transformers_module()}):
            pretrained = build_model(SMALL_BERT_TEACHER, 2, cache_dir=cache_dir)
            offline = build_model(
                SMALL_BERT_TEACHER,
                2,
                pretrained=False,
                cache_dir=cache_dir,
            )

        self.assertEqual(pretrained.config.num_labels, 2)
        self.assertEqual(offline.config.num_labels, 2)
        self.assertEqual(
            _FakePretraining.calls,
            [
                (
                    SMALL_BERT_TEACHER,
                    {
                        "revision": "387825ce42dbb39b87911cdf8e383ee3b25184f8",
                        "cache_dir": str(cache_dir),
                    },
                )
            ],
        )

    def test_registry_and_runner_derive_the_glue_cache_from_data_root(self) -> None:
        cache_dir = Path("/tmp/inheract_data") / "huggingface"
        with patch("experiment_registry.build_glue_model", return_value=object()) as build_glue_model:
            build_pair_model(
                "glue_sst2",
                "bert4_to_bert2",
                "teacher",
                2,
                cache_dir=cache_dir,
            )
        build_glue_model.assert_called_once_with(
            SMALL_BERT_TEACHER,
            2,
            pretrained=True,
            cache_dir=cache_dir,
        )

        args = build_argparser().parse_args(
            [
                "--dataset",
                "glue_sst2",
                "--pair",
                "bert4_to_bert2",
                "--method",
                "teacher",
                "--data-root",
                "/tmp/inheract_data",
            ]
        )
        self.assertEqual(
            pretrained_model_cache_dir(args, DATASET_REGISTRY["glue_sst2"]),
            cache_dir,
        )


if __name__ == "__main__":
    unittest.main()
