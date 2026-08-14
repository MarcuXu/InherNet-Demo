from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

import torch.nn as nn


SMALL_BERT_TEACHER = "google/bert_uncased_L-4_H-256_A-4"
SMALL_BERT_STUDENT = "google/bert_uncased_L-2_H-128_A-2"
SMALL_BERT_TOKENIZER = "bert-base-uncased"
SMALL_BERT_REVISIONS = {
    SMALL_BERT_TEACHER: "387825ce42dbb39b87911cdf8e383ee3b25184f8",
    SMALL_BERT_STUDENT: "30b0a37ccaaa32f332884b96992754e246e48c5f",
}
SMALL_BERT_TOKENIZER_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"

SMALL_BERT_CONFIGS: Dict[str, Mapping[str, int]] = {
    SMALL_BERT_TEACHER: {
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 1024,
    },
    SMALL_BERT_STUDENT: {
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 512,
    },
}


PAIR_REGISTRY: Dict[str, Mapping[str, object]] = {
    "bert4_to_bert2": {
        "teacher_name": SMALL_BERT_TEACHER,
        "student_name": SMALL_BERT_STUDENT,
        "tokenizer_name": SMALL_BERT_TOKENIZER,
        "teacher_revision": SMALL_BERT_REVISIONS[SMALL_BERT_TEACHER],
        "student_revision": SMALL_BERT_REVISIONS[SMALL_BERT_STUDENT],
        "tokenizer_revision": SMALL_BERT_TOKENIZER_REVISION,
        "rank_presets": {"small": 32, "large": 64},
        "default_head_num": 2,
        "inhernet_protocol_source": "compact_bert_adaptation",
        "inhernet_rank_source": "repository_defined",
        "compressed_train_mode": "distillation",
        "compress_linear": True,
        "model_profile": "glue_small_bert_teacher_inheritance",
    }
}


def build_model(
    model_name: str,
    num_classes: int,
    *,
    pretrained: bool = True,
    cache_dir: str | Path | None = None,
) -> nn.Module:
    try:
        from transformers import BertConfig, BertForPreTraining, BertForSequenceClassification
    except ImportError as exc:
        raise ImportError(
            "GLUE experiments require the optional Hugging Face dependencies. "
            "Install them with `pip install -r requirements.txt` in the project environment."
        ) from exc

    if pretrained:
        pretrained_kwargs = {
            "revision": SMALL_BERT_REVISIONS[model_name],
        }
        if cache_dir is not None:
            pretrained_kwargs["cache_dir"] = str(cache_dir)
        pretrained_model = BertForPreTraining.from_pretrained(
            model_name,
            **pretrained_kwargs,
        )
        config = pretrained_model.config
        config.num_labels = num_classes
        model = BertForSequenceClassification(config)
        model.bert.load_state_dict(pretrained_model.bert.state_dict(), strict=True)
        return model
    if model_name not in SMALL_BERT_CONFIGS:
        raise KeyError(f"No offline compact-BERT config registered for {model_name}.")
    config = BertConfig(
        vocab_size=30522,
        max_position_embeddings=512,
        type_vocab_size=2,
        layer_norm_eps=1e-12,
        num_labels=num_classes,
        **SMALL_BERT_CONFIGS[model_name],
    )
    return BertForSequenceClassification(config)
