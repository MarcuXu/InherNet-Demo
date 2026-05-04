from __future__ import annotations

from typing import Dict, Mapping

import torch.nn as nn


SMALL_BERT_TEACHER = "google/bert_uncased_L-4_H-256_A-4"
SMALL_BERT_STUDENT = "google/bert_uncased_L-2_H-128_A-2"
SMALL_BERT_TOKENIZER = "bert-base-uncased"


PAIR_REGISTRY: Dict[str, Mapping[str, object]] = {
    "bert4_to_bert2": {
        "teacher_name": SMALL_BERT_TEACHER,
        "student_name": SMALL_BERT_STUDENT,
        "tokenizer_name": SMALL_BERT_TOKENIZER,
        "rank_presets": {"small": 32, "large": 64},
        "default_head_num": 2,
        "compressed_source": "teacher",
        "compressed_train_mode": "distillation",
        "hetero_compress_linear_default": True,
        "model_profile": "glue_small_bert_teacher_inheritance",
    }
}


def build_model(model_name: str, num_classes: int) -> nn.Module:
    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError as exc:
        raise ImportError(
            "GLUE experiments require the optional Hugging Face dependencies. "
            "Install them with `pip install -r requirements.txt` in the project environment."
        ) from exc

    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
