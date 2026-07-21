from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Dataset


class GlueTextDataset(Dataset):
    def __init__(self, encoded_dataset, *, problem_type: str) -> None:
        self.encoded_dataset = encoded_dataset
        self.problem_type = problem_type

    def __len__(self) -> int:
        return len(self.encoded_dataset)

    def __getitem__(self, index: int) -> tuple[dict[str, Any], int | float]:
        item = self.encoded_dataset[index]
        label = float(item["label"]) if self.problem_type == "regression" else int(item["label"])
        inputs = {
            key: value
            for key, value in item.items()
            if key not in {"label"} and value is not None
        }
        return inputs, label


GLUE_TEXT_FIELDS: dict[str, tuple[str, ...]] = {
    "cola": ("sentence",),
    "sst2": ("sentence",),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
}
GLUE_DATASET = "nyu-mll/glue"
GLUE_DATASET_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"


def split_glue_training_data(
    raw_train,
    *,
    problem_type: str,
    validation_fraction: float,
    validation_split_seed: int,
):
    split_kwargs: dict[str, Any] = {
        "test_size": validation_fraction,
        "seed": validation_split_seed,
    }
    if problem_type == "classification":
        split_kwargs["stratify_by_column"] = "label"
    return raw_train.train_test_split(**split_kwargs)


def _load_hf_dependencies():
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer, DataCollatorWithPadding
    except ImportError as exc:
        raise ImportError(
            "GLUE experiments require Hugging Face `datasets` and `transformers`. "
            "Install them with `pip install -r requirements.txt` in the project environment."
        ) from exc
    return load_dataset, AutoTokenizer, DataCollatorWithPadding


def build_glue_dataloaders(
    *,
    task_name: str,
    eval_split_name: str,
    problem_type: str,
    root: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
    tokenizer_name: str,
    tokenizer_revision: str,
    max_length: int,
    search_validation: bool = False,
    validation_fraction: float = 0.1,
    validation_split_seed: int = 2026,
) -> tuple[DataLoader, DataLoader, DataLoader, Mapping[str, Any]]:
    load_dataset, AutoTokenizer, DataCollatorWithPadding = _load_hf_dependencies()

    if task_name not in GLUE_TEXT_FIELDS:
        available = ", ".join(sorted(GLUE_TEXT_FIELDS))
        raise ValueError(f"Unsupported GLUE task '{task_name}'. Available: {available}")
    if problem_type not in {"classification", "regression"}:
        raise ValueError(f"Unsupported GLUE problem type: {problem_type}")

    cache_dir = Path(root) / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(
        GLUE_DATASET,
        task_name,
        revision=GLUE_DATASET_REVISION,
        cache_dir=str(cache_dir),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=tokenizer_revision,
        cache_dir=str(cache_dir),
    )
    text_fields = GLUE_TEXT_FIELDS[task_name]

    def tokenize(batch: Mapping[str, list[str]]) -> dict[str, Any]:
        texts = [batch[field] for field in text_fields]
        return tokenizer(
            *texts,
            truncation=True,
            max_length=max_length,
        )

    def encode_split(split):
        remove_columns = [field for field in text_fields if field in split.column_names]
        if "idx" in split.column_names:
            remove_columns.append("idx")
        return split.map(
            tokenize,
            batched=True,
            remove_columns=remove_columns,
        )

    raw_full_train = raw["train"]
    raw_official_evaluation = raw[eval_split_name]
    if search_validation:
        selection_split = split_glue_training_data(
            raw_full_train,
            problem_type=problem_type,
            validation_fraction=validation_fraction,
            validation_split_seed=validation_split_seed,
        )
        raw_train = selection_split["train"]
        raw_evaluation = selection_split["test"]
        evaluation_split_label = "train_holdout"
    else:
        raw_train = raw_full_train
        raw_evaluation = raw_official_evaluation
        evaluation_split_label = eval_split_name
    encoded_train = encode_split(raw_train)
    encoded_evaluation = encode_split(raw_evaluation)
    calibration_count = min(16 * batch_size, len(raw_train))
    calibration_kwargs: dict[str, Any] = {
        "test_size": calibration_count,
        "seed": validation_split_seed + 1,
    }
    if problem_type == "classification":
        calibration_kwargs["stratify_by_column"] = "label"
    raw_calibration = raw_train.train_test_split(**calibration_kwargs)["test"]
    encoded_calibration = encode_split(raw_calibration)
    provenance = {
        "profile": "huggingface_glue",
        "dataset": GLUE_DATASET,
        "dataset_revision": GLUE_DATASET_REVISION,
        "task": task_name,
        "datasets_version": version("datasets"),
        "train_split": "train_subset" if search_validation else "train",
        "evaluation_split": evaluation_split_label,
        "train_examples": len(raw_train),
        "evaluation_examples": len(raw_evaluation),
        "official_evaluation_split": eval_split_name,
        "official_evaluation_examples": len(raw_official_evaluation),
        "selection_validation_fraction": validation_fraction if search_validation else None,
        "selection_split_seed": validation_split_seed if search_validation else None,
        "calibration_profile": "fixed_seeded_stratified" if problem_type == "classification" else "fixed_seeded",
        "calibration_examples": len(raw_calibration),
        "calibration_seed": validation_split_seed + 1,
        "tokenizer": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "max_length": max_length,
    }
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    def collate_batch(batch: list[tuple[dict[str, Any], int | float]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        features = []
        for inputs, label in batch:
            feature = dict(inputs)
            feature["labels"] = label
            features.append(feature)
        padded = collator(features)
        labels = padded.pop("labels")
        labels = labels.float() if problem_type == "regression" else labels.long()
        return dict(padded), labels

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        GlueTextDataset(encoded_train, problem_type=problem_type),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        collate_fn=collate_batch,
    )
    validation_loader = DataLoader(
        GlueTextDataset(encoded_evaluation, problem_type=problem_type),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
    )
    calibration_loader = DataLoader(
        GlueTextDataset(encoded_calibration, problem_type=problem_type),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
    )
    return train_loader, validation_loader, calibration_loader, provenance
