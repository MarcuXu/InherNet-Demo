from __future__ import annotations

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
    max_length: int,
) -> tuple[DataLoader, DataLoader]:
    load_dataset, AutoTokenizer, DataCollatorWithPadding = _load_hf_dependencies()

    if task_name not in GLUE_TEXT_FIELDS:
        available = ", ".join(sorted(GLUE_TEXT_FIELDS))
        raise ValueError(f"Unsupported GLUE task '{task_name}'. Available: {available}")
    if problem_type not in {"classification", "regression"}:
        raise ValueError(f"Unsupported GLUE problem type: {problem_type}")

    cache_dir = Path(root) / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw = load_dataset("glue", task_name, cache_dir=str(cache_dir))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=str(cache_dir))
    text_fields = GLUE_TEXT_FIELDS[task_name]

    def tokenize(batch: Mapping[str, list[str]]) -> dict[str, Any]:
        texts = [batch[field] for field in text_fields]
        return tokenizer(
            *texts,
            truncation=True,
            max_length=max_length,
        )

    remove_columns = [field for field in text_fields if field in raw["train"].column_names]
    if "idx" in raw["train"].column_names:
        remove_columns.append("idx")
    encoded = raw.map(
        tokenize,
        batched=True,
        remove_columns=remove_columns,
    )
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
        GlueTextDataset(encoded["train"], problem_type=problem_type),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        collate_fn=collate_batch,
    )
    validation_loader = DataLoader(
        GlueTextDataset(encoded[eval_split_name], problem_type=problem_type),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
    )
    return train_loader, validation_loader


def build_glue_sst2_dataloaders(**kwargs) -> tuple[DataLoader, DataLoader]:
    return build_glue_dataloaders(task_name="sst2", eval_split_name="validation", problem_type="classification", **kwargs)
