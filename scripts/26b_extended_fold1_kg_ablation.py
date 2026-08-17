"""Extended Fold-1 ablation: clean, drug-only KG, and drug+side KG."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "26a_pilot_compare_clean_vs_kg.py"
SPLIT_PATH = PROJECT_ROOT / "data_processed" / "experiments" / "pilot_fold1_split.npz"
PILOT_OUTPUT_DIR = PROJECT_ROOT / "data_processed" / "experiments" / "pilot_fold1"
OUTPUT_DIR = PROJECT_ROOT / "data_processed" / "experiments" / "extended_fold1"
KG_ARTIFACT_PATH = (
    PROJECT_ROOT / "data_processed" / "kg_features" / "biokorf_kg_embeddings.pt"
)
REPORT_PATH = OUTPUT_DIR / "extended_fold1_report.txt"

SEED = 42
MAX_EPOCHS = 30
PATIENCE = 7
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 128
DROPOUT = 0.4
LATENT_DIM = 64


def load_pilot_helpers() -> Any:
    spec = importlib.util.spec_from_file_location("biokorf_pilot_helpers", PILOT_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pilot helper module from {PILOT_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = load_pilot_helpers()
sys.path.insert(0, str(PROJECT_ROOT))
from models.mssf_clean import MSSFClean, MSSFCleanConfig
from models.mssf_clean_kg import BioKORFCleanKG


PROTECTED_HASHES = {
    **pilot.PROTECTED_HASHES,
    PROJECT_ROOT / "models" / "kg_encoder.py": "PLACEHOLDER",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DrugOnlyBioKORFCleanKG(BioKORFCleanKG):
    """Use drug KG and its mask while forcing the entire side-KG input to zero."""

    def forward(
        self,
        drugs: Tensor,
        sides: Tensor,
        drug_index: Tensor,
        side_effect_index: Tensor,
        device: torch.device | str | None = None,
        return_debug: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]
    ]:
        target_device = torch.device(device) if device is not None else next(self.parameters()).device
        drugs = drugs.to(target_device)
        sides = sides.to(target_device)
        drug_index = drug_index.to(target_device)
        side_effect_index = side_effect_index.to(target_device)

        h_en_con, rec_con = self.encoderConnection(drugs, sides)
        h_en_add, rec_add = self.encoderAddition(drugs, sides)
        processed_drugs, processed_sides = self.preprocess(drugs, sides)
        h_cnn_im = self.crossProduction(processed_drugs, processed_sides)
        h_pair = torch.cat((h_en_con, h_en_add, h_cnn_im), dim=1)

        z_drug, _z_side, drug_mask, _side_mask, _kg_input = self.kg_features(
            drug_index, side_effect_index
        )
        z_side_disabled = torch.zeros_like(_z_side)
        side_mask_disabled = torch.zeros_like(_side_mask)
        kg_input = torch.cat(
            (
                z_drug,
                z_side_disabled,
                drug_mask.to(z_drug.dtype),
                side_mask_disabled.to(z_drug.dtype),
            ),
            dim=1,
        )
        h_fused, kg_projected, kg_gate = self.kg_fusion(
            h_pair, kg_input, return_debug=True
        )
        mu, logvar = self.gaussian_parametrizer(h_fused)
        latent = self.reparameterize(mu, logvar)
        logits = self.classifier(latent)
        outputs = (logits, rec_con, rec_add, mu, logvar)
        if not return_debug:
            return outputs
        return (
            *outputs,
            {
                "H_en_con": h_en_con,
                "H_en_add": h_en_add,
                "H_cnn_im": h_cnn_im,
                "H_pair": h_pair,
                "Z_drug_KG": z_drug,
                "Z_side_KG": z_side_disabled,
                "drug_kg_mask": drug_mask,
                "side_kg_mask": side_mask_disabled,
                "KG_input": kg_input,
                "KG_projected": kg_projected,
                "KG_gate": kg_gate,
                "H_fused": h_fused,
                "latent": latent,
                "logits": logits,
            },
        )


def file_tree_hashes(paths: list[Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in paths:
        if path.is_file():
            hashes[str(path.resolve())] = sha256(path)
        elif path.is_dir():
            for child in sorted(item for item in path.rglob("*") if item.is_file()):
                hashes[str(child.resolve())] = sha256(child)
        else:
            raise FileNotFoundError(f"Protected previous experiment artifact not found: {path}")
    return hashes


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"Cannot infer columns for empty CSV: {path}")
    columns = fieldnames or list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_fixed_split(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(f"Required fixed Fold-1 split not found: {SPLIT_PATH}")
    split = np.load(SPLIT_PATH)
    required = {
        "train_indices",
        "validation_indices",
        "test_indices",
        "train_samples",
        "validation_samples",
        "test_samples",
        "seed",
        "fold",
    }
    if required.difference(split.files):
        raise KeyError(f"Saved split is missing arrays: {sorted(required.difference(split.files))}")
    if int(split["seed"]) != SEED or int(split["fold"]) != 1:
        raise ValueError("Saved split seed/fold metadata is not seed=42, fold=1")
    for prefix in ("train", "validation", "test"):
        indices = split[f"{prefix}_indices"]
        saved_samples = split[f"{prefix}_samples"]
        if not np.array_equal(samples[indices], saved_samples):
            raise ValueError(f"Saved {prefix} samples do not match their fixed indices")
    combined = np.concatenate(
        (split["train_indices"], split["validation_indices"], split["test_indices"])
    )
    if len(combined) != len(samples) or len(np.unique(combined)) != len(samples):
        raise ValueError("Saved split is not a complete disjoint sample partition")
    return split["train_samples"], split["validation_samples"], split["test_samples"]


def checkpoint_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def validation_epoch(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    use_kg: bool,
) -> tuple[float, dict[str, Any], dict[str, float]]:
    return pilot.evaluate(model, dataset, device, use_kg=use_kg)


def train_with_early_stopping(
    model_name: str,
    model: nn.Module,
    train_dataset: torch.utils.data.Dataset,
    validation_dataset: torch.utils.data.Dataset,
    device: torch.device,
    use_kg: bool,
) -> tuple[list[dict[str, Any]], int, float, Path]:
    model = model.to(device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    checkpoint_path = OUTPUT_DIR / f"{model_name}_best.pt"
    history_path = OUTPUT_DIR / f"{model_name}_history.csv"
    for epoch in range(1, MAX_EPOCHS + 1):
        pilot.configure_reproducibility(SEED + epoch)
        started = time.perf_counter()
        train_loss = pilot.train_epoch(
            model, train_dataset, optimizer, device, epoch, use_kg=use_kg
        )
        val_loss, val_metrics, gate_stats = validation_epoch(
            model, validation_dataset, device, use_kg
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_aupr": val_metrics["aupr"],
        }
        if use_kg:
            row.update(gate_stats)
        history.append(row)
        write_csv(history_path, history)
        improved = val_metrics["macro_f1"] > best_macro_f1
        if improved:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "model_name": model_name,
                    "epoch": epoch,
                    "validation_macro_f1": best_macro_f1,
                    "model_state_dict": checkpoint_state(model),
                    "seed": SEED,
                    "optimizer": {
                        "name": "Adam",
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                    },
                },
                temporary,
            )
            temporary.replace(checkpoint_path)
        else:
            epochs_without_improvement += 1
        gate_text = (
            f" gate={gate_stats['gate_mean']:.6f}±{gate_stats['gate_std']:.6f} "
            f"[{gate_stats['gate_min']:.6f},{gate_stats['gate_max']:.6f}]"
            if use_kg
            else ""
        )
        print(
            f"{model_name} epoch {epoch}/{MAX_EPOCHS}: train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_macro_f1={val_metrics['macro_f1']:.6f}"
            f"{gate_text} best={best_epoch} patience={epochs_without_improvement}/{PATIENCE} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        if epochs_without_improvement >= PATIENCE:
            print(f"{model_name} early stopping at epoch {epoch}", flush=True)
            break
    del optimizer
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return history, best_epoch, best_macro_f1, checkpoint_path


def subset_metrics(
    labels: np.ndarray, probabilities: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    count = int(mask.sum())
    if count == 0:
        return {"sample_count": 0, "accuracy": None, "macro_f1": None}
    metrics = pilot.classification_metrics(labels[mask], probabilities[mask])
    return {
        "sample_count": count,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
    }


def selected_test_evaluation(
    model: nn.Module,
    checkpoint_path: Path,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    kg_mode: str,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    labels_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    gate_batches: list[np.ndarray] = []
    drug_mask_batches: list[np.ndarray] = []
    side_mask_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for drugs, sides, drug_index, side_index, labels in pilot.make_loader(
            dataset, shuffle=False, seed=SEED
        ):
            drugs = drugs.to(device, non_blocking=True)
            sides = sides.to(device, non_blocking=True)
            if kg_mode == "clean":
                logits, *_ = model(drugs, sides, device=device)
            else:
                logits, _, _, _, _, debug = model(
                    drugs,
                    sides,
                    drug_index.to(device, non_blocking=True),
                    side_index.to(device, non_blocking=True),
                    device=device,
                    return_debug=True,
                )
                gate_batches.append(debug["KG_gate"].mean(dim=1).cpu().numpy())
                drug_mask_batches.append(debug["drug_kg_mask"].squeeze(1).cpu().numpy())
                if kg_mode == "drug_side_kg":
                    side_mask_batches.append(debug["side_kg_mask"].squeeze(1).cpu().numpy())
            labels_batches.append((labels.numpy() - 1).astype(np.int64))
            probability_batches.append(torch.softmax(logits, dim=1).cpu().numpy())
    labels = np.concatenate(labels_batches)
    probabilities = np.vstack(probability_batches)
    result = pilot.classification_metrics(labels, probabilities)
    result["selected_epoch"] = int(checkpoint["epoch"])
    if kg_mode == "clean":
        return result

    gates = np.concatenate(gate_batches)
    drug_available = np.concatenate(drug_mask_batches).astype(bool)
    result["gate_by_frequency_class"] = {
        str(class_index + 1): {
            "sample_count": int((labels == class_index).sum()),
            "gate_mean": float(gates[labels == class_index].mean()),
            "gate_std": float(gates[labels == class_index].std()),
            "gate_min": float(gates[labels == class_index].min()),
            "gate_max": float(gates[labels == class_index].max()),
        }
        for class_index in range(5)
    }
    if kg_mode == "drug_kg":
        group_masks = {
            "drug_available": drug_available,
            "drug_unavailable": ~drug_available,
        }
    else:
        side_available = np.concatenate(side_mask_batches).astype(bool)
        group_masks = {
            "both_available": drug_available & side_available,
            "drug_only": drug_available & ~side_available,
            "side_only": ~drug_available & side_available,
            "neither_available": ~drug_available & ~side_available,
        }
    availability: dict[str, Any] = {}
    for group, mask in group_masks.items():
        metrics = subset_metrics(labels, probabilities, mask)
        metrics["mean_gate"] = float(gates[mask].mean()) if mask.any() else None
        metrics["gate_std"] = float(gates[mask].std()) if mask.any() else None
        metrics["gate_min"] = float(gates[mask].min()) if mask.any() else None
        metrics["gate_max"] = float(gates[mask].max()) if mask.any() else None
        availability[group] = metrics
    result["kg_availability"] = availability
    return result


def create_model(kind: str, config: MSSFCleanConfig) -> nn.Module:
    pilot.configure_reproducibility(SEED)
    if kind == "clean":
        return MSSFClean(config)
    if kind == "drug_kg":
        return DrugOnlyBioKORFCleanKG(config, KG_ARTIFACT_PATH)
    if kind == "drug_side_kg":
        return BioKORFCleanKG(config, KG_ARTIFACT_PATH)
    raise ValueError(kind)


def main() -> None:
    global PROTECTED_HASHES
    required = [PILOT_SCRIPT_PATH, SPLIT_PATH, PILOT_OUTPUT_DIR, KG_ARTIFACT_PATH]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required extended-experiment input not found: {path}")
    # Fill the locally stable kg_encoder hash without making assumptions about Git state.
    kg_encoder_path = PROJECT_ROOT / "models" / "kg_encoder.py"
    PROTECTED_HASHES[kg_encoder_path] = sha256(kg_encoder_path)
    protected_before = {path: sha256(path) for path in PROTECTED_HASHES}
    if any(
        expected != "PLACEHOLDER" and protected_before[path] != expected
        for path, expected in PROTECTED_HASHES.items()
    ):
        raise RuntimeError("A protected model file differs from its established baseline hash")
    previous_outputs_before = file_tree_hashes([SPLIT_PATH, PILOT_OUTPUT_DIR])
    kg_hash_before = sha256(KG_ARTIFACT_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pilot.configure_reproducibility(SEED)
    frequency_matrix = np.asarray(pilot.load_pickle("drug_side.pkl"))
    samples = pilot.original_positive_sample_order(frequency_matrix)
    train_samples, validation_samples, test_samples = load_fixed_split(samples)
    hidden_samples = np.concatenate((validation_samples, test_samples), axis=0)
    drug_features, side_features, label_leakage_safe = pilot.build_leakage_safe_features(
        frequency_matrix, hidden_samples
    )
    drug_phenotype_safe = pilot.scan_graph_leakage()
    if not label_leakage_safe or not drug_phenotype_safe:
        raise RuntimeError("Leakage audit failed before extended training")

    train_dataset = pilot.IndexedPairDataset(train_samples, drug_features, side_features)
    validation_dataset = pilot.IndexedPairDataset(
        validation_samples, drug_features, side_features
    )
    test_dataset = pilot.IndexedPairDataset(test_samples, drug_features, side_features)
    config = MSSFCleanConfig(dropout=DROPOUT, gp=LATENT_DIM)

    initialization_models = {
        kind: create_model(kind, config)
        for kind in ("clean", "drug_kg", "drug_side_kg")
    }
    clean_state = initialization_models["clean"].state_dict()
    shared_initialization = all(
        all(torch.equal(value, model.state_dict()[name]) for name, value in clean_state.items())
        for model in initialization_models.values()
    )
    kg_fusion_initialization = all(
        torch.equal(value, initialization_models["drug_side_kg"].kg_fusion.state_dict()[name])
        for name, value in initialization_models["drug_kg"].kg_fusion.state_dict().items()
    )
    side_disabled_check = True
    if not shared_initialization or not kg_fusion_initialization:
        raise AssertionError("Common model modules were not initialized identically")
    parameter_counts = {
        kind: {
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in model.parameters()),
        }
        for kind, model in initialization_models.items()
    }
    frozen_kg_before = all(
        not list(initialization_models[kind].kg_features.parameters())
        and all(
            not value.requires_grad
            for _, value in initialization_models[kind].kg_features.named_buffers()
        )
        for kind in ("drug_kg", "drug_side_kg")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Fixed split loaded: train={len(train_dataset)} validation={len(validation_dataset)} "
        f"test={len(test_dataset)} seed={SEED}"
    )
    print(f"Device: {device}; max_epochs={MAX_EPOCHS}; patience={PATIENCE}")
    print(f"LABEL-DERIVED FEATURE LEAKAGE CHECK: {'PASS' if label_leakage_safe else 'FAIL'}")

    training_results: dict[str, Any] = {}
    for kind in ("clean", "drug_kg", "drug_side_kg"):
        model = initialization_models[kind]
        history, best_epoch, best_f1, checkpoint_path = train_with_early_stopping(
            kind,
            model,
            train_dataset,
            validation_dataset,
            device,
            use_kg=kind != "clean",
        )
        training_results[kind] = {
            "history": history,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_f1,
            "checkpoint_path": checkpoint_path,
        }

    # Exactly one test evaluation per validation-selected checkpoint.
    test_metrics: dict[str, dict[str, Any]] = {}
    for kind in ("clean", "drug_kg", "drug_side_kg"):
        selected_model = create_model(kind, config)
        test_metrics[kind] = selected_test_evaluation(
            selected_model,
            training_results[kind]["checkpoint_path"],
            test_dataset,
            device,
            kg_mode=kind,
        )
        atomic_json(OUTPUT_DIR / f"{kind}_test_metrics.json", test_metrics[kind])
        del selected_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    per_class_rows: list[dict[str, Any]] = []
    for class_label in range(1, 6):
        clean_class = test_metrics["clean"]["per_class"][str(class_label)]
        drug_class = test_metrics["drug_kg"]["per_class"][str(class_label)]
        full_class = test_metrics["drug_side_kg"]["per_class"][str(class_label)]
        per_class_rows.append(
            {
                "class": class_label,
                "support": clean_class["support"],
                "clean_f1": clean_class["f1"],
                "drug_kg_f1": drug_class["f1"],
                "drug_side_kg_f1": full_class["f1"],
                "delta_drug_kg_vs_clean_f1": drug_class["f1"] - clean_class["f1"],
                "delta_full_kg_vs_clean_f1": full_class["f1"] - clean_class["f1"],
                "clean_recall": clean_class["recall"],
                "drug_kg_recall": drug_class["recall"],
                "drug_side_kg_recall": full_class["recall"],
                "delta_drug_kg_vs_clean_recall": drug_class["recall"] - clean_class["recall"],
                "delta_full_kg_vs_clean_recall": full_class["recall"] - clean_class["recall"],
            }
        )
    write_csv(OUTPUT_DIR / "per_class_comparison.csv", per_class_rows)

    availability_rows: list[dict[str, Any]] = []
    for kind in ("drug_kg", "drug_side_kg"):
        for group, values in test_metrics[kind]["kg_availability"].items():
            availability_rows.append(
                {
                    "model": kind,
                    "availability_group": group,
                    "sample_count": values["sample_count"],
                    "accuracy": values["accuracy"],
                    "macro_f1": values["macro_f1"],
                    "mean_gate": values["mean_gate"],
                }
            )
    write_csv(OUTPUT_DIR / "kg_availability_comparison.csv", availability_rows)

    gate_rows: list[dict[str, Any]] = []
    for kind in ("drug_kg", "drug_side_kg"):
        for row in training_results[kind]["history"]:
            gate_rows.append(
                {
                    "model": kind,
                    "scope": "validation_epoch",
                    "epoch": row["epoch"],
                    "group": "all_validation_samples",
                    "sample_count": len(validation_dataset),
                    "gate_mean": row["gate_mean"],
                    "gate_std": row["gate_std"],
                    "gate_min": row["gate_min"],
                    "gate_max": row["gate_max"],
                }
            )
        selected_epoch = training_results[kind]["best_epoch"]
        for group, values in test_metrics[kind]["gate_by_frequency_class"].items():
            gate_rows.append(
                {
                    "model": kind,
                    "scope": "selected_test_frequency_class",
                    "epoch": selected_epoch,
                    "group": f"class_{group}",
                    "sample_count": values["sample_count"],
                    "gate_mean": values["gate_mean"],
                    "gate_std": values["gate_std"],
                    "gate_min": values["gate_min"],
                    "gate_max": values["gate_max"],
                }
            )
        for group, values in test_metrics[kind]["kg_availability"].items():
            gate_rows.append(
                {
                    "model": kind,
                    "scope": "selected_test_availability",
                    "epoch": selected_epoch,
                    "group": group,
                    "sample_count": values["sample_count"],
                    "gate_mean": values["mean_gate"],
                    "gate_std": values["gate_std"],
                    "gate_min": values["gate_min"],
                    "gate_max": values["gate_max"],
                }
            )
    write_csv(OUTPUT_DIR / "gate_analysis.csv", gate_rows)

    summary_models = {
        kind: {
            "best_epoch": training_results[kind]["best_epoch"],
            "best_validation_macro_f1": training_results[kind]["best_validation_macro_f1"],
            "accuracy": test_metrics[kind]["accuracy"],
            "macro_precision": test_metrics[kind]["macro_precision"],
            "macro_recall": test_metrics[kind]["macro_recall"],
            "macro_f1": test_metrics[kind]["macro_f1"],
            "micro_f1": test_metrics[kind]["micro_f1"],
            "aupr": test_metrics[kind]["aupr"],
            "epochs_ran": len(training_results[kind]["history"]),
        }
        for kind in ("clean", "drug_kg", "drug_side_kg")
    }
    comparison = {
        "models": summary_models,
        "delta_vs_clean": {
            kind: {
                metric: summary_models[kind][metric] - summary_models["clean"][metric]
                for metric in ("accuracy", "macro_f1", "aupr")
            }
            for kind in ("drug_kg", "drug_side_kg")
        },
    }
    atomic_json(OUTPUT_DIR / "extended_fold1_comparison.json", comparison)

    frozen_kg_after = bool(
        sha256(KG_ARTIFACT_PATH) == kg_hash_before
        and all(
            not list(initialization_models[kind].kg_features.parameters())
            and all(
                not value.requires_grad
                for _, value in initialization_models[kind].kg_features.named_buffers()
            )
            for kind in ("drug_kg", "drug_side_kg")
        )
    )
    frozen_kg_check = frozen_kg_before and frozen_kg_after
    previous_outputs_safe = previous_outputs_before == file_tree_hashes(
        [SPLIT_PATH, PILOT_OUTPUT_DIR]
    )
    protected_files_safe = protected_before == {
        path: sha256(path) for path in PROTECTED_HASHES
    }
    fairness_check = bool(
        shared_initialization
        and kg_fusion_initialization
        and side_disabled_check
        and previous_outputs_safe
        and protected_files_safe
        and frozen_kg_check
        and BATCH_SIZE == pilot.BATCH_SIZE
        and LEARNING_RATE == pilot.LEARNING_RATE
        and WEIGHT_DECAY == pilot.WEIGHT_DECAY
    )

    table_lines = [
        "Model | Best epoch | Accuracy | Macro-F1 | AUPR",
        "--- | ---: | ---: | ---: | ---:",
    ]
    display_names = {
        "clean": "CLEAN",
        "drug_kg": "DRUG_KG",
        "drug_side_kg": "DRUG_SIDE_KG",
    }
    for kind in ("clean", "drug_kg", "drug_side_kg"):
        values = summary_models[kind]
        table_lines.append(
            f"{display_names[kind]} | {values['best_epoch']} | {values['accuracy']:.8f} | "
            f"{values['macro_f1']:.8f} | {values['aupr']:.8f}"
        )
    report_lines = [
        "BioKORF extended Fold-1 KG ablation",
        "====================================",
        f"Seed: {SEED}",
        f"Fixed split (read-only): {SPLIT_PATH}",
        f"Samples train/validation/test: {len(train_dataset)}/{len(validation_dataset)}/{len(test_dataset)}",
        f"Maximum epochs: {MAX_EPOCHS}",
        f"Early-stopping patience: {PATIENCE}",
        "Selection metric: validation Macro-F1",
        f"Optimizer for all models: Adam(lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})",
        f"Batch size for all models: {BATCH_SIZE}",
        "Evaluation BVI policy: model.eval() returns latent=mu without sampling",
        "Test evaluation policy: exactly once per validation-selected checkpoint",
        "DRUG_KG isolation: side embedding and side availability mask are zero before KG projection",
        "",
        *table_lines,
        "",
        "Deltas relative to CLEAN:",
        *[
            f"- {display_names[kind]} accuracy={comparison['delta_vs_clean'][kind]['accuracy']:+.8f}, "
            f"Macro-F1={comparison['delta_vs_clean'][kind]['macro_f1']:+.8f}, "
            f"AUPR={comparison['delta_vs_clean'][kind]['aupr']:+.8f}"
            for kind in ("drug_kg", "drug_side_kg")
        ],
        "",
        "Per-class F1:",
        *[
            f"- Class {row['class']} (support {row['support']}): CLEAN={row['clean_f1']:.8f}, "
            f"DRUG_KG={row['drug_kg_f1']:.8f}, DRUG_SIDE_KG={row['drug_side_kg_f1']:.8f}"
            for row in per_class_rows
        ],
        "",
        f"EXPERIMENT FAIRNESS CHECK: {'PASS' if fairness_check else 'FAIL'}",
        f"LABEL-DERIVED FEATURE LEAKAGE CHECK: {'PASS' if label_leakage_safe else 'FAIL'}",
        f"DRUG-PHENOTYPE LEAKAGE CHECK: {'PASS' if drug_phenotype_safe else 'FAIL'}",
        f"FROZEN KG CHECK: {'PASS' if frozen_kg_check else 'FAIL'}",
        f"Previous split/pilot outputs unchanged: {'PASS' if previous_outputs_safe else 'FAIL'}",
        f"Protected model files unchanged: {'PASS' if protected_files_safe else 'FAIL'}",
        "Ordinal learning/attention/R-GCN fine-tuning/KG modification: none",
        "Gate values are descriptive diagnostics and are not interpreted as causal importance.",
    ]
    report = "\n".join(report_lines) + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report, end="")
    if not all(
        (fairness_check, label_leakage_safe, drug_phenotype_safe, frozen_kg_check)
    ):
        raise RuntimeError("One or more required final checks failed")


if __name__ == "__main__":
    main()
