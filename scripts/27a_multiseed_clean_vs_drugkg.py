"""Multi-seed Fold-1 robustness experiment: CLEAN versus DRUG_KG.

Training and test evaluation are deliberately separate CLI operations.  The
fixed Fold-1 split is always read from Step 26 and is never regenerated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_SCRIPT = PROJECT_ROOT / "scripts" / "26a_pilot_compare_clean_vs_kg.py"
EXTENDED_SCRIPT = PROJECT_ROOT / "scripts" / "26b_extended_fold1_kg_ablation.py"
SPLIT_PATH = PROJECT_ROOT / "data_processed" / "experiments" / "pilot_fold1_split.npz"
KG_ARTIFACT_PATH = PROJECT_ROOT / "data_processed" / "kg_features" / "biokorf_kg_embeddings.pt"
OUTPUT_ROOT = PROJECT_ROOT / "data_processed" / "experiments" / "multiseed_fold1"

MODELS = ("clean", "drug_kg")
SEEDS = (42, 123, 2026)
MAX_EPOCHS = 30
PATIENCE = 7
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 128
DROPOUT = 0.4
LATENT_DIM = 64
CLASS_LABELS = (1, 2, 3, 4, 5)

# These established hashes make any change to the fixed split or model sources
# fail closed.  They also protect the frozen KG artifact throughout a run.
PROTECTED_HASHES = {
    PROJECT_ROOT / "mssf.py": "4867fecd04beabb2d715b24073f82a46bd572c13294afa3565ddba99f963fdb1",
    PROJECT_ROOT / "model.py": "9c0d4bf17551a7d0f881a29e0f8e2727227f3561678064fec46f2848156a1e75",
    PROJECT_ROOT / "models" / "mssf_clean.py": "f2a0f68e062807cacc77540c14afd5bf0e66eb7571b76b99e095c4063b8dd6d2",
    PROJECT_ROOT / "models" / "mssf_clean_kg.py": "cbc505f64c718cb5ce861fd4eac1d4d5d7f6eaefb0b045059271aab79bf92b81",
    PROJECT_ROOT / "models" / "kg_fusion.py": "9c9ce093d5e86078e32cd11e8696d0dc37625f8980c3cdb66167b2a893ac7c0f",
    PROJECT_ROOT / "models" / "kg_encoder.py": "0e81eb5f31299f46ac64052ea929811dccb7fccb5e203e7f50f833fab375e464",
    KG_ARTIFACT_PATH: "264717e99ad7a86a25704eeac7459022bf15838bb759c37039940d4192de6a87",
    SPLIT_PATH: "d6f4bb9854bca7296372ea580d07acc9dfe1e2bd3ed8fa6905a7bdb84a7e5575",
}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = load_module("biokorf_pilot_helpers_27a", PILOT_SCRIPT)
extended = load_module("biokorf_extended_helpers_27a", EXTENDED_SCRIPT)
sys.path.insert(0, str(PROJECT_ROOT))
from models.mssf_clean import MSSFClean, MSSFCleanConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protected_inputs() -> dict[Path, str]:
    for path in (PILOT_SCRIPT, EXTENDED_SCRIPT, *PROTECTED_HASHES):
        if not path.is_file():
            raise FileNotFoundError(f"Required experiment input not found: {path}")
    observed = {path: sha256(path) for path in PROTECTED_HASHES}
    changed = [str(path) for path, expected in PROTECTED_HASHES.items() if observed[path] != expected]
    if changed:
        raise RuntimeError("Protected experiment input differs from its baseline: " + ", ".join(changed))
    return observed


def validate_unchanged(before: dict[Path, str]) -> bool:
    return before == {path: sha256(path) for path in before}


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_directory(model_name: str, seed: int) -> Path:
    return OUTPUT_ROOT / f"{model_name}_seed{seed}"


def require_new_outputs(paths: list[Path], operation: str) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to {operation}; output already exists (remove it explicitly to rerun): "
            + ", ".join(existing)
        )


def load_fixed_experiment_data() -> tuple[Any, Any, Any, bool, bool]:
    frequency_matrix = np.asarray(pilot.load_pickle("drug_side.pkl"))
    samples = pilot.original_positive_sample_order(frequency_matrix)
    train_samples, validation_samples, test_samples = extended.load_fixed_split(samples)
    hidden_samples = np.concatenate((validation_samples, test_samples), axis=0)
    drug_features, side_features, label_safe = pilot.build_leakage_safe_features(
        frequency_matrix, hidden_samples
    )
    graph_safe = pilot.scan_graph_leakage()
    if not label_safe or not graph_safe:
        raise RuntimeError("A required leakage check failed")
    datasets = tuple(
        pilot.IndexedPairDataset(split_samples, drug_features, side_features)
        for split_samples in (train_samples, validation_samples, test_samples)
    )
    return (*datasets, bool(label_safe), bool(graph_safe))


def create_model(model_name: str, seed: int) -> nn.Module:
    pilot.configure_reproducibility(seed)
    config = MSSFCleanConfig(dropout=DROPOUT, gp=LATENT_DIM)
    if model_name == "clean":
        return MSSFClean(config)
    if model_name == "drug_kg":
        return extended.DrugOnlyBioKORFCleanKG(config, KG_ARTIFACT_PATH)
    raise ValueError(f"Unsupported model: {model_name}")


def fairness_check(seed: int) -> bool:
    clean = create_model("clean", seed)
    drug_kg = create_model("drug_kg", seed)
    clean_state = clean.state_dict()
    common_initialization = all(
        name in drug_kg.state_dict() and torch.equal(value, drug_kg.state_dict()[name])
        for name, value in clean_state.items()
    )
    frozen_kg = not list(drug_kg.kg_features.parameters()) and all(
        not value.requires_grad for _, value in drug_kg.kg_features.named_buffers()
    )
    policy_match = (
        BATCH_SIZE == pilot.BATCH_SIZE
        and LEARNING_RATE == pilot.LEARNING_RATE
        and WEIGHT_DECAY == pilot.WEIGHT_DECAY
    )
    return bool(common_initialization and frozen_kg and policy_match)


def train_one_epoch(
    model: nn.Module,
    dataset: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    seed: int,
    use_kg: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    loader = pilot.make_loader(dataset, shuffle=True, seed=seed + epoch)
    for drugs, sides, drug_index, side_index, labels in loader:
        drugs = drugs.to(device, non_blocking=True)
        sides = sides.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_kg:
            outputs = model(
                drugs,
                sides,
                drug_index.to(device, non_blocking=True),
                side_index.to(device, non_blocking=True),
                device=device,
            )
        else:
            outputs = model(drugs, sides, device=device)
        logits, rec_con, rec_add, mu, logvar = outputs
        loss = pilot.composite_loss(
            logits, rec_con, rec_add, mu, logvar, labels, drugs, sides
        )
        loss.backward()
        optimizer.step()
        count = int(labels.shape[0])
        total_loss += float(loss.detach()) * count
        total_samples += count
    return total_loss / total_samples


def checkpoint_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_mode(model_name: str, seed: int) -> None:
    before = validate_protected_inputs()
    output_dir = run_directory(model_name, seed)
    history_path = output_dir / "training_history.csv"
    checkpoint_path = output_dir / "best_checkpoint.pt"
    require_new_outputs([history_path, checkpoint_path], "train")
    train_data, validation_data, _test_data, label_safe, graph_safe = load_fixed_experiment_data()
    fair = fairness_check(seed)
    if not fair:
        raise RuntimeError("Experiment fairness check failed before training")

    output_dir.mkdir(parents=True, exist_ok=True)
    pilot.configure_reproducibility(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_name, seed).to(device)
    use_kg = model_name == "drug_kg"
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    columns = [
        "epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "val_aupr",
        "gate_mean", "gate_std", "gate_min", "gate_max",
    ]
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_macro_f1 = -1.0
    stale_epochs = 0
    print(f"Training {model_name}, seed={seed}, device={device}; fixed split={SPLIT_PATH}")
    for epoch in range(1, MAX_EPOCHS + 1):
        pilot.configure_reproducibility(seed + epoch)
        started = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_data, optimizer, device, epoch, seed, use_kg
        )
        val_loss, metrics, gates = pilot.evaluate(
            model, validation_data, device, use_kg=use_kg
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "val_aupr": metrics["aupr"],
            "gate_mean": gates.get("gate_mean", ""),
            "gate_std": gates.get("gate_std", ""),
            "gate_min": gates.get("gate_min", ""),
            "gate_max": gates.get("gate_max", ""),
        }
        history.append(row)
        write_csv(history_path, history, columns)
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(metrics["macro_f1"])
            best_epoch = epoch
            stale_epochs = 0
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "model": model_name,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_macro_f1": best_macro_f1,
                    "model_state_dict": checkpoint_state(model),
                    "selection_metric": "validation_macro_f1",
                    "optimizer_policy": {
                        "name": "Adam",
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "batch_size": BATCH_SIZE,
                        "loss": "CrossEntropyLoss plus unchanged MSSF BVI/reconstruction policy",
                    },
                    "checks": {
                        "experiment_fairness": fair,
                        "label_derived_feature_leakage": label_safe,
                        "drug_phenotype_leakage": graph_safe,
                        "frozen_kg": use_kg is False or sha256(KG_ARTIFACT_PATH) == before[KG_ARTIFACT_PATH],
                    },
                },
                temporary,
            )
            temporary.replace(checkpoint_path)
        else:
            stale_epochs += 1
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_macro_f1={metrics['macro_f1']:.6f} best_epoch={best_epoch} "
            f"patience={stale_epochs}/{PATIENCE} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        if stale_epochs >= PATIENCE:
            break
    if not validate_unchanged(before):
        raise RuntimeError("A protected model, split, or frozen KG artifact changed during training")
    print(f"Best validation epoch: {best_epoch}; Macro-F1={best_macro_f1:.8f}")
    print("EXPERIMENT FAIRNESS CHECK: PASS")
    print("LABEL-DERIVED FEATURE LEAKAGE CHECK: PASS")
    print("DRUG-PHENOTYPE LEAKAGE CHECK: PASS")
    print("FROZEN KG CHECK: PASS")


def evaluate_test_once(
    model: nn.Module, dataset: Any, device: torch.device, use_kg: bool, seed: int
) -> dict[str, Any]:
    model.eval()
    labels_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for drugs, sides, drug_index, side_index, labels in pilot.make_loader(
            dataset, shuffle=False, seed=seed
        ):
            drugs = drugs.to(device, non_blocking=True)
            sides = sides.to(device, non_blocking=True)
            if use_kg:
                logits, *_ = model(
                    drugs,
                    sides,
                    drug_index.to(device, non_blocking=True),
                    side_index.to(device, non_blocking=True),
                    device=device,
                )
            else:
                logits, *_ = model(drugs, sides, device=device)
            labels_batches.append((labels.numpy() - 1).astype(np.int64))
            probability_batches.append(torch.softmax(logits, dim=1).cpu().numpy())
    return pilot.classification_metrics(
        np.concatenate(labels_batches), np.vstack(probability_batches)
    )


def test_mode(model_name: str, seed: int) -> None:
    before = validate_protected_inputs()
    output_dir = run_directory(model_name, seed)
    history_path = output_dir / "training_history.csv"
    checkpoint_path = output_dir / "best_checkpoint.pt"
    if not history_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Train mode must complete before test mode: {output_dir}")
    metrics_path = output_dir / "test_metrics.json"
    confusion_path = output_dir / "confusion_matrix.csv"
    per_class_path = output_dir / "per_class_metrics.csv"
    require_new_outputs([metrics_path, confusion_path, per_class_path], "test")
    _train_data, _validation_data, test_data, label_safe, graph_safe = load_fixed_experiment_data()
    fair = fairness_check(seed)
    if not fair:
        raise RuntimeError("Experiment fairness check failed before test evaluation")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model") != model_name or int(checkpoint.get("seed", -1)) != seed:
        raise ValueError("Checkpoint model/seed does not match the requested run")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_name, seed)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    metrics = evaluate_test_once(model, test_data, device, model_name == "drug_kg", seed)
    metrics.update(
        {
            "model": model_name,
            "seed": seed,
            "selected_epoch": int(checkpoint["epoch"]),
            "selection_metric": "validation_macro_f1",
            "best_validation_macro_f1": float(checkpoint["validation_macro_f1"]),
            "test_evaluation_policy": "one evaluation after validation-only model selection",
            "checks": {
                "experiment_fairness": fair,
                "label_derived_feature_leakage": label_safe,
                "drug_phenotype_leakage": graph_safe,
                "frozen_kg": model_name == "clean" or sha256(KG_ARTIFACT_PATH) == before[KG_ARTIFACT_PATH],
            },
        }
    )
    write_csv(
        confusion_path,
        [
            {"true_class": true_label, **{f"predicted_{label}": row[label - 1] for label in CLASS_LABELS}}
            for true_label, row in zip(CLASS_LABELS, metrics["confusion_matrix"])
        ],
        ["true_class", *[f"predicted_{label}" for label in CLASS_LABELS]],
    )
    write_csv(
        per_class_path,
        [
            {"class": label, **metrics["per_class"][str(label)]}
            for label in CLASS_LABELS
        ],
        ["class", "precision", "recall", "f1", "support"],
    )
    atomic_json(metrics_path, metrics)
    if not validate_unchanged(before):
        raise RuntimeError("A protected model, split, or frozen KG artifact changed during testing")
    print(
        f"{model_name} seed={seed}: Accuracy={metrics['accuracy']:.8f} "
        f"Macro-F1={metrics['macro_f1']:.8f} AUPR={metrics['aupr']:.8f}"
    )
    print("EXPERIMENT FAIRNESS CHECK: PASS")
    print("LABEL-DERIVED FEATURE LEAKAGE CHECK: PASS")
    print("DRUG-PHENOTYPE LEAKAGE CHECK: PASS")
    print("FROZEN KG CHECK: PASS")


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1))


def compare_mode() -> None:
    completed: dict[str, dict[int, dict[str, Any]]] = {name: {} for name in MODELS}
    for model_name in MODELS:
        for seed in SEEDS:
            path = run_directory(model_name, seed) / "test_metrics.json"
            if not path.is_file():
                raise FileNotFoundError(f"Compare mode requires completed test result: {path}")
            completed[model_name][seed] = json.loads(path.read_text(encoding="utf-8"))

    metrics = ("accuracy", "macro_f1", "aupr")
    rows: list[dict[str, Any]] = []
    report = [
        "BioKORF multi-seed Fold-1 robustness comparison",
        "================================================",
        f"Seeds: {', '.join(map(str, SEEDS))}",
        f"Fixed split: {SPLIT_PATH}",
        "Models: CLEAN and DRUG_KG",
        "",
        "Per-seed results:",
        "Model | Seed | Accuracy | Macro-F1 | AUPR",
        "--- | ---: | ---: | ---: | ---:",
    ]
    for model_name in MODELS:
        for seed in SEEDS:
            result = completed[model_name][seed]
            rows.append(
                {
                    "row_type": "run",
                    "model": model_name,
                    "seed": seed,
                    "accuracy": result["accuracy"],
                    "macro_f1": result["macro_f1"],
                    "aupr": result["aupr"],
                    "accuracy_std": "",
                    "macro_f1_std": "",
                    "aupr_std": "",
                    "drug_kg_beats_clean_macro_f1": "",
                }
            )
            report.append(
                f"{model_name.upper()} | {seed} | {result['accuracy']:.8f} | "
                f"{result['macro_f1']:.8f} | {result['aupr']:.8f}"
            )

    report.extend(["", "Aggregate results (sample standard deviation):"])
    aggregate: dict[str, dict[str, tuple[float, float]]] = {}
    for model_name in MODELS:
        aggregate[model_name] = {
            metric: mean_std([float(completed[model_name][seed][metric]) for seed in SEEDS])
            for metric in metrics
        }
        values = aggregate[model_name]
        rows.append(
            {
                "row_type": "model_summary",
                "model": model_name,
                "seed": "",
                "accuracy": values["accuracy"][0],
                "macro_f1": values["macro_f1"][0],
                "aupr": values["aupr"][0],
                "accuracy_std": values["accuracy"][1],
                "macro_f1_std": values["macro_f1"][1],
                "aupr_std": values["aupr"][1],
                "drug_kg_beats_clean_macro_f1": "",
            }
        )
        report.append(
            f"{model_name.upper()}: Accuracy={values['accuracy'][0]:.8f} +/- {values['accuracy'][1]:.8f}; "
            f"Macro-F1={values['macro_f1'][0]:.8f} +/- {values['macro_f1'][1]:.8f}; "
            f"AUPR={values['aupr'][0]:.8f} +/- {values['aupr'][1]:.8f}"
        )

    report.extend(["", "Paired per-seed deltas (DRUG_KG - CLEAN):", "Seed | Accuracy | Macro-F1 | AUPR", "---: | ---: | ---: | ---:"])
    wins = 0
    for seed in SEEDS:
        delta = {
            metric: float(completed["drug_kg"][seed][metric]) - float(completed["clean"][seed][metric])
            for metric in metrics
        }
        wins += int(delta["macro_f1"] > 0.0)
        rows.append(
            {
                "row_type": "paired_delta",
                "model": "drug_kg_minus_clean",
                "seed": seed,
                "accuracy": delta["accuracy"],
                "macro_f1": delta["macro_f1"],
                "aupr": delta["aupr"],
                "accuracy_std": "",
                "macro_f1_std": "",
                "aupr_std": "",
                "drug_kg_beats_clean_macro_f1": int(delta["macro_f1"] > 0.0),
            }
        )
        report.append(
            f"{seed} | {delta['accuracy']:+.8f} | {delta['macro_f1']:+.8f} | {delta['aupr']:+.8f}"
        )
    report.extend(
        [
            "",
            f"DRUG_KG beats CLEAN on Macro-F1 for {wins} of {len(SEEDS)} seeds.",
            "Compare mode performed no training or test evaluation.",
        ]
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUTPUT_ROOT / "multiseed_summary.csv",
        rows,
        [
            "row_type", "model", "seed", "accuracy", "macro_f1", "aupr",
            "accuracy_std", "macro_f1_std", "aupr_std", "drug_kg_beats_clean_macro_f1",
        ],
    )
    text = "\n".join(report) + "\n"
    (OUTPUT_ROOT / "multiseed_report.txt").write_text(text, encoding="utf-8")
    print(text, end="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("train", "test", "compare"))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.mode in ("train", "test") and (args.model is None or args.seed is None):
        parser.error("--model and --seed are required for train and test modes")
    if args.mode == "compare" and (args.model is not None or args.seed is not None):
        parser.error("compare mode does not accept --model or --seed")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train_mode(args.model, args.seed)
    elif args.mode == "test":
        test_mode(args.model, args.seed)
    else:
        compare_mode()


if __name__ == "__main__":
    main()
