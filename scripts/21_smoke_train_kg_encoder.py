"""Perform exactly one self-supervised BioKORF KG optimization step."""

from __future__ import annotations

import csv
import json
import random
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RGCN_DIR = PROJECT_ROOT / "data_processed" / "rgcn"
NODE_INDEX_PATH = RGCN_DIR / "node_index.csv"
RELATION_INDEX_PATH = RGCN_DIR / "relation_index.csv"
MESSAGE_EDGES_PATH = RGCN_DIR / "edges_with_inverse.csv"
SUPERVISION_EDGES_PATH = RGCN_DIR / "edges_indexed.csv"
FEATURE_PATH = RGCN_DIR / "node_feature_metadata.csv"
METADATA_PATH = RGCN_DIR / "graph_metadata.json"
REPORT_PATH = RGCN_DIR / "kg_pretraining_smoke_test.txt"

SEED = 42
MAX_POSITIVES = 50_000
HIDDEN_DIM = 128
OUTPUT_DIM = 128
NUM_BASES = 8
DROPOUT = 0.2
EXCLUDED_SUPERVISION = {"MAPS_TO_DRUG", "MAPS_TO_PHENOTYPE"}


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required graph artifact not found: {path}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def emit_report(lines: list[str]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines).rstrip() + "\n"
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(text, end="")


def load_message_graph() -> tuple[array, array, array]:
    sources = array("q")
    targets = array("q")
    relations = array("q")
    with MESSAGE_EDGES_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sources.append(int(row["source_index"]))
            targets.append(int(row["target_index"]))
            relations.append(int(row["relation_index"]))
    return sources, targets, relations


def load_original_edges() -> tuple[
    list[tuple[int, int, int, str, str, str]],
    set[tuple[int, int, int]],
    dict[str, set[tuple[str, str]]],
    Counter[str],
]:
    eligible: list[tuple[int, int, int, str, str, str]] = []
    all_positive: set[tuple[int, int, int]] = set()
    allowed_types: dict[str, set[tuple[str, str]]] = defaultdict(set)
    relation_counts: Counter[str] = Counter()
    with SUPERVISION_EDGES_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source = int(row["source_index"])
            target = int(row["target_index"])
            relation_index = int(row["relation_index"])
            relation = row["relation"]
            source_type = row["source_type"]
            target_type = row["target_type"]
            triple = (source, relation_index, target)
            all_positive.add(triple)
            allowed_types[relation].add((source_type, target_type))
            relation_counts[relation] += 1
            if {source_type, target_type} == {"DRUG", "PHENOTYPE"}:
                raise ValueError(f"Drug-Phenotype edge detected in original graph: {row}")
            if relation.upper() == "ADVERSE_DRUG_REACTION":
                raise ValueError(f"ADVERSE_DRUG_REACTION relation detected: {row}")
            if {source_type, target_type} == {"BIOKORF_DRUG", "BIOKORF_SIDE"}:
                raise ValueError(f"Direct BioKORF drug-side anchor edge detected: {row}")
            if relation not in EXCLUDED_SUPERVISION:
                eligible.append(
                    (source, relation_index, target, source_type, target_type, relation)
                )
    return eligible, all_positive, allowed_types, relation_counts


def sample_negatives(
    positives: list[tuple[int, int, int, str, str, str]],
    all_positive: set[tuple[int, int, int]],
    nodes_by_type: dict[str, list[int]],
    rng: random.Random,
) -> list[tuple[int, int, int]]:
    negatives: list[tuple[int, int, int]] = []
    for source, relation, target, source_type, target_type, _ in positives:
        for _attempt in range(10_000):
            if rng.randrange(2) == 0:
                candidate = (rng.choice(nodes_by_type[source_type]), relation, target)
            else:
                candidate = (source, relation, rng.choice(nodes_by_type[target_type]))
            if candidate not in all_positive:
                negatives.append(candidate)
                break
        else:
            raise RuntimeError(
                f"Unable to sample a type-valid negative for ({source}, {relation}, {target})"
            )
    return negatives


def parameter_finite(model: Any) -> bool:
    import torch

    return all(torch.isfinite(parameter).all().item() for parameter in model.parameters())


def main() -> int:
    import torch
    import torch_geometric

    sys.path.insert(0, str(PROJECT_ROOT))
    from models.kg_encoder import BioKORFKGEncoder
    from models.kg_pretraining import BioKORFDistMultDecoder

    for path in (
        NODE_INDEX_PATH,
        RELATION_INDEX_PATH,
        MESSAGE_EDGES_PATH,
        SUPERVISION_EDGES_PATH,
        FEATURE_PATH,
        METADATA_PATH,
    ):
        require_file(path)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("leakage_check") != "PASS":
        raise ValueError("Graph metadata leakage check is not PASS")
    node_rows = read_rows(NODE_INDEX_PATH)
    feature_rows = read_rows(FEATURE_PATH)
    relation_rows = read_rows(RELATION_INDEX_PATH)
    num_nodes = len(node_rows)
    num_relations_message = len(relation_rows)
    num_relations_original = int(metadata["num_relations_original"])
    if [int(row["node_index"]) for row in node_rows] != list(range(num_nodes)):
        raise ValueError("node_index is not exactly continuous")
    if [int(row["node_index"]) for row in feature_rows] != list(range(num_nodes)):
        raise ValueError("node_feature_metadata ordering does not match node_index")
    if [int(row["relation_index"]) for row in relation_rows] != list(range(num_relations_message)):
        raise ValueError("relation_index is not exactly continuous")
    if any(
        int(row["relation_index"]) >= num_relations_original
        for row in relation_rows
        if not row["relation"].endswith("__INV")
    ):
        raise ValueError("An original relation has an inverse-range index")

    node_type_values = [int(row["type_index"]) for row in feature_rows]
    nodes_by_type: dict[str, list[int]] = defaultdict(list)
    for row in node_rows:
        nodes_by_type[row["node_type"]].append(int(row["node_index"]))

    eligible, all_positive, allowed_types, relation_counts = load_original_edges()
    rng = random.Random(SEED)
    positive_count = min(MAX_POSITIVES, len(eligible))
    positives = rng.sample(eligible, positive_count)
    negatives = sample_negatives(positives, all_positive, nodes_by_type, rng)
    if len(negatives) != positive_count:
        raise AssertionError("Negative count does not equal positive count")
    if any(relation in EXCLUDED_SUPERVISION for *_, relation in positives):
        raise AssertionError("Anchor mapping relation entered link supervision")
    for source, relation, target in negatives:
        if (source, relation, target) in all_positive:
            raise AssertionError("A sampled negative is an original positive triple")

    message_sources, message_targets, message_relations = load_message_graph()
    expected_message_edges = int(metadata["num_edges_with_inverse"])
    if len(message_sources) != expected_message_edges:
        raise ValueError(
            f"Expected {expected_message_edges} message-passing edges; found {len(message_sources)}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    node_type_index = torch.tensor(node_type_values, dtype=torch.long, device=device)
    edge_index = torch.stack(
        (
            torch.tensor(message_sources, dtype=torch.long),
            torch.tensor(message_targets, dtype=torch.long),
        )
    ).to(device)
    edge_type = torch.tensor(message_relations, dtype=torch.long, device=device)
    positive_source = torch.tensor([row[0] for row in positives], dtype=torch.long, device=device)
    positive_relation = torch.tensor([row[1] for row in positives], dtype=torch.long, device=device)
    positive_target = torch.tensor([row[2] for row in positives], dtype=torch.long, device=device)
    negative_source = torch.tensor([row[0] for row in negatives], dtype=torch.long, device=device)
    negative_relation = torch.tensor([row[1] for row in negatives], dtype=torch.long, device=device)
    negative_target = torch.tensor([row[2] for row in negatives], dtype=torch.long, device=device)

    encoder = BioKORFKGEncoder(
        num_node_types=len(set(node_type_values)),
        num_relations=num_relations_message,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_bases=NUM_BASES,
        dropout=DROPOUT,
    ).to(device)
    decoder = BioKORFDistMultDecoder(
        num_relations=num_relations_original, embedding_dim=OUTPUT_DIM
    ).to(device)
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *decoder.parameters()],
        lr=1e-3,
        weight_decay=1e-5,
    )
    loss_function = torch.nn.BCEWithLogitsLoss()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        encoder.train()
        decoder.train()
        optimizer.zero_grad(set_to_none=True)
        node_embeddings = encoder(node_type_index, edge_index, edge_type)
        positive_scores = decoder(
            node_embeddings, positive_source, positive_relation, positive_target
        )
        negative_scores = decoder(
            node_embeddings, negative_source, negative_relation, negative_target
        )
        logits = torch.cat((positive_scores, negative_scores))
        labels = torch.cat(
            (torch.ones_like(positive_scores), torch.zeros_like(negative_scores))
        )
        loss = loss_function(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Initial BCE loss is not finite")
        loss.backward()
        gradient_parameters = [
            parameter
            for parameter in [*encoder.parameters(), *decoder.parameters()]
            if parameter.grad is not None
        ]
        gradients_finite = bool(gradient_parameters) and all(
            torch.isfinite(parameter.grad).all().item()
            for parameter in gradient_parameters
        )
        if not gradients_finite:
            raise FloatingPointError("At least one computed gradient is non-finite")
        optimizer.step()  # Exactly one optimization step.
        parameters_finite = parameter_finite(encoder) and parameter_finite(decoder)
        if not parameters_finite:
            raise FloatingPointError("A parameter became non-finite after optimizer.step()")
    except torch.cuda.OutOfMemoryError as error:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        lines = [
            "BioKORF KG pretraining smoke test: CUDA OOM",
            "=" * 45,
            f"Device: {device}",
            f"Positive sample count: {positive_count}",
            f"Negative sample count: {len(negatives)}",
            f"CUDA error: {error}",
            "Optimization step: FAIL (gracefully caught CUDA out-of-memory)",
            "LEAKAGE CHECK: PASS",
            "Training steps completed: 0",
            "Checkpoint saved: no",
            "Embeddings saved: no",
        ]
        emit_report(lines)
        return 0

    peak_allocated = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    )
    supervised_relation_counts = Counter(row[5] for row in positives)
    type_signatures = {
        relation: sorted(f"{source_type}->{target_type}" for source_type, target_type in pairs)
        for relation, pairs in sorted(allowed_types.items())
        if relation not in EXCLUDED_SUPERVISION
    }
    lines = [
        "BioKORF KG pretraining smoke test: PASS",
        "=" * 41,
        f"PyTorch version: {torch.__version__}",
        f"PyG version: {torch_geometric.__version__}",
        f"Device: {device}",
        f"Message-passing nodes: {num_nodes}",
        f"Message-passing edges (with inverse): {len(message_sources)}",
        f"Message-passing relations: {num_relations_message}",
        f"Original decoder relations: {num_relations_original}",
        f"Positive sample count: {positive_count}",
        f"Negative sample count: {len(negatives)}",
        f"Initial loss: {loss.detach().item():.8f}",
        f"Mean positive score: {positive_scores.detach().mean().item():.8f}",
        f"Mean negative score: {negative_scores.detach().mean().item():.8f}",
        f"Gradient finite check: {'PASS' if gradients_finite else 'FAIL'}",
        f"Parameters with gradients: {len(gradient_parameters)}",
        f"All parameters finite after step: {'PASS' if parameters_finite else 'FAIL'}",
        f"Peak CUDA allocated memory: {peak_allocated} bytes ({peak_allocated / 1024**2:.2f} MiB)",
        f"Peak CUDA reserved memory: {peak_reserved} bytes ({peak_reserved / 1024**2:.2f} MiB)",
        "Optimizer: AdamW(lr=0.001, weight_decay=0.00001)",
        "Optimization steps completed: 1",
        "Optimization step: PASS",
        f"Excluded supervision relations: {sorted(EXCLUDED_SUPERVISION)}",
        f"Sampled supervision relation counts: {dict(sorted(supervised_relation_counts.items()))}",
        f"Observed type signatures by supervised relation: {type_signatures}",
        "No Drug-Phenotype edge used: PASS",
        "No ADVERSE_DRUG_REACTION relation: PASS",
        "No drug-side-effect frequency labels loaded: PASS",
        "No direct BIOKORF_DRUG -> BIOKORF_SIDE supervision: PASS",
        "LEAKAGE CHECK: PASS",
        "Checkpoint saved: no",
        "Embeddings saved: no",
    ]
    emit_report(lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
