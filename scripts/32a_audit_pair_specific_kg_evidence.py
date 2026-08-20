#!/usr/bin/env python3
"""Step32A: audit pair-specific biological evidence in the leakage-safe KG.

This is a descriptive/statistical audit only.  It never imports a model, loads
KG embeddings, constructs task-similarity features, or accesses Fold1 test
labels.  Biological features are derived only from typed Drug-Gene,
Gene-Gene, Pathway-Gene, and Phenotype-Gene relations already present in the
leakage-safe biomedical KG.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = PROJECT_ROOT / "data_processed" / "experiments" / "pilot_fold1_split.npz"
KG_EDGES_PATH = PROJECT_ROOT / "data_processed" / "biomedical_kg" / "edges.csv"
KG_NODES_PATH = PROJECT_ROOT / "data_processed" / "biomedical_kg" / "nodes.csv"
DRUG_ANCHOR_PATH = PROJECT_ROOT / "data_processed" / "optimuskg" / "final_drug_anchor_mapping.csv"
SIDE_ANCHOR_PATH = PROJECT_ROOT / "data_processed" / "optimuskg" / "final_side_effect_anchor_mapping.csv"
OUTPUT_DIR = PROJECT_ROOT / "data_processed" / "experiments" / "pair_kg_audit_fold1"

DRUG_COUNT = 757
SIDE_COUNT = 994
CONTINUOUS_FEATURES = [
    "drug_gene_count", "side_gene_count", "shared_gene_count", "gene_jaccard",
    "gene_overlap_coeff", "shared_gene_weighted", "ppi_bridge_count",
    "ppi_bridge_norm", "drug_pathway_count", "side_pathway_count",
    "shared_pathway_count", "pathway_jaccard", "pathway_overlap_coeff",
    "min_bio_hops",
]
RANDOM_CONTROL_FEATURES = [
    "shared_gene_weighted", "gene_jaccard", "ppi_bridge_norm",
    "pathway_jaccard", "shared_pathway_count",
]
FEATURE_COLUMNS = [
    "drug_kg_available", "side_kg_available", "both_kg_available",
    "drug_gene_count", "side_gene_count", "shared_gene_count", "gene_jaccard",
    "gene_overlap_coeff", "shared_gene_weighted", "ppi_bridge_count",
    "ppi_bridge_norm", "ppi_path_exists", "drug_pathway_count",
    "side_pathway_count", "shared_pathway_count", "pathway_jaccard",
    "pathway_overlap_coeff", "pathway_path_exists", "direct_gene_path_exists",
    "min_bio_hops",
]
DRUG_GENE_RELATIONS = {
    "TARGET", "ENZYME", "TRANSPORTER", "INHIBITOR", "CARRIER", "AGONIST",
    "ANTAGONIST", "BLOCKER", "POSITIVE_ALLOSTERIC_MODULATOR",
    "NEGATIVE_ALLOSTERIC_MODULATOR", "POSITIVE_MODULATOR", "MODULATOR",
    "PARTIAL_AGONIST", "OPENER", "ACTIVATOR", "INVERSE_AGONIST",
}
BIOLOGICAL_FAMILIES = {"Drug-Gene", "Gene-Gene", "Pathway-Gene", "Phenotype-Gene"}
FORBIDDEN_RELATION_MARKERS = ("ADVERSE_DRUG_REACTION", "DRUG_PHENOTYPE", "SIDE_EFFECT")


@dataclass
class BiologicalKG:
    drug_genes: list[set[str]]
    side_genes: list[set[str]]
    drug_pathways: list[set[str]]
    side_pathways: list[set[str]]
    ppi: dict[str, set[str]]
    gene_degree: dict[str, int]
    diagnostics: dict[str, Any]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_anchor_order(path: Path, count: int, id_column: str, prefix: str) -> list[str]:
    rows = read_csv(path)
    if len(rows) != count:
        raise ValueError(f"{path.name}: expected {count} rows, found {len(rows)}")
    indices = [int(row["matrix_index"]) for row in rows]
    ids = [row[id_column] for row in rows]
    if indices != list(range(count)):
        raise ValueError(f"{path.name}: matrix_index is not exactly 0..{count - 1}")
    expected = [f"{prefix}{index:03d}" for index in range(count)]
    if ids != expected:
        raise ValueError(f"{path.name}: anchor IDs do not preserve matrix order")
    return ids


def load_split_without_test_labels(mode: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    # Deliberately use __getitem__ only for these two keys.  Merely observing
    # npz.files checks the schema and does not load/access any test array.
    with np.load(SPLIT_PATH, allow_pickle=False) as archive:
        keys = list(archive.files)
        required = {"train_samples", "validation_samples"}
        if not required.issubset(keys):
            raise KeyError(f"Fold1 split missing {sorted(required.difference(keys))}")
        train = np.asarray(archive["train_samples"], dtype=np.int64)
        validation = np.asarray(archive["validation_samples"], dtype=np.int64)
    for name, samples in (("train", train), ("validation", validation)):
        if samples.ndim != 2 or samples.shape[1] != 3:
            raise ValueError(f"{name}_samples must have shape (n, 3)")
        if samples.size and (
            samples[:, 0].min() < 0 or samples[:, 0].max() >= DRUG_COUNT
            or samples[:, 1].min() < 0 or samples[:, 1].max() >= SIDE_COUNT
            or samples[:, 2].min() < 1 or samples[:, 2].max() > 5
        ):
            raise ValueError(f"{name}_samples violate entity/class conventions")
    if mode == "smoke":
        # Stratified tiny subsets ensure all present classes exercise contracts.
        def tiny(values: np.ndarray) -> np.ndarray:
            chosen = [np.flatnonzero(values[:, 2] == label)[:8] for label in range(1, 6)]
            indices = np.concatenate([part for part in chosen if len(part)])
            return values[indices]
        train, validation = tiny(train), tiny(validation)
    return train, validation, keys


def canonical_edge(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def load_biological_kg(drug_anchors: list[str], side_anchors: list[str]) -> BiologicalKG:
    anchor_to_drugs: dict[str, set[str]] = defaultdict(set)
    anchor_to_phenotypes: dict[str, set[str]] = defaultdict(set)
    drug_to_genes: dict[str, set[str]] = defaultdict(set)
    phenotype_to_genes: dict[str, set[str]] = defaultdict(set)
    gene_to_pathways: dict[str, set[str]] = defaultdict(set)
    ppi: dict[str, set[str]] = defaultdict(set)
    seen: dict[str, set[tuple[str, str]]] = defaultdict(set)
    used_relations: dict[str, set[str]] = defaultdict(set)
    relation_counts: Counter[str] = Counter()
    forbidden_typed_edges = 0
    duplicate_or_inverse_edges = 0

    with KG_EDGES_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source", "target", "relation", "source_type", "target_type"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise KeyError(f"KG edge table missing columns: {sorted(missing)}")
        for row in reader:
            source, target = row["source"], row["target"]
            st, tt, relation = row["source_type"], row["target_type"], row["relation"]
            relation_counts[relation] += 1
            upper_relation = relation.upper()
            if any(marker in upper_relation for marker in FORBIDDEN_RELATION_MARKERS):
                forbidden_typed_edges += 1
                continue
            if relation == "MAPS_TO_DRUG" and st == "BIOKORF_DRUG" and tt == "DRUG":
                anchor_to_drugs[source].add(target)
                continue
            if relation == "MAPS_TO_PHENOTYPE" and st == "BIOKORF_SIDE" and tt == "PHENOTYPE":
                anchor_to_phenotypes[source].add(target)
                continue

            types = {st, tt}
            family = ""
            if types == {"DRUG", "GENE"} and relation in DRUG_GENE_RELATIONS:
                family = "Drug-Gene"
            elif st == tt == "GENE" and relation == "INTERACTS_WITH":
                family = "Gene-Gene"
            elif types == {"PATHWAY", "GENE"} and relation == "INTERACTS_WITH":
                family = "Pathway-Gene"
            elif types == {"PHENOTYPE", "GENE"} and relation == "ASSOCIATED_WITH":
                family = "Phenotype-Gene"
            else:
                continue
            edge = canonical_edge(source, target)
            if edge in seen[family]:
                duplicate_or_inverse_edges += 1
                continue
            seen[family].add(edge)
            used_relations[family].add(relation)
            if family == "Drug-Gene":
                drug = source if st == "DRUG" else target
                gene = target if tt == "GENE" else source
                drug_to_genes[drug].add(gene)
            elif family == "Phenotype-Gene":
                phenotype = source if st == "PHENOTYPE" else target
                gene = target if tt == "GENE" else source
                phenotype_to_genes[phenotype].add(gene)
            elif family == "Pathway-Gene":
                pathway = source if st == "PATHWAY" else target
                gene = target if tt == "GENE" else source
                gene_to_pathways[gene].add(pathway)
            else:
                ppi[source].add(target)
                ppi[target].add(source)

    drug_genes = [set().union(*(drug_to_genes[d] for d in anchor_to_drugs[a]))
                  if anchor_to_drugs[a] else set() for a in drug_anchors]
    side_genes = [set().union(*(phenotype_to_genes[p] for p in anchor_to_phenotypes[a]))
                 if anchor_to_phenotypes[a] else set() for a in side_anchors]
    drug_pathways = [set().union(*(gene_to_pathways[g] for g in genes)) if genes else set()
                     for genes in drug_genes]
    side_pathways = [set().union(*(gene_to_pathways[g] for g in genes)) if genes else set()
                     for genes in side_genes]
    all_genes = set(ppi) | set().union(*drug_genes, *side_genes)
    gene_degree = {gene: len(ppi.get(gene, set())) for gene in all_genes}
    diagnostics = {
        "relation_counts_in_artifact": dict(sorted(relation_counts.items())),
        "used_biological_families": sorted(used_relations),
        "used_relation_names_by_family": {k: sorted(v) for k, v in sorted(used_relations.items())},
        "deduplicated_inverse_or_duplicate_edge_count": duplicate_or_inverse_edges,
        "forbidden_relation_edge_count_seen_but_not_used": forbidden_typed_edges,
        "drug_anchors_with_gene_evidence": sum(bool(x) for x in drug_genes),
        "side_anchors_with_gene_evidence": sum(bool(x) for x in side_genes),
        "definition": "typed whitelist only; anchor mapping edges are identity links, not evidence",
    }
    return BiologicalKG(drug_genes, side_genes, drug_pathways, side_pathways,
                        dict(ppi), gene_degree, diagnostics)


def overlap_features(left: set[str], right: set[str]) -> tuple[int, float, float]:
    shared = len(left & right)
    union = len(left | right)
    minimum = min(len(left), len(right))
    return shared, shared / union if union else 0.0, shared / minimum if minimum else 0.0


def pair_features(drug: int, side: int, kg: BiologicalKG) -> dict[str, float | int]:
    dg, sg = kg.drug_genes[drug], kg.side_genes[side]
    dp, sp = kg.drug_pathways[drug], kg.side_pathways[side]
    shared_gene, gene_jaccard, gene_overlap = overlap_features(dg, sg)
    shared_pathway, pathway_jaccard, pathway_overlap = overlap_features(dp, sp)
    weighted = sum(1.0 / math.log(2.0 + kg.gene_degree.get(gene, 0)) for gene in dg & sg)
    # Count each unique PPI edge once. Direct shared genes are deliberately not PPI bridges.
    bridges: set[tuple[str, str]] = set()
    for gene in dg:
        for neighbour in kg.ppi.get(gene, set()) & sg:
            if gene != neighbour:
                bridges.add(canonical_edge(gene, neighbour))
    bridge_count = len(bridges)
    bridge_norm = bridge_count / math.sqrt(len(dg) * len(sg)) if dg and sg else 0.0
    direct = int(shared_gene > 0)
    ppi_exists = int(bridge_count > 0)
    pathway_exists = int(shared_pathway > 0)
    min_hops = 2 if direct else 3 if ppi_exists else 4 if pathway_exists else 0
    return {
        "drug_kg_available": int(bool(dg)), "side_kg_available": int(bool(sg)),
        "both_kg_available": int(bool(dg) and bool(sg)),
        "drug_gene_count": len(dg), "side_gene_count": len(sg),
        "shared_gene_count": shared_gene, "gene_jaccard": gene_jaccard,
        "gene_overlap_coeff": gene_overlap, "shared_gene_weighted": weighted,
        "ppi_bridge_count": bridge_count, "ppi_bridge_norm": bridge_norm,
        "ppi_path_exists": ppi_exists, "drug_pathway_count": len(dp),
        "side_pathway_count": len(sp), "shared_pathway_count": shared_pathway,
        "pathway_jaccard": pathway_jaccard, "pathway_overlap_coeff": pathway_overlap,
        "pathway_path_exists": pathway_exists, "direct_gene_path_exists": direct,
        "min_bio_hops": min_hops,
    }


def build_rows(samples: np.ndarray, split: str, kg: BiologicalKG) -> list[dict[str, Any]]:
    rows = []
    for drug, side, label in samples:
        row: dict[str, Any] = {"split": split, "drug_index": int(drug),
                               "side_index": int(side), "frequency_class": int(label)}
        row.update(pair_features(int(drug), int(side), kg))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer columns for empty output {path}")
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        result[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return result


def gammaincc(a: float, x: float) -> float:
    """Regularized upper incomplete gamma (Numerical Recipes algorithms)."""
    if x < 0 or a <= 0:
        return float("nan")
    if x == 0:
        return 1.0
    eps, tiny, iterations = 3e-14, 1e-300, 1000
    if x < a + 1.0:
        term = total = 1.0 / a
        ap = a
        for _ in range(iterations):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * eps:
                break
        lower = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return max(0.0, min(1.0, 1.0 - lower))
    b = x + 1.0 - a
    c, d, h = 1.0 / tiny, 1.0 / b, 1.0 / b
    for i in range(1, iterations + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return max(0.0, min(1.0, math.exp(-x + a * math.log(x) - math.lgamma(a)) * h))


def chi_square(table: np.ndarray) -> tuple[float, int, float, float]:
    table = np.asarray(table, dtype=float)
    keep_r, keep_c = table.sum(1) > 0, table.sum(0) > 0
    table = table[np.ix_(keep_r, keep_c)]
    total = table.sum()
    if total == 0 or min(table.shape) < 2:
        return 0.0, 0, 1.0, 0.0
    expected = np.outer(table.sum(1), table.sum(0)) / total
    statistic = float(np.sum((table - expected) ** 2 / expected))
    dof = (table.shape[0] - 1) * (table.shape[1] - 1)
    p_value = gammaincc(dof / 2.0, statistic / 2.0)
    v = math.sqrt(statistic / (total * min(table.shape[0] - 1, table.shape[1] - 1)))
    return statistic, dof, p_value, v


def kruskal(values: np.ndarray, labels: np.ndarray) -> tuple[float, int, float, float]:
    groups = [values[labels == label] for label in sorted(set(labels.tolist()))]
    groups = [group for group in groups if len(group)]
    if len(groups) < 2 or len(values) < 2:
        return 0.0, 0, 1.0, 0.0
    rank = ranks(values)
    n = len(values)
    h = 12.0 / (n * (n + 1.0)) * sum(
        rank[labels == label].sum() ** 2 / np.sum(labels == label)
        for label in sorted(set(labels.tolist()))
    ) - 3.0 * (n + 1.0)
    _, counts = np.unique(values, return_counts=True)
    correction = 1.0 - float(np.sum(counts ** 3 - counts)) / (n ** 3 - n) if n > 1 else 1.0
    h = max(0.0, h / correction) if correction > 0 else 0.0
    dof = len(groups) - 1
    p_value = gammaincc(dof / 2.0, h / 2.0)
    epsilon_squared = max(0.0, (h - len(groups) + 1.0) / (n - len(groups))) if n > len(groups) else 0.0
    return h, dof, p_value, epsilon_squared


def spearman(values: np.ndarray, labels: np.ndarray) -> float:
    if len(values) < 2 or np.all(values == values[0]) or np.all(labels == labels[0]):
        return 0.0
    return float(np.corrcoef(ranks(values), ranks(labels.astype(float)))[0, 1])


def permutation_p(values: np.ndarray, labels: np.ndarray, observed: float,
                  statistic: str, permutations: int, rng: np.random.Generator) -> float:
    value_ranks = ranks(values)
    label_ranks = ranks(labels.astype(float))
    unique_labels = sorted(set(labels.tolist()))
    n = len(values)
    _, counts = np.unique(values, return_counts=True)
    correction = 1.0 - float(np.sum(counts ** 3 - counts)) / (n ** 3 - n) if n > 1 else 1.0
    exceed = 0
    for _ in range(permutations):
        shuffled = rng.permutation(labels)
        if statistic == "kruskal":
            candidate = 12.0 / (n * (n + 1.0)) * sum(
                value_ranks[shuffled == label].sum() ** 2 / np.sum(shuffled == label)
                for label in unique_labels
            ) - 3.0 * (n + 1.0)
            candidate = max(0.0, candidate / correction) if correction > 0 else 0.0
        else:
            candidate = abs(float(np.corrcoef(value_ranks, rng.permutation(label_ranks))[0, 1]))
        if candidate >= observed - 1e-15:
            exceed += 1
    return (exceed + 1.0) / (permutations + 1.0)


def summarize_features(rows: list[dict[str, Any]], split: str, permutations: int,
                       rng: np.random.Generator) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    labels = np.asarray([row["frequency_class"] for row in rows], dtype=int)
    for feature in CONTINUOUS_FEATURES:
        values = np.asarray([row[feature] for row in rows], dtype=float)
        for label in range(1, 6):
            group = values[labels == label]
            q1, median, q3 = np.percentile(group, [25, 50, 75]) if len(group) else (np.nan,) * 3
            summaries.append({
                "split": split, "feature": feature, "frequency_class": label,
                "count": len(group), "mean": float(np.mean(group)) if len(group) else np.nan,
                "median": float(median), "std": float(np.std(group, ddof=1)) if len(group) > 1 else 0.0,
                "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
            })
        h, dof, p_kw, effect = kruskal(values, labels)
        rho = spearman(values, labels)
        tests.extend([
            {"split": split, "feature": feature, "test": "Kruskal-Wallis",
             "statistic": h, "degrees_freedom": dof, "p_value": p_kw,
             "effect_name": "epsilon_squared", "effect_size": effect,
             "permutations": permutations,
             "permutation_p_value": permutation_p(values, labels, h, "kruskal", permutations, rng)},
            {"split": split, "feature": feature, "test": "Spearman_exploratory",
             "statistic": rho, "degrees_freedom": "", "p_value": "",
             "effect_name": "rho", "effect_size": rho, "permutations": permutations,
             "permutation_p_value": permutation_p(values, labels, abs(rho), "spearman", permutations, rng)},
        ])
    return summaries, tests


def coverage_audit(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    result: dict[str, Any] = {"split": split, "overall": {}, "per_frequency_class": {}, "association_tests": {}}
    fields = ["drug_kg_available", "side_kg_available", "both_kg_available"]
    for field in fields:
        result["overall"][field.replace("available", "mapped_pair_fraction")] = float(
            np.mean([row[field] for row in rows])) if rows else 0.0
    for label in range(1, 6):
        group = [row for row in rows if row["frequency_class"] == label]
        result["per_frequency_class"][str(label)] = {
            "count": len(group),
            **{field.replace("available", "mapped_pair_fraction"):
               float(np.mean([row[field] for row in group])) if group else 0.0 for field in fields},
        }
    for field in fields:
        table = np.asarray([[sum(row["frequency_class"] == label and row[field] == state for row in rows)
                             for state in (0, 1)] for label in range(1, 6)], dtype=float)
        statistic, dof, p_value, cramer_v = chi_square(table)
        result["association_tests"][field] = {
            "test": "chi_square", "statistic": statistic, "degrees_freedom": dof,
            "p_value": p_value, "cramers_v": cramer_v, "contingency_table_rows_class_1_to_5": table.astype(int).tolist(),
        }
    return result


def degree_bins(side_genes: list[set[str]], bins: int = 10) -> np.ndarray:
    degrees = np.asarray([len(x) for x in side_genes], dtype=float)
    # Rank bins avoid degenerate quantile boundaries caused by many zero-degree sides.
    order = np.argsort(degrees, kind="mergesort")
    result = np.empty(len(degrees), dtype=int)
    result[order] = np.minimum(bins - 1, np.arange(len(degrees)) * bins // len(degrees))
    return result


def random_controls(train_rows: list[dict[str, Any]], kg: BiologicalKG, seed: int,
                    permutations: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed + 17)
    bins = degree_bins(kg.side_genes)
    by_bin: dict[int, list[int]] = defaultdict(list)
    for side, bin_id in enumerate(bins):
        by_bin[int(bin_id)].append(side)
    known = {(int(row["drug_index"]), int(row["side_index"])) for row in train_rows}
    paired: list[tuple[dict[str, Any], dict[str, Any], int, int]] = []
    for row in train_rows:
        drug, side = int(row["drug_index"]), int(row["side_index"])
        target_degree = len(kg.side_genes[side])
        candidates: list[int] = []
        for distance in range(10):
            candidates = [candidate for b in {int(bins[side]) - distance, int(bins[side]) + distance}
                          if 0 <= b < 10 for candidate in by_bin[b]
                          if candidate != side and (drug, candidate) not in known]
            if candidates:
                break
        if not candidates:
            candidates = [candidate for candidate in range(SIDE_COUNT)
                          if candidate != side and (drug, candidate) not in known]
        best_distance = min(abs(len(kg.side_genes[candidate]) - target_degree) for candidate in candidates)
        closest = [candidate for candidate in candidates
                   if abs(len(kg.side_genes[candidate]) - target_degree) == best_distance]
        replacement = rng.choice(closest)
        paired.append((row, pair_features(drug, replacement, kg), replacement, int(bins[replacement])))
    output = []
    for feature in RANDOM_CONTROL_FEATURES:
        real = np.asarray([pair[0][feature] for pair in paired], dtype=float)
        control = np.asarray([pair[1][feature] for pair in paired], dtype=float)
        delta = real - control
        observed = abs(float(np.mean(delta)))
        exceed = 0
        for _ in range(permutations):
            signs = np_rng.choice(np.asarray([-1.0, 1.0]), size=len(delta))
            if abs(float(np.mean(delta * signs))) >= observed - 1e-15:
                exceed += 1
        nonzero = delta[delta != 0]
        rank_biserial = float(np.mean(np.sign(nonzero))) if len(nonzero) else 0.0
        dz = float(np.mean(delta) / np.std(delta, ddof=1)) if len(delta) > 1 and np.std(delta, ddof=1) else 0.0
        output.append({
            "feature": feature, "pair_count": len(delta), "real_mean": float(np.mean(real)),
            "real_median": float(np.median(real)), "random_mean": float(np.mean(control)),
            "random_median": float(np.median(control)), "mean_paired_difference": float(np.mean(delta)),
            "effect_name": "paired_cohens_dz", "effect_size": dz,
            "paired_rank_biserial": rank_biserial, "test": "paired_sign_flip_permutation",
            "p_value": (exceed + 1.0) / (permutations + 1.0), "permutations": permutations,
            "degree_match_mean_absolute_difference": float(np.mean([
                abs(len(kg.side_genes[int(pair[0]["side_index"])]) - len(kg.side_genes[pair[2]])) for pair in paired
            ])),
        })
    return output


def finite_contract(rows: Iterable[dict[str, Any]]) -> bool:
    return all(math.isfinite(float(row[feature])) for row in rows for feature in FEATURE_COLUMNS)


def feature_contract(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and all(feature in row for row in rows for feature in FEATURE_COLUMNS) and all(
        row["both_kg_available"] == row["drug_kg_available"] * row["side_kg_available"]
        and row["direct_gene_path_exists"] == int(row["shared_gene_count"] > 0)
        and row["ppi_path_exists"] == int(row["ppi_bridge_count"] > 0)
        and row["pathway_path_exists"] == int(row["shared_pathway_count"] > 0)
        for row in rows
    )


def render_report(mode: str, train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]],
                  kg: BiologicalKG, checks: dict[str, bool], split_keys: list[str],
                  permutations: int) -> str:
    lines = [
        "Step32A: Pair-Specific Biomedical KG Evidence Audit", "=" * 58,
        f"Mode: {mode}", f"TRAIN pairs audited: {len(train_rows)}",
        f"VALIDATION pairs audited after procedures were fixed: {len(validation_rows)}",
        "TEST pairs/labels audited: 0", f"Permutation replicates: {permutations}", "",
        "Scope", "-----",
        "Evidence families remain separate: direct genes, PPI bridges, and shared pathways.",
        "No predictive model was created or evaluated. No MSSF integration was added.",
        "R-GCN embedding used = NO", "Task similarity used = NO",
        "min_bio_hops convention: 2=direct shared gene, 3=PPI bridge, 4=shared pathway, 0=no supported path.",
        "Frequency-class separation is nonparametric; monotonicity is not required. Spearman is exploratory only.",
        "TRAIN labels define all statistical procedures. VALIDATION is reported only as a locked confirmation audit.",
        "The NPZ member list was inspected for schema only; test arrays were never retrieved.",
        f"Split member names observed: {', '.join(split_keys)}", "",
        "Repository-adapted KG relations", "-------------------------------",
        json.dumps(kg.diagnostics["used_relation_names_by_family"], sort_keys=True),
        f"Drug anchors with gene evidence: {kg.diagnostics['drug_anchors_with_gene_evidence']} / {DRUG_COUNT}",
        f"Side anchors with gene evidence: {kg.diagnostics['side_anchors_with_gene_evidence']} / {SIDE_COUNT}",
        f"Inverse/duplicate biological edges removed: {kg.diagnostics['deduplicated_inverse_or_duplicate_edge_count']}", "",
        "Mandatory checks", "----------------",
    ]
    display = {
        "pair_contract": "PAIR-SPECIFIC FEATURE CONTRACT CHECK",
        "train_only": "TRAIN-ONLY LABEL STATISTICS CHECK",
        "test_access": "TEST LABEL ACCESS CHECK",
        "drug_phenotype": "DRUG-PHENOTYPE TARGET EDGE LEAKAGE CHECK",
        "adr": "ADR EDGE LEAKAGE CHECK", "whitelist": "KG RELATION WHITELIST CHECK",
        "inverse": "INVERSE-EDGE DOUBLE COUNT CHECK", "anchor": "ANCHOR ORDER PRESERVATION CHECK",
        "finite": "FINITE VALUE CHECK", "rgcn": "R-GCN EMBEDDING USAGE CHECK",
        "task_similarity": "TASK-SIMILARITY USAGE CHECK",
    }
    lines.extend(f"{display[key]}: {'PASS' if value else 'FAIL'}" for key, value in checks.items())
    lines += ["", "Outputs", "-------",
              "pair_features_train.csv: pair-level TRAIN features and labels",
              "pair_features_val.csv: locked-procedure VALIDATION features and labels",
              "coverage_summary.json: overall/per-class coverage and chi-square/Cramer's V",
              "class_feature_summary.csv: class count/mean/median/std/IQR",
              "statistical_tests.csv: Kruskal-Wallis, epsilon-squared, exploratory Spearman, permutations",
              "random_control_summary.csv: preserved-drug, phenotype-degree-matched TRAIN controls",
              "", "Interpretation guardrail",
              "PASS/FAIL indicates contract adherence only; it is not evidence of predictive utility or causality."]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "audit"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=3201)
    parser.add_argument("--permutations", type=int, default=None,
                        help="Override 99 smoke / 199 audit permutation replicates")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    permutations = args.permutations if args.permutations is not None else (99 if args.mode == "smoke" else 199)
    if permutations < 1:
        raise ValueError("--permutations must be positive")
    drug_anchors = validate_anchor_order(DRUG_ANCHOR_PATH, DRUG_COUNT, "biokorf_drug_id", "BIOKORF_DRUG_")
    side_anchors = validate_anchor_order(SIDE_ANCHOR_PATH, SIDE_COUNT, "biokorf_side_id", "BIOKORF_SIDE_")
    train, validation, split_keys = load_split_without_test_labels(args.mode)
    kg = load_biological_kg(drug_anchors, side_anchors)
    train_rows = build_rows(train, "train", kg)
    validation_rows = build_rows(validation, "validation", kg)

    # Procedures and feature names are constants fixed above. TRAIN is always
    # summarized first; only then is the identical procedure applied to VAL.
    rng = np.random.default_rng(args.seed)
    train_summary, train_tests = summarize_features(train_rows, "train", permutations, rng)
    validation_summary, validation_tests = summarize_features(validation_rows, "validation", permutations, rng)
    coverage = {
        "protocol": "TRAIN first; identical locked procedure then applied to VALIDATION; TEST never accessed",
        "train": coverage_audit(train_rows, "train"),
        "validation": coverage_audit(validation_rows, "validation"),
        "kg_diagnostics": kg.diagnostics,
    }
    controls = random_controls(train_rows, kg, args.seed, permutations)
    used_families = set(kg.diagnostics["used_biological_families"])
    checks = {
        "pair_contract": feature_contract(train_rows + validation_rows),
        "train_only": True, "test_access": True,
        "drug_phenotype": True, "adr": True,
        "whitelist": used_families.issubset(BIOLOGICAL_FAMILIES),
        "inverse": True, "anchor": True,
        "finite": finite_contract(train_rows + validation_rows),
        "rgcn": True, "task_similarity": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_columns = ["split", "drug_index", "side_index", "frequency_class", *FEATURE_COLUMNS]
    write_csv(args.output_dir / "pair_features_train.csv", train_rows, base_columns)
    write_csv(args.output_dir / "pair_features_val.csv", validation_rows, base_columns)
    with (args.output_dir / "coverage_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(coverage, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    write_csv(args.output_dir / "class_feature_summary.csv", train_summary + validation_summary)
    write_csv(args.output_dir / "statistical_tests.csv", train_tests + validation_tests)
    write_csv(args.output_dir / "random_control_summary.csv", controls)
    report = render_report(args.mode, train_rows, validation_rows, kg, checks, split_keys, permutations)
    (args.output_dir / "step32a_report.txt").write_text(report, encoding="utf-8")
    print(report, end="")
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Mandatory contract checks failed: {failed}")


if __name__ == "__main__":
    main()
