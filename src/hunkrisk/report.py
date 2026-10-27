"""Score every system against the gold test set and write the results.

    uv run python -m hunkrisk.report

Writes reports/results.md and the results section of README.md. Local models are scored with a
P(high) threshold tuned on validation to reach the release bar's recall, then applied unchanged
to test. The threshold is tuned against the panel's validation labels, made the same way as the
test labels; the first runs used teacher labels, and both are reported. Results are marked
provisional while any panel disagreement is still waiting for a human label.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hunkrisk.metrics import category_f1, risk_metrics
from hunkrisk.panel import DISAGREEMENTS_PATH, load_gold
from hunkrisk.schema import InvalidOutput, parse_label

GPU_USD_PER_HOUR = 0.40  # assumed rental price of a comparable GPU (see release bar)
TARGET_RECALL = 0.90
BOOTSTRAP = 1000
README = Path("README.md")
RESULTS = Path("reports/results.md")
VAL_REFERENCES = {
    "panel": Path("data/labels/val-panel/panel.jsonl"),
    "teacher": Path("data/labels/teacher/val.jsonl"),
}
THRESHOLD_ON = "panel"  # the first runs used "teacher"; see docs/decisions.md, 2026-10-27


@dataclass(frozen=True)
class System:
    name: str
    kind: str  # rules | local | api
    path: str  # predictions folder (rules, local) or test label file (api)
    note: str = ""


SYSTEMS = [
    System("Keyword/path heuristic", "rules", "data/predictions/heuristic"),
    System("Qwen3-1.7B, zero-shot", "local", "data/predictions/qwen3-1.7b-zeroshot"),
    System("Qwen3.5-2B, zero-shot", "local", "data/predictions/qwen3.5-2b-zeroshot"),
    System("**Qwen3-1.7B, fine-tuned**", "local", "data/predictions/ft-qwen3-1.7b"),
    System("**Qwen3.5-2B, fine-tuned**", "local", "data/predictions/ft-qwen3.5-2b"),
    System("**Qwen3.5-2B, fine-tuned, 2× data**", "local", "data/predictions/ft-qwen3.5-2b-x2"),
    System("Jev 1.14 (decision API)", "api", "data/labels/test/jev-1.14.jsonl"),
    System("DeepSeek V4.1 Flash (teacher)", "api", "data/labels/test/deepseek-v4.1-flash.jsonl"),
    System("Kimi K3.1 Pro (ceiling)", "api", "data/labels/test/kimi-k3.1-pro.jsonl"),
    System("Claude Sonnet 5", "api", "data/labels/test/claude-sonnet-5.jsonl", "panel"),
    System("GPT-5.6 Terra", "api", "data/labels/test/gpt-5.6-terra.jsonl", "panel"),
]


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def tune_threshold(val: dict[str, dict[str, Any]], reference: dict[str, dict[str, Any]]) -> float:
    """Highest P(high) threshold that still reaches the target recall on validation.

    Only hunks with a reference label count; the panel reference leaves out its disagreements.
    """
    positives = sorted(
        (val[i]["p_high"] for i in reference if reference[i]["label"]["risk"] == "high"),
        reverse=True,
    )
    needed = int(np.ceil(TARGET_RECALL * len(positives)))
    return positives[needed - 1]


def as_generated(row: dict[str, Any]) -> dict[str, Any]:
    """The model's own answer, leniently parsed so a code-fenced reply still counts here."""
    try:
        return parse_label(row["raw"], strict=False).to_json()
    except InvalidOutput:
        return {"risk": "low", "categories": []}


def thresholded(
    rows: dict[str, dict[str, Any]], generated: dict[str, dict[str, Any]], threshold: float
) -> dict[str, dict[str, Any]]:
    """Risk from P(high) at the threshold; categories from the model's own answer."""
    labels = {}
    for i, r in rows.items():
        high = r["p_high"] >= threshold
        categories = generated[i]["categories"] if generated[i]["risk"] == "high" else []
        labels[i] = {
            "risk": "high" if high else "low",
            "categories": categories if high else ["none"],
        }
    return labels


def decisions(system: System) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Predicted labels per hunk id plus facts for the table (validity, cost, latency)."""
    if system.kind == "api":
        rows = read_jsonl(Path(system.path))
        costs = [r["cost"] for r in rows.values()]
        info = {"valid": None, "usd_per_1k": 1000 * float(np.mean(costs)), "latency": None}
        return {i: r["label"] or {"risk": "low", "categories": []} for i, r in rows.items()}, info

    folder = Path(system.path)
    rows = read_jsonl(folder / "test.jsonl")
    if system.kind == "rules":
        info = {"valid": 1.0, "usd_per_1k": 0.0, "latency": None}
        return {i: r["label"] for i, r in rows.items()}, info

    val = read_jsonl(folder / "val.jsonl")
    generated = {i: as_generated(r) for i, r in rows.items()}
    protocols = {}
    for reference, path in VAL_REFERENCES.items():
        threshold = tune_threshold(val, read_jsonl(path))
        protocols[reference] = (threshold, thresholded(rows, generated, threshold))
    threshold, labels = protocols[THRESHOLD_ON]
    latency = float(np.mean([r["latency_s"] for r in rows.values()]))
    summary = json.loads((folder / "test.summary.json").read_text())
    gpu_seconds = summary.get("throughput_s_per_hunk", latency)  # batched when measured
    info = {
        "valid": sum(r["label"] is not None for r in rows.values()) / len(rows),
        "usd_per_1k": 1000 * gpu_seconds / 3600 * GPU_USD_PER_HOUR,
        "latency": latency,
        "threshold": threshold,
        "protocols": protocols,
        "as_generated": generated,
    }
    return labels, info


def bootstrap_ci(ref: list[str], pred: list[str], key: str, seed: int = 13) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    ref_a, pred_a = np.array(ref), np.array(pred)
    values = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, len(ref), len(ref))
        values.append(risk_metrics(ref_a[idx].tolist(), pred_a[idx].tolist())[key])
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def score(gold: dict[str, Any], labels: dict[str, Any], strata: dict[str, str]) -> dict[str, Any]:
    ids = sorted(gold)
    ref = [gold[i]["risk"] for i in ids]
    pred = [labels[i]["risk"] for i in ids]
    m = risk_metrics(ref, pred)
    m["recall_ci"] = bootstrap_ci(ref, pred, "recall")
    m["f1_ci"] = bootstrap_ci(ref, pred, "f1")
    # Precision at the natural base rate, estimated from the randomly sampled hunks.
    random_ids = [i for i in ids if strata[i] == "random"]
    prevalence = sum(gold[i]["risk"] == "high" for i in random_ids) / len(random_ids)
    negatives = [i for i in ids if gold[i]["risk"] == "low"]
    fpr = sum(labels[i]["risk"] == "high" for i in negatives) / len(negatives)
    tp_rate = m["recall"] * prevalence
    m["precision_natural"] = tp_rate / (tp_rate + fpr * (1 - prevalence)) if tp_rate else 0.0
    m["categories"] = category_f1(
        [gold[i]["categories"] for i in ids], [labels[i]["categories"] for i in ids]
    )
    m["recall_by_stratum"] = {
        s: risk_metrics(
            [gold[i]["risk"] for i in ids if strata[i] == s],
            [labels[i]["risk"] for i in ids if strata[i] == s],
        )["recall"]
        for s in ("targeted", "random")
    }
    return m


def fmt_cost(usd: float) -> str:
    return "$0" if usd == 0 else f"${usd:.3f}" if usd < 1 else f"${usd:.2f}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def main_table(results: list[tuple[System, dict[str, Any], dict[str, Any]]]) -> str:
    rows = []
    for system, info, m in results:
        recall_ci, f1_ci = m["recall_ci"], m["f1_ci"]
        rows.append([
            system.name + (" †" if system.note == "panel" else ""),
            f"{m['recall']:.2f} ({recall_ci[0]:.2f}–{recall_ci[1]:.2f})",
            f"{m['precision']:.2f}",
            f"{m['precision_natural']:.2f}",
            f"{m['f1']:.2f} ({f1_ci[0]:.2f}–{f1_ci[1]:.2f})",
            "n/a" if info["valid"] is None else f"{info['valid']:.1%}",
            fmt_cost(info["usd_per_1k"]),
            f"{info['latency']:.2f} s" if info.get("latency") else "n/a",
        ])  # fmt: skip
    header = ["System", "Recall (95% CI)", "Precision", "Precision at base rate",
              "F1 (95% CI)", "Valid JSON", "Cost / 1k hunks", "Latency (GPU)"]  # fmt: skip
    return md_table(header, rows)


REFERENCE_NAMES = {"panel": "panel-labeled validation", "teacher": "teacher-labeled validation"}


def release_bar(results: list[tuple[System, dict[str, Any], dict[str, Any]]]) -> str:
    """Every fine-tuned model against the bar, under each threshold reference."""
    ceiling_m, ceiling_info = next((m, i) for s, i, m in results if "(ceiling)" in s.name)
    floor = ceiling_m["f1"] - 0.10
    rows = []
    for system, info, _ in results:
        if system.kind != "local" or "fine-tuned" not in system.name:
            continue
        cheaper = info["usd_per_1k"] * 10 <= ceiling_info["usd_per_1k"]
        for reference, m in info["protocol_scores"].items():
            checks = [
                (m["recall"] >= TARGET_RECALL, f"{m['recall']:.2f}"),
                (info["valid"] >= 0.98, f"{info['valid']:.1%}"),
                (cheaper and m["f1"] >= floor, f"F1 {m['f1']:.2f} vs ≥ {floor:.2f}"),
            ]
            marks = [f"{'yes' if ok else 'no'} ({value})" for ok, value in checks]
            passed = all(ok for ok, _ in checks)
            rows.append(
                [
                    system.name,
                    REFERENCE_NAMES[reference],
                    *marks,
                    f"**{'yes' if passed else 'no'}**",
                ]
            )
    header = ["System", "Threshold tuned on", "Recall ≥ 0.90", "Valid ≥ 98%",
              "≥ 10× cheaper at ≤ 10 pts lower F1 than the ceiling", "Pass"]  # fmt: skip
    return md_table(header, rows)


def details(results: list[tuple[System, dict[str, Any], dict[str, Any]]], gold: dict[str, Any],
            strata: dict[str, str]) -> str:  # fmt: skip
    categories = list(results[0][2]["categories"])
    per_category = md_table(["System", *categories], [
        [s.name, *("n/a" if v is None else f"{v:.2f}" for v in m["categories"].values())]
        for s, _, m in results
    ])  # fmt: skip
    by_stratum = md_table(["System", "Targeted", "Random"], [
        [s.name, f"{m['recall_by_stratum']['targeted']:.2f}",
         f"{m['recall_by_stratum']['random']:.2f}"]
        for s, _, m in results
    ])  # fmt: skip
    local = [(s, i) for s, i, _ in results if s.kind == "local"]
    header = ["System", "Threshold tuned on", "Threshold", "Recall", "Precision", "F1"]
    protocols = md_table(header, [
        [s.name, REFERENCE_NAMES[ref], f"{i['protocols'][ref][0]:.3f}", f"{m['recall']:.2f}",
         f"{m['precision']:.2f}", f"{m['f1']:.2f}"]
        for s, i in local for ref, m in i["protocol_scores"].items()
    ])  # fmt: skip
    plain = []
    for s, i in local:
        m = score(gold, i["as_generated"], strata)
        plain.append([s.name, f"{m['recall']:.2f}", f"{m['precision']:.2f}", f"{m['f1']:.2f}",
                      f"{i['valid']:.1%}"])  # fmt: skip
    plain_table = md_table(["System", "Recall", "Precision", "F1", "Valid JSON (strict)"], plain)
    return (
        f"## Per-category F1\n\n{per_category}\n\n"
        "## Recall by sampling stratum\n\n"
        "Targeted hunks come from risk-looking paths, which the heuristic also keys on. Random "
        f"hunks are where it has no head start.\n\n{by_stratum}\n\n"
        "## Local models: threshold reference\n\n"
        "Each threshold is the highest one that reaches 0.90 recall on validation. The first runs "
        "tuned it against the teacher's validation labels; it is now tuned against the panel's, "
        "which are made the same way as the test labels (see "
        f"[decisions](../docs/decisions.md)).\n\n{protocols}\n\n"
        "## Local models: plain decoding\n\n"
        "The model's own `risk` field with no threshold, parsed leniently (a code fence is "
        f"tolerated); the valid column is strict, as in the release bar.\n\n{plain_table}\n\n"
        f"## Release bar\n\n{release_bar(results)}\n"
    )


def main() -> None:
    gold = load_gold()
    pending = [i for i in DISAGREEMENTS_PATH.read_text().split() if i not in gold]
    hunks = read_jsonl(Path("data/hunks/test.jsonl"))
    strata = {i: hunks[i]["stratum"] for i in hunks}
    gold = {i: g for i, g in gold.items() if i in hunks}
    n_high = sum(g["risk"] == "high" for g in gold.values())

    results = []
    for system in SYSTEMS:
        target = Path(system.path)
        if not (target if system.kind == "api" else target / "test.jsonl").exists():
            continue
        labels, info = decisions(system)
        if system.kind == "local":
            info["protocol_scores"] = {
                ref: score(gold, ref_labels, strata)
                for ref, (_, ref_labels) in info["protocols"].items()
            }
        results.append((system, info, score(gold, labels, strata)))

    header = (
        f"Scored on {len(gold)} test hunks from 5 held-out repositories ({n_high} `high`). "
        "Local models use a P(high) threshold tuned on panel-labeled validation; API models are "
        "scored as returned. † Panel member: it helped write the answer key, so its scores are "
        "inflated."
    )
    if pending:
        header = (
            f"**Provisional:** {len(pending)} hunks where the two judges disagreed still need a "
            "human label and are left out. " + header
        )
    table = main_table(results)

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(f"# Results\n\n{header}\n\n{table}\n\n{details(results, gold, strata)}")
    block = (
        f"<!-- results:start -->\n{header}\n\n{table}\n\n"
        "Full breakdown: [reports/results.md](reports/results.md).\n<!-- results:end -->"
    )
    pattern = re.compile(r"<!-- results:start -->.*<!-- results:end -->", re.DOTALL)
    README.write_text(pattern.sub(lambda _: block, README.read_text()))
    print(f"{header}\n\n{table}")


if __name__ == "__main__":
    main()
