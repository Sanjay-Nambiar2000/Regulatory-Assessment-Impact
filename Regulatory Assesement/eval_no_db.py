#!/usr/bin/env python3
# eval_no_db.py — Generate RESULTS (tables + charts) with ZERO DB/Session Manager.
# Inputs (provide ONE of):
#   1) data/results.jsonl         — one JSON object per line
#   2) data/performance_log.txt   — text lines like:
#        - 2060 ms | status=Compliant | cites=3 | conflicts=0 | review=False | hash=... | question...
# Optional:
#   gold_labels.csv — columns: query_hash,true_label

import json, re
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(".")
DATA = ROOT / "data"
OUT = ROOT / "Results"
OUT.mkdir(parents=True, exist_ok=True)

JSONL = DATA / "results.jsonl"
PERF  = DATA / "performance_log.txt"
GOLD  = ROOT / "gold_labels.csv"

def load_from_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append({
                "timestamp": obj.get("timestamp"),
                "query_text": obj.get("query_text") or obj.get("question") or "",
                "query_hash": obj.get("query_hash") or obj.get("hash") or "",
                "compliance_verdict": obj.get("status") or obj.get("verdict") or obj.get("compliance_verdict") or "",
                "cites": int(obj.get("cites") or obj.get("citations_count") or obj.get("citations") or 0),
                "conflicts": int(obj.get("conflicts") or obj.get("conflicts_detected") or 0),
                "response_time_ms": float(obj.get("response_time_ms") or obj.get("latency_ms") or 0),
                "review_required": bool(obj.get("review") or obj.get("review_required") or False),
                "confidence_score": obj.get("confidence_score"),
            })
    return pd.DataFrame(rows)

def load_from_perf_text(path: Path) -> pd.DataFrame:
    # Example:
    # - 2060 ms | status=Compliant | cites=3 | conflicts=0 | review=False | hash=812e... | Question...
    pat = re.compile(
        r"-\s*(?P<ms>\d+)\s*ms\s*\|\s*status=(?P<status>\w+)\s*\|\s*cites=(?P<cites>\d+)"
        r"\s*\|\s*conflicts=(?P<conflicts>\d+)\s*\|\s*review=(?P<review>\w+)"
        r"\s*\|\s*hash=(?P<hash>[a-fA-F0-9]+)\s*\|\s*(?P<question>.+)"
    )
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            d = m.groupdict()
            rows.append({
                "timestamp": None,
                "query_text": d["question"].strip(),
                "query_hash": d["hash"],
                "compliance_verdict": d["status"],
                "cites": int(d["cites"]),
                "conflicts": int(d["conflicts"]),
                "response_time_ms": float(d["ms"]),
                "review_required": d["review"].lower() == "true",
                "confidence_score": None,
            })
    return pd.DataFrame(rows)

def load_logs() -> pd.DataFrame:
    if JSONL.exists():
        df = load_from_jsonl(JSONL)
        if not df.empty:
            return df
    if PERF.exists():
        df = load_from_perf_text(PERF)
        if not df.empty:
            return df
    # If no inputs, create samples and return empty df.
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "results.jsonl.sample").open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query_text": "According to the Australian Code of GMP (2013), what must be recorded?",
            "query_hash": "812e3081f62792c6f9e642bcb6c3b2fb7048597ff9d56e4dee4e06000f22ee5e",
            "status": "Compliant", "cites": 3, "conflicts": 0,
            "response_time_ms": 2060, "review": False
        }) + "\n")
    with (DATA / "performance_log.sample.txt").open("w", encoding="utf-8") as f:
        f.write("- 2060 ms | status=Compliant | cites=3 | conflicts=0 | review=False | hash=812e3081f6 | According to the Australian Code of GMP (2013), what must be recorded\n")
    return pd.DataFrame()

def save_headline_summary(df: pd.DataFrame):
    if df.empty:
        pd.DataFrame([{
            "n_requests": 0,
            "p50_ms": None, "p90_ms": None, "p99_ms": None,
            "avg_citations_per_answer": None,
            "conflicts_detected_total": 0
        }]).to_csv(OUT / "headline_summary.csv", index=False)
        return
    n = len(df)
    p50 = int(df["response_time_ms"].quantile(0.50))
    p90 = int(df["response_time_ms"].quantile(0.90))
    p99 = int(df["response_time_ms"].quantile(0.99))
    verdict_counts = df["compliance_verdict"].value_counts().to_dict()
    cites_avg = float(df["cites"].mean())
    conflicts_total = int(df["conflicts"].sum())
    summary = {
        "n_requests": n,
        "p50_ms": p50, "p90_ms": p90, "p99_ms": p99,
        "avg_citations_per_answer": round(cites_avg, 2),
        "conflicts_detected_total": conflicts_total,
        **{f"verdict_{k}": v for k, v in verdict_counts.items()}
    }
    pd.DataFrame([summary]).to_csv(OUT / "headline_summary.csv", index=False)

def plot_latency(df: pd.DataFrame):
    if df.empty: return
    plt.figure()
    df["response_time_ms"].plot(kind="hist", bins=20, edgecolor="black")
    plt.title("Latency Distribution (ms)")
    plt.xlabel("response_time_ms"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(OUT/"latency_hist.png"); plt.close()

def plot_verdicts(df: pd.DataFrame):
    if df.empty: return
    vc = df["compliance_verdict"].value_counts().rename_axis("verdict").reset_index(name="count")
    vc.to_csv(OUT/"verdict_counts.csv", index=False)
    plt.figure()
    plt.bar(vc["verdict"], vc["count"])
    plt.title("Compliance Verdicts")
    plt.xlabel("verdict"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(OUT/"verdict_counts.png"); plt.close()

def plot_confidence(df: pd.DataFrame):
    if "confidence_score" in df.columns and df["confidence_score"].notna().any():
        plt.figure()
        df["confidence_score"].dropna().astype(float).plot(kind="hist", bins=20, edgecolor="black")
        plt.title("Confidence Score Distribution")
        plt.xlabel("confidence_score"); plt.ylabel("count")
        plt.tight_layout(); plt.savefig(OUT/"confidence_hist.png"); plt.close()

def compute_ml_metrics(df: pd.DataFrame, gold_path: Path):
    if not gold_path.exists() or df.empty:
        return None
    gold = pd.read_csv(gold_path)
    if not {"query_hash","true_label"}.issubset(set(gold.columns)):
        print("gold_labels.csv must have columns: query_hash,true_label")
        return None
    merged = df.merge(gold, on="query_hash", how="inner")
    if merged.empty:
        print("No overlap between logs and gold labels.")
        return None

    y_true = merged["true_label"].astype(str).tolist()
    y_pred = merged["compliance_verdict"].astype(str).tolist()
    labels = sorted(set(y_true) | set(y_pred))
    label_to_idx = {l:i for i,l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t,p in zip(y_true,y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1

    per_class = []
    total = cm.sum()
    accuracy = (np.trace(cm) / total) if total > 0 else 0.0
    for i,l in enumerate(labels):
        tp = cm[i,i]
        fp = cm[:,i].sum() - tp
        fn = cm[i,:].sum() - tp
        prec = tp / (tp + fp) if (tp+fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp+fn) > 0 else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
        per_class.append({"label": l, "precision": prec, "recall": rec, "f1": f1, "support": int(cm[i,:].sum())})

    macro_p = np.mean([x["precision"] for x in per_class]) if per_class else 0.0
    macro_r = np.mean([x["recall"] for x in per_class]) if per_class else 0.0
    macro_f1 = np.mean([x["f1"] for x in per_class]) if per_class else 0.0

    pd.DataFrame(per_class).to_csv(OUT/"ml_per_class_metrics.csv", index=False)
    pd.DataFrame([{
        "accuracy": accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "n_evaluated": int(total)
    }]).to_csv(OUT/"ml_overall_metrics.csv", index=False)

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (verdict)")
    plt.xlabel("pred"); plt.ylabel("true")
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45, ha="right")
    plt.yticks(ticks=range(len(labels)), labels=labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center")
    plt.tight_layout(); plt.savefig(OUT/"ml_confusion_matrix.png"); plt.close()

    return {
        "accuracy": accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
    }

def write_readme(df: pd.DataFrame, ml_summary):
    lines = []
    lines.append("# RESULTS, EVALUATION & ANALYSIS (No DB / No Session Manager)\n")
    if df.empty:
        lines.append("**No inputs found.** Create either `data/results.jsonl` or `data/performance_log.txt` and re-run.")
    else:
        lines.append(f"- Requests evaluated: **{len(df)}**")
        lines.append(f"- Latency: **P50={int(df['response_time_ms'].quantile(0.5))} ms**, "
                     f"**P90={int(df['response_time_ms'].quantile(0.9))} ms**, "
                     f"**P99={int(df['response_time_ms'].quantile(0.99))} ms**")
        vc = df['compliance_verdict'].value_counts().to_dict()
        lines.append(f"- Verdict counts: `{vc}`")
        lines.append(f"- Avg citations per answer: **{df['cites'].mean():.2f}**")
        lines.append(f"- Total conflicts detected: **{int(df['conflicts'].sum())}**")
    lines.append("\n## Figures")
    lines.append("- Latency histogram: `Results/latency_hist.png`")
    lines.append("- Verdict distribution: `Results/verdict_counts.png`")
    if (OUT/"confidence_hist.png").exists():
        lines.append("- Confidence distribution: `Results/confidence_hist.png`")
    if (OUT/"ml_confusion_matrix.png").exists():
        lines.append("- Confusion matrix: `Results/ml_confusion_matrix.png`")
    lines.append("\n## Tables")
    lines.append("- Headline summary: `Results/headline_summary.csv`")
    if (OUT/"verdict_counts.csv").exists():
        lines.append("- Verdict counts: `Results/verdict_counts.csv`")
    if (OUT/"ml_per_class_metrics.csv").exists():
        lines.append("- Per-class ML metrics: `Results/ml_per_class_metrics.csv`")
    if (OUT/"ml_overall_metrics.csv").exists():
        lines.append("- Overall ML metrics: `Results/ml_overall_metrics.csv`")
    (OUT/"README_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

def main():
    df = load_logs()
    df.to_csv(OUT/"raw_export_no_db.csv", index=False)
    save_headline_summary(df)
    plot_latency(df)
    plot_verdicts(df)
    plot_confidence(df)
    ml_summary = compute_ml_metrics(df, GOLD)
    write_readme(df, ml_summary)
    print("✅ Done. See the 'Results/' folder for charts and tables.")
    if df.empty:
        print("Samples created: data/results.jsonl.sample and data/performance_log.sample.txt")
        print("Fill one of them and rename to results.jsonl or performance_log.txt, then re-run.")

if __name__ == "__main__":
    main()
