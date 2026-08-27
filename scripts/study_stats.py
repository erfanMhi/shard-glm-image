#!/usr/bin/env python3
"""Markdown statistics report for the WAN-PP throughput study.

Reads one or more results JSONL files, one record per (config, prompt, rep),
and prints a Markdown report with five sections:

  1. per-configuration throughput: median tok/s with a seeded bootstrap CI,
     mean, IQR, min/max, mean accept, stale fraction, per-category medians
  2. paired comparisons of every configuration against a baseline: median
     paired difference with bootstrap CI, fraction of prompts faster, and a
     Wilcoxon signed-rank p-value
  3. acceptance by category: mean_accept and accepted / (valid * K)
  4. output correctness between configurations: output_sha agreement
  5. jitter: within-prompt coefficient of variation of tok/s across reps

Record fields, all tolerated when missing:
  config, depth, K, compile, id, cat, rep, ntok, seconds, tok_s,
  valid, stale, accepted, mean_accept, output_sha (output_sha_eos preferred when present)

Only the standard library is required. numpy (bootstrap speed) and scipy
(Wilcoxon test) are used when importable; --no-numpy and --no-scipy force
the pure-Python paths.

Examples:
  python3 study_stats.py runs/*.jsonl --baseline d6_k2 --out report.md --csv summary.csv
  python3 study_stats.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
import zlib

try:
    import numpy as np
except ImportError:  # numpy is optional
    np = None

try:
    from scipy import stats as scipy_stats
except ImportError:  # scipy is optional
    scipy_stats = None

NAN = float("nan")
CONFIG_RE = re.compile(r"d(\d+)_k(\d+)")


# ---------------------------------------------------------------------------
# Loading and normalisation
# ---------------------------------------------------------------------------

def to_float(x):
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return None if math.isnan(v) else v
    if isinstance(x, str):
        try:
            v = float(x)
        except ValueError:
            return None
        return None if math.isnan(v) else v
    return None


def to_int(x):
    v = to_float(x)
    return None if v is None else int(round(v))


def to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
    return None


def normalize(raw, fallback_config, lineno):
    """Turn one parsed JSON object into a typed record, or None if unusable."""
    if not isinstance(raw, dict):
        return None
    cfg = raw.get("config")
    cfg = str(cfg) if cfg not in (None, "") else str(fallback_config)
    depth = to_int(raw.get("depth"))
    k = to_int(raw.get("K", raw.get("k")))
    m = CONFIG_RE.search(cfg)
    if m:
        if depth is None:
            depth = int(m.group(1))
        if k is None:
            k = int(m.group(2))
    pid = raw.get("id")
    pid = str(pid) if pid not in (None, "") else f"anon-{lineno}"
    cat = raw.get("cat")
    cat = str(cat) if cat not in (None, "") else "unknown"
    rep = to_int(raw.get("rep"))
    rep = 0 if rep is None else rep
    tok_s = to_float(raw.get("tok_s"))
    ntok = to_float(raw.get("ntok"))
    seconds = to_float(raw.get("seconds"))
    if tok_s is None and ntok is not None and seconds is not None and seconds > 0:
        tok_s = ntok / seconds
    valid = to_float(raw.get("valid"))
    stale = to_float(raw.get("stale"))
    accepted = to_float(raw.get("accepted"))
    mean_accept = to_float(raw.get("mean_accept"))
    if mean_accept is None and accepted is not None and valid is not None and valid > 0:
        mean_accept = accepted / valid
    sha = raw.get("output_sha_eos") or raw.get("output_sha")   # prefer the EOS-truncated, max_new-capped hash (chunk-aligned stop rule)
    sha = str(sha) if sha not in (None, "") else None
    return {
        "config": cfg, "depth": depth, "K": k, "compile": to_bool(raw.get("compile")),
        "id": pid, "cat": cat, "rep": rep, "tok_s": tok_s, "ntok": ntok,
        "seconds": seconds, "valid": valid, "stale": stale, "accepted": accepted,
        "mean_accept": mean_accept, "output_sha": sha,
    }


def parse_lines(lines, fallback_config):
    records = []
    n_bad = 0
    for lineno, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            n_bad += 1
            continue
        rec = normalize(raw, fallback_config, lineno)
        if rec is None:
            n_bad += 1
            continue
        records.append(rec)
    return records, n_bad


def load_files(paths):
    """Return (records, [(path, n_ok, n_bad), ...])."""
    records = []
    info = []
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as fh:
            recs, n_bad = parse_lines(fh, stem)
        records.extend(recs)
        info.append((path, len(recs), n_bad))
    return records, info


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def percentile_sorted(s, q):
    """Linear-interpolation percentile of an already sorted list (numpy default)."""
    n = len(s)
    if n == 0:
        return NAN
    pos = (n - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def describe(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return {"n": 0, "median": None, "mean": None, "q1": None, "q3": None, "min": None, "max": None}
    return {
        "n": n,
        "median": percentile_sorted(s, 50),
        "mean": sum(s) / n,
        "q1": percentile_sorted(s, 25),
        "q3": percentile_sorted(s, 75),
        "min": float(s[0]),
        "max": float(s[-1]),
    }


def seed_for(seed, label):
    """Stable per-label seed so results do not depend on config order."""
    return (int(seed) + zlib.crc32(label.encode("utf-8"))) % (2 ** 32)


def bootstrap_median_ci(values, n_boot, seed, use_numpy=True, alpha=0.05):
    """Percentile bootstrap CI of the median. Returns (lo, hi)."""
    n = len(values)
    if n == 0:
        return (None, None)
    if n == 1:
        return (float(values[0]), float(values[0]))
    lo_q = 100.0 * alpha / 2.0
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    if use_numpy and np is not None:
        rng = np.random.default_rng(seed)
        arr = np.asarray(values, dtype=float)
        meds = np.empty(n_boot, dtype=float)
        chunk = max(1, min(n_boot, 2_000_000 // n))
        done = 0
        while done < n_boot:
            m = min(chunk, n_boot - done)
            idx = rng.integers(0, n, size=(m, n))
            meds[done:done + m] = np.median(arr[idx], axis=1)
            done += m
        return (float(np.percentile(meds, lo_q)), float(np.percentile(meds, hi_q)))
    rng = random.Random(seed)
    vals = [float(v) for v in values]
    half = n // 2
    odd = (n % 2 == 1)
    meds = []
    choices = rng.choices
    for _ in range(n_boot):
        s = sorted(choices(vals, k=n))
        meds.append(s[half] if odd else 0.5 * (s[half - 1] + s[half]))
    meds.sort()
    return (percentile_sorted(meds, lo_q), percentile_sorted(meds, hi_q))


def wilcoxon_normal(diffs):
    """Two-sided Wilcoxon signed-rank p-value by normal approximation.

    Zero differences are dropped (Wilcoxon's original rule), ties in |d| get
    average ranks with the usual variance correction, and a 0.5 continuity
    correction is applied.
    """
    d = [float(x) for x in diffs if x != 0.0]
    n = len(d)
    if n == 0:
        return None
    absd = [abs(x) for x in d]
    order = sorted(range(n), key=lambda i: absd[i])
    ranks = [0.0] * n
    tie_sum = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and absd[order[j + 1]] == absd[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        t = j - i + 1
        if t > 1:
            tie_sum += t ** 3 - t
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    w_minus = sum(r for r, x in zip(ranks, d) if x < 0)
    t_stat = min(w_plus, w_minus)
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_sum / 48.0
    if var <= 0:
        return 1.0
    z = (t_stat - mean + 0.5) / math.sqrt(var)
    if z > 0:
        z = 0.0
    p = 2.0 * 0.5 * math.erfc(-z / math.sqrt(2.0))
    return min(1.0, p)


def wilcoxon_scipy(diffs):
    d = [float(x) for x in diffs if x != 0.0]
    if not d:
        return None
    return float(scipy_stats.wilcoxon(d).pvalue)


def wilcoxon_method(use_scipy):
    if use_scipy and scipy_stats is not None:
        return "scipy.stats.wilcoxon (exact distribution for small n without ties, normal approximation otherwise; zero differences dropped)"
    return "pure-Python normal approximation (average ranks for ties, tie-corrected variance, continuity correction; zero differences dropped)"


def wilcoxon_p(diffs, use_scipy=True):
    if use_scipy and scipy_stats is not None:
        try:
            return wilcoxon_scipy(diffs)
        except Exception:  # fall back to the built-in approximation
            return wilcoxon_normal(diffs)
    return wilcoxon_normal(diffs)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def primary_rows(records):
    """One row per (config, id): the lowest rep (normally rep 0)."""
    best = {}
    for r in records:
        key = (r["config"], r["id"])
        if key not in best or r["rep"] < best[key]["rep"]:
            best[key] = r
    return best


def order_configs(records, baseline):
    metas = {}
    for r in records:
        m = metas.setdefault(r["config"], {"depth": None, "K": None, "compile": None})
        for k in ("depth", "K", "compile"):
            if m[k] is None and r[k] is not None:
                m[k] = r[k]

    def key(c):
        m = metas[c]
        return (
            0 if c == baseline else 1,
            m["depth"] if m["depth"] is not None else 10 ** 9,
            m["K"] if m["K"] is not None else 10 ** 9,
            c,
        )

    return sorted(metas, key=key), metas


def pooled_ratio(rows, num_field, den_fn):
    num = den = 0.0
    for r in rows:
        d = den_fn(r)
        if r[num_field] is not None and d is not None and d > 0:
            num += r[num_field]
            den += d
    return (num / den) if den > 0 else None


def analyze(records, baseline, n_boot, seed, use_numpy=True, use_scipy=True):
    configs, metas = order_configs(records, baseline)
    prim = primary_rows(records)
    by_cfg = {c: {} for c in configs}
    for (c, pid), r in prim.items():
        by_cfg[c][pid] = r
    cats = sorted({r["cat"] for r in records})

    def k_of(r, c):
        return r["K"] if r["K"] is not None else metas[c]["K"]

    # 1. per-configuration throughput
    summary = {}
    cat_medians = {}
    for c in configs:
        rows = list(by_cfg[c].values())
        toks = [r["tok_s"] for r in rows if r["tok_s"] is not None]
        d = describe(toks)
        d["ci"] = bootstrap_median_ci(toks, n_boot, seed_for(seed, c), use_numpy)
        ma = [r["mean_accept"] for r in rows if r["mean_accept"] is not None]
        d["mean_accept"] = (sum(ma) / len(ma)) if ma else None
        d["n_accept"] = len(ma)
        d["stale_frac"] = pooled_ratio(
            rows, "stale",
            lambda r: (r["stale"] + r["valid"]) if (r["stale"] is not None and r["valid"] is not None) else None)
        d["accept_rate"] = pooled_ratio(
            rows, "accepted",
            lambda r, c=c: (r["valid"] * k_of(r, c)) if (r["valid"] is not None and k_of(r, c)) else None)
        summary[c] = d
        cm = {}
        for cat in cats:
            vals = [r["tok_s"] for r in rows if r["cat"] == cat and r["tok_s"] is not None]
            cm[cat] = (statistics.median(vals), len(vals)) if vals else (None, 0)
        cat_medians[c] = cm

    # 2. paired comparisons against the baseline
    paired = {}
    if baseline in by_cfg:
        base = by_cfg[baseline]
        for c in configs:
            if c == baseline:
                continue
            other = by_cfg[c]
            diffs = []
            for pid in sorted(set(base) & set(other)):
                b = base[pid]["tok_s"]
                o = other[pid]["tok_s"]
                if b is None or o is None:
                    continue
                diffs.append(o - b)
            n = len(diffs)
            if n == 0:
                paired[c] = {"n": 0, "median": None, "mean": None, "ci": (None, None),
                             "frac_faster": None, "frac_tied": None, "p": None}
                continue
            s = sorted(diffs)
            paired[c] = {
                "n": n,
                "median": percentile_sorted(s, 50),
                "mean": sum(diffs) / n,
                "ci": bootstrap_median_ci(diffs, n_boot, seed_for(seed, baseline + "|" + c), use_numpy),
                "frac_faster": sum(1 for x in diffs if x > 0) / n,
                "frac_tied": sum(1 for x in diffs if x == 0) / n,
                "p": wilcoxon_p(diffs, use_scipy),
            }

    # 3. acceptance by category
    acceptance = {}
    for c in configs:
        rows = list(by_cfg[c].values())
        per = {}
        for cat in cats:
            sub = [r for r in rows if r["cat"] == cat]
            ma = [r["mean_accept"] for r in sub if r["mean_accept"] is not None]
            per[cat] = {
                "n": len(sub),
                "mean_accept": (sum(ma) / len(ma)) if ma else None,
                "accept_rate": pooled_ratio(
                    sub, "accepted",
                    lambda r, c=c: (r["valid"] * k_of(r, c)) if (r["valid"] is not None and k_of(r, c)) else None),
            }
        acceptance[c] = per

    # 4. correctness: output_sha agreement between configuration pairs
    correctness = []
    for i, a in enumerate(configs):
        for b in configs[i + 1:]:
            ra, rb = by_cfg[a], by_cfg[b]
            shared = [pid for pid in sorted(set(ra) & set(rb))
                      if ra[pid]["output_sha"] and rb[pid]["output_sha"]]
            if not shared:
                continue
            mism = [pid for pid in shared if ra[pid]["output_sha"] != rb[pid]["output_sha"]]
            correctness.append({
                "a": a, "b": b, "shared": len(shared), "identical": len(shared) - len(mism),
                "fraction": (len(shared) - len(mism)) / len(shared), "mismatches": mism,
            })

    # 5. jitter across reps
    groups = {}
    for r in records:
        if r["tok_s"] is not None:
            groups.setdefault((r["config"], r["id"]), []).append(r)
    jitter = {}
    for c in configs:
        cvs = []
        nreps = []
        for (cc, pid), rows in groups.items():
            if cc != c or len(rows) < 2 or not any(r["rep"] > 0 for r in rows):
                continue
            vals = [r["tok_s"] for r in rows]
            m = sum(vals) / len(vals)
            if m <= 0:
                continue
            cvs.append(statistics.stdev(vals) / m)
            nreps.append(len(vals))
        if cvs:
            jitter[c] = {"n": len(cvs), "median_cv": statistics.median(cvs), "max_cv": max(cvs),
                         "mean_reps": sum(nreps) / len(nreps)}
        else:
            jitter[c] = None

    return {
        "configs": configs, "metas": metas, "cats": cats, "summary": summary,
        "cat_medians": cat_medians, "paired": paired, "acceptance": acceptance,
        "correctness": correctness, "jitter": jitter, "baseline": baseline,
        "n_boot": n_boot, "seed": seed, "n_records": len(records),
        "n_primary": len(prim), "used_numpy": bool(use_numpy and np is not None),
        "wilcoxon_method": wilcoxon_method(use_scipy),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def f2(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def fpct(x):
    return "n/a" if x is None else f"{100.0 * x:.1f}%"


def fp(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    return f"{p:.1e}" if p < 1e-4 else f"{p:.4f}"


def fci(ci, nd=2):
    lo, hi = ci
    if lo is None or hi is None:
        return "n/a"
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def fbool(b):
    return "?" if b is None else ("yes" if b else "no")


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def render(an, load_info):
    L = []
    cfgs = an["configs"]
    L.append("# WAN-PP throughput study report")
    L.append("")
    L.append(f"Inputs: {len(load_info)} file(s), {an['n_records']} records, "
             f"{an['n_primary']} primary (config, prompt) rows, {len(cfgs)} configuration(s). "
             f"Baseline: `{an['baseline']}`.")
    for path, ok, bad in load_info:
        L.append(f"- `{path}`: {ok} records" + (f", {bad} unreadable line(s) skipped" if bad else ""))
    L.append("")
    L.append("Method notes:")
    L.append("- One primary row per (config, prompt): the lowest rep (normally rep 0). Higher reps feed only the jitter section.")
    L.append("- tok/s is `tok_s`, or `ntok / seconds` when `tok_s` is missing.")
    L.append(f"- Bootstrap CIs: {an['n_boot']} resamples of the median, percentile method, seed {an['seed']}, "
             + ("numpy" if an["used_numpy"] else "pure Python") + " sampler.")
    L.append("- Paired differences are other minus baseline in tok/s, so a positive value means the other configuration is faster.")
    L.append(f"- Wilcoxon signed-rank test: {an['wilcoxon_method']}.")
    L.append("- Stale fraction = stale / (valid + stale), pooled over prompts. Accept rate = accepted / (valid * K), pooled over prompts. Mean accept = mean of per-prompt `mean_accept`.")
    L.append("")

    # 1
    L.append("## 1. Per-configuration throughput (tok/s)")
    L.append("")
    rows = []
    for c in cfgs:
        s = an["summary"][c]
        m = an["metas"][c]
        iqr = "n/a" if s["q1"] is None else f"{s['q1']:.2f} .. {s['q3']:.2f}"
        rows.append([f"`{c}`", m["depth"] if m["depth"] is not None else "?", m["K"] if m["K"] is not None else "?",
                     fbool(m["compile"]), s["n"], f2(s["median"]), fci(s["ci"]), f2(s["mean"]), iqr,
                     f2(s["min"]), f2(s["max"]), f2(s["mean_accept"]), fpct(s["stale_frac"])])
    L.append(md_table(["config", "depth", "K", "compile", "n", "median", "95% CI (median)", "mean",
                       "IQR (Q1 .. Q3)", "min", "max", "mean accept", "stale frac"], rows))
    L.append("")
    L.append("Per-category median tok/s (n prompts in parentheses):")
    L.append("")
    rows = []
    for c in cfgs:
        cells = [f"`{c}`"]
        for cat in an["cats"]:
            med, n = an["cat_medians"][c][cat]
            cells.append("n/a" if med is None else f"{med:.2f} ({n})")
        rows.append(cells)
    L.append(md_table(["config"] + an["cats"], rows))
    L.append("")

    # 2
    L.append(f"## 2. Paired comparisons against `{an['baseline']}`")
    L.append("")
    if an["baseline"] not in an["summary"]:
        L.append(f"Baseline `{an['baseline']}` not found in the data. Available: "
                 + ", ".join(f"`{c}`" for c in cfgs) + ".")
    elif not an["paired"]:
        L.append("No other configuration to compare.")
    else:
        rows = []
        for c in cfgs:
            if c not in an["paired"]:
                continue
            p = an["paired"][c]
            rows.append([f"`{c}`", p["n"], f2(p["median"]), fci(p["ci"]), f2(p["mean"]),
                         fpct(p["frac_faster"]), fpct(p["frac_tied"]), fp(p["p"])])
        L.append(md_table(["config", "n pairs", "median diff", "95% CI (median diff)", "mean diff",
                           "faster than baseline", "tied", "Wilcoxon p"], rows))
    L.append("")

    # 3
    L.append("## 3. Acceptance by category")
    L.append("")
    rows = []
    for c in cfgs:
        for cat in an["cats"]:
            a = an["acceptance"][c][cat]
            if a["n"] == 0:
                continue
            rows.append([f"`{c}`", cat, a["n"], f2(a["mean_accept"]), fpct(a["accept_rate"])])
    if rows:
        L.append(md_table(["config", "category", "n", "mean accept", "accept rate (accepted / (valid * K))"], rows))
    else:
        L.append("No acceptance data.")
    L.append("")

    # 4
    L.append("## 4. Correctness (output_sha agreement between configurations)")
    L.append("")
    if not an["correctness"]:
        L.append("No configuration pair shares prompts with `output_sha` on both sides.")
    else:
        rows = []
        for e in an["correctness"]:
            mism = e["mismatches"]
            shown = ", ".join(f"`{x}`" for x in mism[:100])
            if len(mism) > 100:
                shown += f", and {len(mism) - 100} more"
            rows.append([f"`{e['a']}`", f"`{e['b']}`", e["shared"], e["identical"], fpct(e["fraction"]),
                         shown if mism else "(none)"])
        L.append(md_table(["config A", "config B", "shared prompts", "identical", "fraction identical",
                           "mismatching ids"], rows))
    L.append("")

    # 5
    L.append("## 5. Jitter (within-prompt CV of tok/s across reps)")
    L.append("")
    rows = []
    for c in cfgs:
        j = an["jitter"][c]
        if j is None:
            rows.append([f"`{c}`", 0, "n/a", "n/a", "n/a"])
        else:
            rows.append([f"`{c}`", j["n"], f2(j["mean_reps"], 1), fpct(j["median_cv"]), fpct(j["max_cv"])])
    L.append(md_table(["config", "prompts with reps", "mean reps per prompt", "median CV", "max CV"], rows))
    L.append("")
    L.append("CV uses the sample standard deviation (n - 1) divided by the mean of tok/s over all reps of one prompt.")
    L.append("")
    return "\n".join(L)


CSV_FIELDS = ["config", "depth", "K", "compile", "n", "median_tok_s", "ci_lo", "ci_hi", "mean_tok_s",
              "q1", "q3", "min_tok_s", "max_tok_s", "mean_accept", "stale_frac", "accept_rate",
              "jitter_prompts", "jitter_median_cv", "paired_n", "paired_median_diff", "paired_ci_lo",
              "paired_ci_hi", "paired_frac_faster", "paired_wilcoxon_p"]


def csv_rows(an):
    rows = []
    for c in an["configs"]:
        s = an["summary"][c]
        m = an["metas"][c]
        j = an["jitter"][c]
        p = an["paired"].get(c)
        rows.append({
            "config": c, "depth": m["depth"], "K": m["K"], "compile": m["compile"], "n": s["n"],
            "median_tok_s": s["median"], "ci_lo": s["ci"][0], "ci_hi": s["ci"][1], "mean_tok_s": s["mean"],
            "q1": s["q1"], "q3": s["q3"], "min_tok_s": s["min"], "max_tok_s": s["max"],
            "mean_accept": s["mean_accept"], "stale_frac": s["stale_frac"], "accept_rate": s["accept_rate"],
            "jitter_prompts": j["n"] if j else 0, "jitter_median_cv": j["median_cv"] if j else None,
            "paired_n": p["n"] if p else None, "paired_median_diff": p["median"] if p else None,
            "paired_ci_lo": p["ci"][0] if p else None, "paired_ci_hi": p["ci"][1] if p else None,
            "paired_frac_faster": p["frac_faster"] if p else None, "paired_wilcoxon_p": p["p"] if p else None,
        })
    return rows


def write_csv(path, an):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in csv_rows(an):
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def fabricate(seed=7):
    """3 configs x 30 prompts x 2 reps of plausible records, with planted effects.

    d6_k4 is about 3 tok/s faster than the baseline d6_k2, d6_k2_nc (compile
    off) is about 1.5 tok/s slower and disagrees on two output hashes. A few
    fields are deleted to exercise the tolerance paths.
    """
    rng = random.Random(seed)
    cats = ["code"] * 12 + ["prose"] * 6 + ["reasoning"] * 6 + ["instruct"] * 6
    counters = {}
    ids = []
    for cat in cats:
        counters[cat] = counters.get(cat, 0) + 1
        ids.append(f"{cat}-{counters[cat]:03d}")
    latent = {pid: rng.gauss(0.0, 1.0) for pid in ids}
    cat_base = {"code": 22.0, "prose": 20.0, "reasoning": 19.0, "instruct": 21.0}
    configs = [("d6_k2", 6, 2, True, 0.0), ("d6_k4", 6, 4, True, 3.0), ("d6_k2_nc", 6, 2, False, -1.5)]
    planted = {"code-003", "prose-002"}
    lines = []
    for cfg, depth, k, comp, effect in configs:
        for pid, cat in zip(ids, cats):
            for rep in (0, 1):
                tok_s = cat_base[cat] + latent[pid] + effect + rng.gauss(0.0, 0.3)
                mean_accept = min(float(k), max(1.0, rng.gauss(1.0 + 0.45 * k, 0.15)))
                valid = max(1, int(round(96 / mean_accept)))
                accepted = min(96, int(round(valid * mean_accept)))
                sha_src = pid + ("x" if (cfg == "d6_k2_nc" and pid in planted) else "")
                rec = {
                    "config": cfg, "depth": depth, "K": k, "compile": comp, "id": pid, "cat": cat,
                    "rep": rep, "ntok": 96, "seconds": round(96 / tok_s, 3), "tok_s": round(tok_s, 3),
                    "valid": valid, "stale": rng.randint(0, 8), "accepted": accepted,
                    "mean_accept": round(accepted / valid, 3),
                    "output_sha": hashlib.sha1(sha_src.encode()).hexdigest()[:16],
                }
                lines.append(rec)
    # tolerance: delete some fields
    for rec in lines:
        if rec["config"] == "d6_k4" and rec["id"] in ("code-001", "code-002", "prose-001"):
            del rec["tok_s"]  # derivable from ntok / seconds
        if rec["config"] == "d6_k4" and rec["id"] in ("instruct-001", "instruct-002") and rec["rep"] == 0:
            del rec["output_sha"]
        if rec["config"] == "d6_k2" and rec["id"] == "reasoning-006" and rec["rep"] == 1:
            del rec["mean_accept"]  # derivable from accepted / valid
        if rec["config"] == "d6_k2_nc" and rec["id"] == "code-012" and rec["rep"] == 0:
            del rec["rep"]  # defaults to 0
    text = "\n".join(json.dumps(r) for r in lines) + "\nthis line is not json\n"
    return text, planted, ids


def selftest(args):
    print("# study_stats selftest: 3 configs x 30 prompts x 2 reps (fabricated), "
          f"boot={args.boot}, numpy={'on' if (np is not None and not args.no_numpy) else 'off'}, "
          f"scipy={'on' if (scipy_stats is not None and not args.no_scipy) else 'off'}")
    text, planted, ids = fabricate()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "fabricated.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        records, info = load_files([path])
        assert info[0][1] == 180 and info[0][2] == 1, info
        an = analyze(records, "d6_k2", args.boot, args.seed,
                     use_numpy=not args.no_numpy, use_scipy=not args.no_scipy)
        report = render(an, info)
        csv_path = os.path.join(tmp, "summary.csv")
        write_csv(csv_path, an)
        with open(csv_path, newline="") as fh:
            csv_lines = list(csv.DictReader(fh))
        md_path = os.path.join(tmp, "report.md")
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(report)
        md_size = os.path.getsize(md_path)
    print(report)

    checks = []

    def check(name, cond):
        checks.append((name, bool(cond)))

    for h in ("## 1. Per-configuration", "## 2. Paired comparisons", "## 3. Acceptance by category",
              "## 4. Correctness", "## 5. Jitter"):
        check(f"section present: {h}", h in report)
    check("no em-dash in report", "\u2014" not in report)
    check("three configs found", an["configs"] == ["d6_k2", "d6_k2_nc", "d6_k4"])
    s = an["summary"]
    for c in an["configs"]:
        check(f"{c}: n == 30", s[c]["n"] == 30)
        lo, hi = s[c]["ci"]
        check(f"{c}: CI brackets median", lo <= s[c]["median"] <= hi)
        check(f"{c}: min <= q1 <= median <= q3 <= max",
              s[c]["min"] <= s[c]["q1"] <= s[c]["median"] <= s[c]["q3"] <= s[c]["max"])
        check(f"{c}: stale frac in (0, 1)", s[c]["stale_frac"] is not None and 0 < s[c]["stale_frac"] < 1)
        check(f"{c}: accept rate in (0, 1]", s[c]["accept_rate"] is not None and 0 < s[c]["accept_rate"] <= 1)
        check(f"{c}: per-category medians for 4 categories",
              sum(1 for cat, (m, n) in an["cat_medians"][c].items() if m is not None) == 4)
    check("d6_k4 median > baseline median", s["d6_k4"]["median"] > s["d6_k2"]["median"])
    check("d6_k2_nc median < baseline median", s["d6_k2_nc"]["median"] < s["d6_k2"]["median"])
    p4 = an["paired"]["d6_k4"]
    pnc = an["paired"]["d6_k2_nc"]
    check("paired d6_k4: 30 pairs", p4["n"] == 30)
    check("paired d6_k4: median diff between 2 and 4", 2.0 < p4["median"] < 4.0)
    check("paired d6_k4: CI brackets median diff", p4["ci"][0] <= p4["median"] <= p4["ci"][1])
    check("paired d6_k4: CI excludes zero", p4["ci"][0] > 0)
    check("paired d6_k4: fraction faster == 1", p4["frac_faster"] == 1.0)
    check("paired d6_k4: Wilcoxon p < 0.001", p4["p"] is not None and p4["p"] < 1e-3)
    check("paired d6_k2_nc: median diff between -2.5 and -0.5", -2.5 < pnc["median"] < -0.5)
    check("paired d6_k2_nc: fraction faster < 0.1", pnc["frac_faster"] < 0.1)
    # Wilcoxon sanity on a symmetric sample: p should be large
    sym = [x for x in range(-10, 11) if x != 0]
    check("Wilcoxon normal approx: symmetric sample gives p > 0.9", wilcoxon_normal(sym) > 0.9)
    check("Wilcoxon normal approx: all-positive n=30 gives p < 1e-5",
          wilcoxon_normal([1.0 + 0.1 * i for i in range(30)]) < 1e-5)
    check("Wilcoxon normal approx: all zeros gives None", wilcoxon_normal([0.0, 0.0]) is None)
    if scipy_stats is not None:
        d = [1.0 + 0.1 * i for i in range(30)] + [-0.3, -0.7]
        check("Wilcoxon scipy vs normal approx agree within 0.01",
              abs(wilcoxon_scipy(d) - wilcoxon_normal(d)) < 0.01)
    # bootstrap: both samplers bracket the plain median
    toks = [r["tok_s"] for r in primary_rows(records).values() if r["config"] == "d6_k2"]
    lo, hi = bootstrap_median_ci(toks, args.boot, 123, use_numpy=False)
    check("pure-Python bootstrap brackets median", lo <= statistics.median(toks) <= hi)
    check("pure-Python bootstrap is reproducible",
          bootstrap_median_ci(toks, args.boot, 123, use_numpy=False) == (lo, hi))
    if np is not None:
        lo2, hi2 = bootstrap_median_ci(toks, args.boot, 123, use_numpy=True)
        check("numpy bootstrap brackets median", lo2 <= statistics.median(toks) <= hi2)
        check("numpy and pure-Python CIs within 0.5 tok/s", abs(lo2 - lo) < 0.5 and abs(hi2 - hi) < 0.5)
    corr = {(e["a"], e["b"]): e for e in an["correctness"]}
    e_nc = corr[("d6_k2", "d6_k2_nc")]
    e_k4 = corr[("d6_k2", "d6_k4")]
    check("correctness d6_k2 vs d6_k2_nc: planted mismatches found", set(e_nc["mismatches"]) == planted)
    check("correctness d6_k2 vs d6_k2_nc: 28/30 identical", e_nc["shared"] == 30 and e_nc["identical"] == 28)
    check("correctness d6_k2 vs d6_k4: 28 shared (2 shas missing), all identical",
          e_k4["shared"] == 28 and e_k4["fraction"] == 1.0)
    for c in an["configs"]:
        j = an["jitter"][c]
        check(f"jitter {c}: 30 prompts with reps, median CV in (0, 5%)",
              j is not None and j["n"] == 30 and 0 < j["median_cv"] < 0.05)
    check("tolerance: tok_s derived from ntok/seconds",
          all(r["tok_s"] is not None for r in records if r["config"] == "d6_k4"))
    check("tolerance: mean_accept derived from accepted/valid",
          all(r["mean_accept"] is not None for r in records))
    check("csv: 3 data rows with all fields", len(csv_lines) == 3 and all(set(r) == set(CSV_FIELDS) for r in csv_lines))
    check("csv: baseline row has empty paired columns", csv_lines[0]["paired_n"] == "")
    check("markdown written", md_size > 1000)

    failed = [n for n, ok in checks if not ok]
    print(f"# selftest checks: {len(checks) - len(failed)} passed, {len(failed)} failed")
    for n in failed:
        print(f"#   FAIL: {n}")
    if failed:
        print("SELFTEST FAIL")
        return 1
    print("SELFTEST PASS")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Markdown statistics report for WAN-PP results JSONL files.")
    ap.add_argument("files", nargs="*", help="results JSONL files (one record per config, prompt, rep)")
    ap.add_argument("--baseline", default="d6_k2", help="baseline config for paired comparisons (default d6_k2)")
    ap.add_argument("--out", help="write the Markdown report here (default: stdout)")
    ap.add_argument("--csv", help="write a per-config summary CSV here")
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap resamples (default 10000)")
    ap.add_argument("--seed", type=int, default=20260824, help="bootstrap seed (default 20260824)")
    ap.add_argument("--no-numpy", action="store_true", help="force the pure-Python bootstrap")
    ap.add_argument("--no-scipy", action="store_true", help="force the pure-Python Wilcoxon approximation")
    ap.add_argument("--selftest", action="store_true", help="fabricate data and exercise every code path")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest(args)
    if not args.files:
        ap.error("no input files given (or use --selftest)")

    records, info = load_files(args.files)
    if not records:
        print("no usable records found", file=sys.stderr)
        return 1
    an = analyze(records, args.baseline, args.boot, args.seed,
                 use_numpy=not args.no_numpy, use_scipy=not args.no_scipy)
    report = render(an, info)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(report)
    if args.csv:
        write_csv(args.csv, an)
        print(f"wrote {args.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
