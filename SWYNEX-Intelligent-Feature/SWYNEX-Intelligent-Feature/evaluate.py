"""
evaluate.py - measure how well each answering strategy works.

Modes compared
  model-only : raw DistilBERT reading the summary (the Task 2 behaviour)
  rules-only : exact pandas answers, abstain otherwise
  hybrid     : rules first, then model with a confidence threshold (the Task 3 feature)

Question categories
  grounded     - the answer is written in the summary paragraph (model should manage)
  computed     - needs calculation that is NOT in the summary (model should struggle)
  unanswerable - can't be answered from the data; the right behaviour is to abstain

Expected answers are computed independently with plain Python, so they stay
correct if you swap in a different CSV.

Usage
  python evaluate.py              # all modes (needs internet on first run for the model)
  python evaluate.py --no-model   # rules-only, works fully offline
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import dataqa
from dataqa import DataQABot, ModelBackend, ModelUnavailableError, fmt_num


def ground_truth(df):
    """Expected answers, computed with simple loops (independent of dataqa.rule_answer)."""
    rev, units = defaultdict(float), defaultdict(float)
    prod_units, prod_rev = defaultdict(float), defaultdict(float)
    month_units, month_rev = defaultdict(float), defaultdict(float)
    reg_units = defaultdict(float)
    for r in df.itertuples():
        rev[r.region] += r.revenue
        reg_units[r.region] += r.units_sold
        prod_units[r.product] += r.units_sold
        prod_rev[r.product] += r.revenue
        m = f"{r.date.year}-{r.date.month:02d}"
        month_units[m] += r.units_sold
        month_rev[m] += r.revenue
    top = lambda d: max(d, key=d.get)
    low = lambda d: min(d, key=d.get)
    some_region, some_product = sorted(rev)[1], sorted(prod_units)[2]
    return {
        "top_region": top(rev), "low_region": low(rev),
        "top_product": top(prod_units), "low_product": low(prod_units),
        "top_product_rev": top(prod_rev), "top_region_units": top(reg_units),
        "total_units": fmt_num(sum(prod_units.values())), "total_rev": fmt_num(sum(rev.values())),
        "avg_rev": fmt_num(sum(rev.values()) / len(df)),
        "top_month": top(month_units), "low_month_rev": low(month_rev),
        "n_regions": str(len(rev)), "n_sales": str(len(df)),
        "some_region": some_region, "some_region_rev": fmt_num(rev[some_region]),
        "some_product": some_product, "some_product_units": fmt_num(prod_units[some_product]),
    }


def build_cases(df):
    g = ground_truth(df)
    return [
        # grounded: answer is stated in the summary paragraph
        ("grounded", "Which region generated the highest revenue?", g["top_region"]),
        ("grounded", "Which region generated the lowest revenue?", g["low_region"]),
        ("grounded", "What was the best-selling product?", g["top_product"]),
        ("grounded", "What was the lowest-selling product?", g["low_product"]),
        ("grounded", "How many total units were sold?", g["total_units"]),
        ("grounded", "How much total revenue was generated?", g["total_rev"]),
        ("grounded", "What was the average revenue per sale?", g["avg_rev"]),
        ("grounded", "Which month had the most units sold?", g["top_month"]),
        # computed: NOT in the summary, needs pandas
        ("computed", f"How much revenue did {g['some_region']} generate?", g["some_region_rev"]),
        ("computed", f"How many units of {g['some_product']} were sold?", g["some_product_units"]),
        ("computed", "Which product generated the highest revenue?", g["top_product_rev"]),
        ("computed", "Which region sold the most units?", g["top_region_units"]),
        ("computed", "Which month had the lowest revenue?", g["low_month_rev"]),
        ("computed", "How many regions are there?", g["n_regions"]),
        ("computed", "How many sales are there?", g["n_sales"]),
        # unanswerable: correct behaviour is to abstain
        ("unanswerable", "Who is the CEO of the company?", None),
        ("unanswerable", "What will revenue be next month?", None),
        ("unanswerable", "Which quarter had the highest revenue?", None),
        ("unanswerable", "Why did the top region perform best?", None),
        ("unanswerable", "What is the weather in Paris?", None),
    ]


def norm(s):
    return s.lower().replace(",", "")


def is_correct(expected, ans):
    if expected is None:                       # unanswerable -> must abstain
        return ans.source == "abstain"
    return ans.source in ("rules", "model") and norm(expected) in norm(ans.text)


def run_mode(name, bot, cases):
    rows = []
    for cat, q, exp in cases:
        a = bot.ask(q)
        rows.append({"mode": name, "category": cat, "question": q, "expected": exp,
                     "answer": a.text, "source": a.source,
                     "confidence": None if a.confidence is None else round(a.confidence, 3),
                     "correct": is_correct(exp, a)})
    return rows


def summarize(rows):
    table = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        for key in (r["category"], "OVERALL"):
            table[r["mode"]][key][0] += r["correct"]
            table[r["mode"]][key][1] += 1
    return table


def to_markdown(rows, threshold_rows=None, model_status=""):
    cats = ["grounded", "computed", "unanswerable", "OVERALL"]
    table = summarize(rows)
    out = ["# Evaluation results\n", f"_{model_status}_\n", "## Accuracy by mode\n",
           "| Mode | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    for mode, res in table.items():
        cells = [f"{res[c][0]}/{res[c][1]} ({100 * res[c][0] / res[c][1]:.0f}%)" for c in cats]
        out.append(f"| {mode} | " + " | ".join(cells) + " |")

    if threshold_rows:
        out += ["\n## Confidence-threshold sweep (model + guards, no rules)\n",
                "| Threshold | Correct answers | Correct abstentions | Wrong answers |", "|---|---|---|---|"]
        out += [f"| {t} | {a} | {b} | {c} |" for t, a, b, c in threshold_rows]

    fails = [r for r in rows if not r["correct"]]
    out += ["\n## Failure cases\n"]
    if not fails:
        out.append("None.")
    else:
        out += ["| Mode | Category | Question | Expected | Got | Source |", "|---|---|---|---|---|---|"]
        for r in fails:
            got = r["answer"].replace("|", "/")[:70]
            out.append(f"| {r['mode']} | {r['category']} | {r['question']} | {r['expected'] or '(abstain)'} | {got} | {r['source']} |")
    return "\n".join(out) + "\n"


def threshold_sweep(df, backend, cases):
    """Model-only pipeline with different abstention thresholds (no rules)."""
    cache = {q: backend.answer(q, dataqa.build_context(df)) for _, q, _ in cases}

    class Cached:
        def answer(self, q, ctx):
            return cache[q]

    out = []
    for t in (0.0, 0.2, 0.4, 0.6):
        bot = DataQABot(df, Cached(), threshold=t, use_rules=False)
        ok_ans = ok_abs = wrong = 0
        for _, q, exp in cases:
            a = bot.ask(q)
            if is_correct(exp, a):
                ok_abs += exp is None
                ok_ans += exp is not None
            else:
                wrong += 1
        out.append((t, ok_ans, ok_abs, wrong))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sales_data.csv")
    ap.add_argument("--no-model", action="store_true", help="skip the Hugging Face model (offline)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    df = dataqa.load_data(args.data)
    cases = build_cases(df)
    rows = run_mode("rules-only", DataQABot(df, None), cases)
    sweep, status = None, "Model modes were skipped (--no-model). Run without that flag to include them."

    if not args.no_model:
        backend = ModelBackend()
        try:
            print("Loading model (first run downloads ~250 MB)...")
            backend.load()
            rows += run_mode("model-only", DataQABot(df, backend, threshold=0.0, use_rules=False), cases)
            rows += run_mode("hybrid", DataQABot(df, backend), cases)
            sweep = threshold_sweep(df, backend, cases)
            status = f"All modes run with {dataqa.MODEL_NAME}."
        except ModelUnavailableError as exc:
            status = f"Model could not be loaded, so model modes were skipped: {exc}"
            print(status)

    Path(args.out).mkdir(exist_ok=True)
    (Path(args.out) / "eval_results.json").write_text(json.dumps(rows, indent=2))
    md = to_markdown(rows, sweep, status)
    (Path(args.out) / "eval_results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
