"""
dataqa.py - core logic for the DataQA Chatbot (Task 3: Intelligent Feature)

The "intelligent feature" is a HYBRID ANSWER ROUTER:

  1. Rules   - questions that need computation (lowest region, per-region
               revenue, averages, ...) are answered exactly with pandas.
  2. Model   - everything else goes to a free Hugging Face extractive QA
               model (DistilBERT-SQuAD) reading a generated summary.
  3. Abstain - if the model's confidence is low, or its answer looks
               wrong for the question type, the bot says it isn't sure
               instead of guessing.

This fixes the Task 2 failure mode where the model answered confidently
about things that were never in its context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

REQUIRED_COLUMNS = ("date", "region", "product", "units_sold", "revenue")
MODEL_NAME = "distilbert-base-cased-distilled-squad"
MAX_QUESTION_CHARS = 300
DEFAULT_THRESHOLD = 0.30

HIGH_WORDS = ("highest", "most", "top", "best", "maximum", "max", "largest", "biggest", "greatest")
LOW_WORDS = ("lowest", "least", "worst", "minimum", "min", "smallest", "bottom", "fewest", "weakest")


# --------------------------------------------------------------------------
# Errors and result type
# --------------------------------------------------------------------------
class DataError(Exception):
    """The dataset is missing, unreadable, or invalid."""


class ModelUnavailableError(Exception):
    """The Hugging Face model could not be loaded (no internet, no torch...)."""


@dataclass
class Answer:
    text: str
    source: str  # "rules" | "model" | "abstain" | "error"
    confidence: Optional[float] = None
    note: str = ""


# --------------------------------------------------------------------------
# Data loading + validation
# --------------------------------------------------------------------------
def load_data(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise DataError(f"Data file not found: {p}")
    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError as exc:
        raise DataError(f"Data file is empty: {p}") from exc
    except Exception as exc:  # malformed CSV, encoding problems, ...
        raise DataError(f"Could not read CSV '{p}': {exc}") from exc
    return validate_data(df)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Check columns, coerce types, drop unusable rows (and remember how many)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"Missing required column(s): {', '.join(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("units_sold", "revenue"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)
    if df.empty:
        raise DataError("No valid rows left after cleaning the data.")
    df.attrs["dropped_rows"] = before - len(df)
    return df


# --------------------------------------------------------------------------
# Context paragraph (what the extractive model gets to read)
# --------------------------------------------------------------------------
def fmt_num(x: float) -> str:
    x = float(x)
    return f"{int(round(x)):,}" if abs(x - round(x)) < 1e-9 else f"{x:,.2f}"


def build_context(df: pd.DataFrame) -> str:
    by_region = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    by_product = df.groupby("product")["units_sold"].sum().sort_values(ascending=False)
    monthly = df.groupby(df["date"].dt.to_period("M"))["units_sold"].sum().sort_values(ascending=False)

    total_units = df["units_sold"].sum()
    total_rev = df["revenue"].sum()
    return (
        f"The dataset covers sales from {df['date'].min().date()} to {df['date'].max().date()} "
        f"across regions: {', '.join(sorted(df['region'].unique()))}. "
        f"A total of {fmt_num(total_units)} units were sold across {len(df)} sales, generating "
        f"{fmt_num(total_rev)} dollars in revenue, an average of {fmt_num(total_rev / len(df))} dollars per sale. "
        f"The region with the highest revenue was {by_region.index[0]}, which generated {fmt_num(by_region.iloc[0])} dollars. "
        f"The region with the lowest revenue was {by_region.index[-1]}, which generated {fmt_num(by_region.iloc[-1])} dollars. "
        f"The best-selling product overall was {by_product.index[0]}, with {fmt_num(by_product.iloc[0])} units sold. "
        f"The lowest-selling product was {by_product.index[-1]}, with {fmt_num(by_product.iloc[-1])} units sold. "
        f"{df['product'].nunique()} products were sold: {', '.join(sorted(df['product'].unique()))}. "
        f"The month with the most units sold was {monthly.index[0]}, with {fmt_num(monthly.iloc[0])} units sold."
    )


# --------------------------------------------------------------------------
# Rule-based analytics (exact answers straight from pandas)
# --------------------------------------------------------------------------
def _has(q: str, words) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", q) for w in words)


def _metric(q: str, default: str) -> str:
    if "revenue" in q or "sales value" in q or "earn" in q:
        return "revenue"
    if _has(q, ("unit", "units", "sold", "selling", "volume", "quantity")):
        return "units_sold"
    return default


def _label(metric: str, value: float) -> str:
    return f"{fmt_num(value)} dollars" if metric == "revenue" else f"{fmt_num(value)} units"


def rule_answer(df: pd.DataFrame, question: str) -> Optional[Answer]:
    """Return an exact answer if the question matches a known analytics intent, else None."""
    q = question.lower().strip()

    # Found via evaluate.py: "Why did the top region perform best?" was wrongly answered
    # "East". Causal and forecasting questions can't be computed from this data.
    if re.search(r"\b(why|because|reason|predict|forecast|will|next|future)\b", q):
        return None

    is_high, is_low = _has(q, HIGH_WORDS), _has(q, LOW_WORDS)
    superlative = is_high != is_low  # exactly one of them

    def ok(text: str, note: str) -> Answer:
        return Answer(text, "rules", 1.0, note)

    # 1) A specific region / product named in the question -> its total
    if not superlative:
        metric = _metric(q, "")
        if metric:
            for col in ("region", "product"):
                for name in df[col].unique():
                    if re.search(rf"\b{re.escape(str(name).lower())}\b", q):
                        val = df.loc[df[col] == name, metric].sum()
                        return ok(f"{name}: {_label(metric, val)}", f"pandas: sum of {metric} where {col} = {name}")

    # 2) Highest / lowest by region, product or month
    if superlative:
        dims = (("region", "revenue"), ("product", "units_sold"))
        for col, default in dims:
            if re.search(rf"\b{col}s?\b", q):
                metric = _metric(q, default)
                grp = df.groupby(col)[metric].sum()
                name = grp.idxmax() if is_high else grp.idxmin()
                return ok(f"{name} ({_label(metric, grp[name])})", f"pandas: groupby {col}, sum {metric}")
        if re.search(r"\bmonths?\b", q):
            metric = _metric(q, "units_sold")
            grp = df.groupby(df["date"].dt.to_period("M"))[metric].sum()
            name = grp.idxmax() if is_high else grp.idxmin()
            return ok(f"{name} ({_label(metric, grp[name])})", f"pandas: groupby month, sum {metric}")

    # 3) Averages
    if _has(q, ("average", "mean", "avg")):
        if "unit" in q:
            return ok(f"{fmt_num(df['units_sold'].mean())} units per sale", "pandas: mean units_sold")
        if "revenue" in q or "sale" in q:
            return ok(f"{fmt_num(df['revenue'].mean())} dollars per sale", "pandas: mean revenue")

    # 4) Totals
    if _has(q, ("total", "overall")) or q.startswith(("how many units", "how much revenue")):
        metric = _metric(q, "")
        if metric:
            return ok(_label(metric, df[metric].sum()), f"pandas: sum of {metric}")

    # 5) Counts
    if q.startswith("how many"):
        if re.search(r"\bregions?\b", q):
            return ok(str(df["region"].nunique()), "pandas: nunique region")
        if re.search(r"\bproducts?\b", q):
            return ok(str(df["product"].nunique()), "pandas: nunique product")
        if _has(q, ("sales", "transactions", "records", "rows", "orders")):
            return ok(str(len(df)), "pandas: row count")

    # 6) Date range
    if re.search(r"\b(start|begin|first date|earliest)\b", q):
        return ok(str(df["date"].min().date()), "pandas: min date")
    if re.search(r"\b(end|last date|latest)\b", q):
        return ok(str(df["date"].max().date()), "pandas: max date")

    return None


# --------------------------------------------------------------------------
# Model backend (free Hugging Face model, runs locally, no API key)
# --------------------------------------------------------------------------
def extract_answer(model, tokenizer, question: str, context: str, max_answer_tokens: int = 30) -> Tuple[str, float]:
    """
    Extractive QA without the pipeline() helper (works on transformers v4 and v5).
    Scores every (start, end) token pair jointly and returns the best span
    plus its probability.
    """
    import torch

    enc = tokenizer(
        question, context, return_tensors="pt", truncation="only_second",
        max_length=512, return_offsets_mapping=True,
    )
    offsets = enc.pop("offset_mapping")[0]
    seq_ids = enc.sequence_ids(0)
    ctx_mask = torch.tensor([s == 1 for s in seq_ids])

    with torch.no_grad():
        out = model(**enc)

    neg = -1e4
    start = torch.softmax(out.start_logits[0].masked_fill(~ctx_mask, neg), dim=-1)
    end = torch.softmax(out.end_logits[0].masked_fill(~ctx_mask, neg), dim=-1)

    joint = torch.triu(start[:, None] * end[None, :])                    # start <= end
    joint = torch.tril(joint, diagonal=max_answer_tokens - 1)             # cap span length
    idx = int(joint.argmax())
    s, e = divmod(idx, joint.shape[1])
    text = context[int(offsets[s][0]): int(offsets[e][1])].strip()
    return text, float(joint[s, e])


class ModelBackend:
    def __init__(self, name: str = MODEL_NAME):
        self.name = name
        self._tok = None
        self._model = None

    def load(self) -> None:
        try:
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.name)
            self._model = AutoModelForQuestionAnswering.from_pretrained(self.name).eval()
        except Exception as exc:
            raise ModelUnavailableError(
                f"Could not load '{self.name}'. Check your internet connection and that "
                f"torch/transformers are installed. ({type(exc).__name__}: {exc})"
            ) from exc

    def answer(self, question: str, context: str) -> Tuple[str, float]:
        if self._model is None:
            self.load()
        return extract_answer(self._model, self._tok, question, context)


# --------------------------------------------------------------------------
# The hybrid bot
# --------------------------------------------------------------------------
class DataQABot:
    def __init__(self, df: pd.DataFrame, backend: Optional[ModelBackend] = None,
                 threshold: float = DEFAULT_THRESHOLD, use_rules: bool = True):
        self.df = df
        self.context = build_context(df)
        self.backend = backend
        self.threshold = threshold
        self.use_rules = use_rules

    def ask(self, question: str) -> Answer:
        # ---- input validation
        if not isinstance(question, str) or not question.strip():
            return Answer("Please type a question about the sales data.", "error", note="empty question")
        if len(question) > MAX_QUESTION_CHARS:
            return Answer(f"That question is too long (max {MAX_QUESTION_CHARS} characters).", "error",
                          note="question too long")

        if re.fullmatch(
                r"(?:hi|hello|hey|good morning|good afternoon|good evening|how are you)(?: there)?[!.?]*",
                question.strip(),
                re.IGNORECASE,
        ):
            return Answer("Hi! I can help answer questions about the sales data.", "abstain",
                          note="greeting, not a data question")

        # ---- 1) exact analytics
        if self.use_rules:
            hit = rule_answer(self.df, question)
            if hit:
                return hit

        # ---- 2) model
        if self.backend is None:
            return Answer("I can't answer that: it isn't a question I can compute from the data, "
                          "and no language model is loaded.", "abstain", note="rules-only mode")
        try:
            text, score = self.backend.answer(question, self.context)
        except ModelUnavailableError as exc:
            return Answer("The language model isn't available, so I can only answer questions "
                          "I can compute directly (totals, highest/lowest, averages...).",
                          "abstain", note=str(exc))
        except Exception as exc:
            return Answer("Something went wrong while running the model.", "error",
                          note=f"{type(exc).__name__}: {exc}")

        # ---- 3) confidence + sanity guards
        q = question.lower()
        if score < self.threshold:
            return Answer("I'm not sure - that doesn't seem to be covered by the data I have.",
                          "abstain", score, f"model said '{text}' but confidence {score:.2f} < {self.threshold}")
        if q.startswith(("how many", "how much")) and not re.search(r"\d", text):
            return Answer("I'm not sure - that question needs a number and I couldn't find one.",
                          "abstain", score, f"model said '{text}', which has no number")
        return Answer(text, "model", score, "extracted from the generated data summary")
