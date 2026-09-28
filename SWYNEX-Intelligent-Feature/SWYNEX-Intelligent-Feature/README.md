# SWYNEX-Intelligent-Feature (Task 3)

Builds on the Task 2 DataQA Chatbot. Task 2 answered questions with a free
Hugging Face model (DistilBERT-SQuAD) reading a text summary of a sales CSV.
It worked for questions whose answer was written in the summary and failed,
**confidently**, for everything else.

**Task 3 adds an intelligent feature, error handling, evaluation examples with
failure cases, and a simple web interface.**

## The intelligent feature: a hybrid answer router

```
question ─► input checks ─► 1. RULES (pandas, exact) ──► answer
                                   │ no match
                                   ▼
                            2. MODEL (DistilBERT-SQuAD on the summary)
                                   │
                                   ▼
                            3. GUARDS: confidence >= threshold? numeric question has a number?
                                   │ yes                         │ no
                                   ▼                             ▼
                                answer                   "I'm not sure" (abstain)
```

- **Rules** handle questions that need *computation* (lowest region, revenue of a
  named region, month with the lowest revenue, averages, counts). Exact, instant,
  and they need no model.
- **Model** handles the rest by extracting a span from the generated summary.
- **Abstention** replaces confident-but-wrong answers with an honest "I'm not sure".

Every answer reports how it was produced (`rules` / `model` / `abstain` / `error`).

## Error handling

| Situation | Behaviour |
|---|---|
| CSV missing / empty / unreadable | `DataError` with a clear message, no stack trace in the UI |
| Required column missing | Lists exactly which columns are missing |
| Bad rows (unparseable dates or numbers) | Dropped and counted; error only if nothing is left |
| Empty or very long question (>300 chars) | Friendly message, model not called |
| Model can't load (offline, no torch) | Falls back to rules-only and says so |
| Model crashes at runtime | Returned as an `error` answer with details, never an exception |
| Low confidence / numeric question without a number | Abstains |

## Run it

```bash
pip install -r requirements.txt

python interface.py               # web UI (downloads the ~250 MB model on first run)
python interface.py --no-model    # web UI, rules-only, fully offline
python evaluate.py                # full evaluation incl. the model
python evaluate.py --no-model     # offline evaluation
pytest                            # 24 unit tests, all offline
```

In a notebook/Colab: `from interface import build_ui; build_ui().launch()`

## Evaluation

`evaluate.py` runs 20 questions in three categories through three modes and
saves `results/eval_results.md` and `results/eval_results.json`.

| Category | Meaning | Example |
|---|---|---|
| grounded | Answer is written in the summary | "Which region generated the highest revenue?" |
| computed | Needs calculation not in the summary | "How much revenue did North generate?" |
| unanswerable | Not answerable from the data; correct behaviour is to abstain | "Who is the CEO of the company?" |

Modes: **model-only** (Task 2 behaviour), **rules-only**, **hybrid** (Task 3).
Expected answers are computed independently in plain Python, not by the code under test.

### Results

Rules-only (offline, reproducible with `python evaluate.py --no-model`):

| Mode | grounded | computed | unanswerable | OVERALL |
|---|---|---|---|---|
| rules-only | 8/8 | 7/7 | 5/5 | 20/20 |

Model modes: **run `python evaluate.py` and paste the model-only and hybrid rows
here.** They need the model download, so they are not included until they come from
a real run. The script also prints a confidence-threshold sweep (0.0-0.6) showing
how raising the threshold trades wrong answers for abstentions.

### Failure cases found and what was done

1. **Task 2, observed:** "Which month had the most units sold?" returned `2024-02`
   with confidence 0.90 although the summary contained no monthly data. High
   confidence did not mean the answer was grounded. -> Monthly stats added to the
   summary, month questions moved to exact rules, and low-confidence answers abstain.
2. **Task 2, observed:** "What was the lowest revenue region?" returned `West` at
   0.50; the summary only named the *highest* region. -> Lowest-region rule added.
3. **Found by `evaluate.py`:** "Why did the top region perform best?" was answered
   "East", because the router matched "region" + "top" and ignored the word *why*.
   The first evaluation run scored 19/20 because of this. -> Causal and forecasting
   words (why, because, predict, forecast, will, next, future) now bypass the
   rules; regression test added (`tests/test_dataqa.py`). Now 20/20.

### Known limitations

- Rule matching is keyword-based, so unusual phrasings fall through to the model.
- Quarter, year-over-year and multi-condition questions ("East revenue in March")
  are not supported and correctly abstain.
- Extractive QA can only copy text from the summary; it cannot calculate.
- The dataset is synthetic sample data.

## Project structure

```
dataqa.py        core: validation, summary builder, rule router, model backend, hybrid bot
interface.py     Gradio web UI
evaluate.py      evaluation harness (3 modes, threshold sweep, failure table)
tests/           unit tests (data errors, rules, abstention, span extraction)
data/            sales_data.csv (120 rows, Jan-Aug 2024)
results/         saved evaluation output
```
