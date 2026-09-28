import pandas as pd
import pytest

import dataqa
from dataqa import Answer, DataError, DataQABot, ModelUnavailableError


@pytest.fixture()
def df():
    return dataqa.validate_data(pd.DataFrame({
        "date": ["2024-01-05", "2024-01-20", "2024-02-10", "2024-02-15"],
        "region": ["North", "South", "North", "East"],
        "product": ["Widget A", "Widget B", "Widget A", "Widget C"],
        "units_sold": [10, 20, 30, 5],
        "revenue": [100.0, 350.0, 300.0, 50.0],
    }))


# ---------------- error handling: data ----------------
def test_missing_file():
    with pytest.raises(DataError, match="not found"):
        dataqa.load_data("nope.csv")


def test_empty_file(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("")
    with pytest.raises(DataError, match="empty"):
        dataqa.load_data(p)


def test_missing_columns():
    with pytest.raises(DataError, match="Missing required column"):
        dataqa.validate_data(pd.DataFrame({"date": ["2024-01-01"], "region": ["N"]}))


def test_bad_rows_are_dropped_and_counted():
    raw = pd.DataFrame({"date": ["2024-01-01", "not a date", "2024-01-03"],
                        "region": ["N", "S", "E"], "product": ["A", "B", "C"],
                        "units_sold": [1, 2, "oops"], "revenue": [10, 20, 30]})
    clean = dataqa.validate_data(raw)
    assert len(clean) == 1 and clean.attrs["dropped_rows"] == 2


def test_all_rows_invalid():
    raw = pd.DataFrame({"date": ["x"], "region": ["N"], "product": ["A"], "units_sold": ["a"], "revenue": ["b"]})
    with pytest.raises(DataError, match="No valid rows"):
        dataqa.validate_data(raw)


# ---------------- rules ----------------
def test_top_region(df):
    ans = dataqa.rule_answer(df, "Which region generated the highest revenue?")
    assert ans.source == "rules" and ans.text.startswith("North") and "400" in ans.text


def test_lowest_region(df):
    assert dataqa.rule_answer(df, "Which region had the lowest revenue?").text.startswith("East")


def test_best_and_lowest_product(df):
    assert dataqa.rule_answer(df, "What was the best-selling product?").text.startswith("Widget A")
    assert dataqa.rule_answer(df, "What was the lowest-selling product?").text.startswith("Widget C")


def test_named_region_revenue(df):
    assert "400" in dataqa.rule_answer(df, "How much revenue did North generate?").text


def test_totals_and_average(df):
    assert dataqa.rule_answer(df, "How many total units were sold?").text == "65 units"
    assert dataqa.rule_answer(df, "What was the average revenue per sale?").text.startswith("200")


def test_month_lookup(df):
    assert dataqa.rule_answer(df, "Which month had the most units sold?").text.startswith("2024-02")


@pytest.mark.parametrize("q", ["Why did North perform best?", "What will revenue be next month?",
                               "Who is the CEO?", "Which quarter had the highest revenue?"])
def test_out_of_scope_questions_are_not_answered_by_rules(df, q):
    assert dataqa.rule_answer(df, q) is None


# ---------------- bot behaviour ----------------
class StubBackend:
    def __init__(self, text="42", score=0.9, exc=None):
        self.text, self.score, self.exc = text, score, exc

    def answer(self, q, ctx):
        if self.exc:
            raise self.exc
        return self.text, self.score


def test_empty_and_long_questions(df):
    bot = DataQABot(df)
    assert bot.ask("   ").source == "error"
    assert bot.ask(None).source == "error"
    assert bot.ask("a" * 1000).source == "error"


def test_rules_only_mode_abstains_on_unknown(df):
    assert DataQABot(df).ask("Who is the CEO?").source == "abstain"


def test_low_confidence_abstains(df):
    bot = DataQABot(df, StubBackend("2024-02", 0.12))
    a = bot.ask("Who is the CEO?")
    assert a.source == "abstain" and "0.12" in a.note


def test_greeting_does_not_get_answered_from_data_summary(df):
    a = DataQABot(df, StubBackend("2024-08-26", 0.40)).ask("hello")
    assert a.source == "abstain"
    assert "sales data" in a.text


def test_confident_model_answer_is_returned(df):
    a = DataQABot(df, StubBackend("Widget A", 0.9)).ask("Which item is popular?")
    assert a.source == "model" and a.text == "Widget A"


def test_numeric_question_needs_a_number(df):
    a = DataQABot(df, StubBackend("East", 0.95)).ask("How many staff work here?")
    assert a.source == "abstain"


def test_model_unavailable_falls_back_gracefully(df):
    bot = DataQABot(df, StubBackend(exc=ModelUnavailableError("no internet")))
    assert bot.ask("Who is the CEO?").source == "abstain"
    assert bot.ask("Which region had the lowest revenue?").source == "rules"   # rules still work


def test_unexpected_model_crash_is_reported_not_raised(df):
    a = DataQABot(df, StubBackend(exc=RuntimeError("boom"))).ask("Who is the CEO?")
    assert a.source == "error" and "boom" in a.note
