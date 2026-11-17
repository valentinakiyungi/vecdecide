"""Tests for the CSV workflow and command line."""
import csv
import json
from pathlib import Path

import pytest

from vecdecide import Model, evaluate, read_rows
from vecdecide.__main__ import main

SAMPLE = Path(__file__).parent.parent / "examples" / "tickets.csv"


def write_csv(path, rows, header):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def test_read_rows_sample():
    rows = read_rows(SAMPLE)
    assert len(rows) == 84
    assert {r.label for r in rows} == {"billing", "technical", "account", "shipping"}
    assert any(r.outcome == 0.0 for r in rows)
    assert rows[0].meta == {"channel": "email"}


def test_read_rows_custom_columns_and_missing_outcome(tmp_path):
    p = write_csv(tmp_path / "d.csv", [["hello there", "greet"], ["", "x"], ["bye", ""]],
                  ["message", "category"])
    rows = read_rows(p, text_col="message", label_col="category")
    assert len(rows) == 1 and rows[0].outcome == 1.0 and rows[0].label == "greet"


def test_read_rows_errors(tmp_path):
    p = write_csv(tmp_path / "d.csv", [["a", "b"]], ["foo", "bar"])
    with pytest.raises(ValueError, match="no column 'text'"):
        read_rows(p)
    p = write_csv(tmp_path / "e.csv", [["a", "b", "great"]], ["text", "label", "outcome"])
    with pytest.raises(ValueError, match="not a number"):
        read_rows(p)


def test_model_build_save_load_decide(tmp_path):
    model = Model.build(read_rows(SAMPLE))
    assert model.head is not None
    model.save(tmp_path / "m")
    again = Model.load(tmp_path / "m")
    assert len(again.store) == 84
    for text, label in [("My card was charged two times", "billing"),
                        ("The app keeps freezing", "technical"),
                        ("Where is my parcel", "shipping")]:
        assert again.decide(text).answer == label
    assert again.decide("zzzz qqqq").answer is None
    assert again.decide("refund the double charge on my card", "head").answer == "billing"


def test_load_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a vecdecide model folder"):
        Model.load(tmp_path / "nope")


def test_single_label_has_no_head(tmp_path):
    p = write_csv(tmp_path / "d.csv", [["a b c", "only"], ["d e f", "only"]], ["text", "label"])
    model = Model.build(read_rows(p))
    assert model.head is None
    with pytest.raises(ValueError, match="no trained head"):
        model.decide("a b", "head")


def test_evaluate_beats_baseline():
    results, n_test = evaluate(read_rows(SAMPLE))
    acc = {r.method.split()[0]: r.accuracy for r in results}
    assert n_test == 79
    assert acc["store"] > acc["baseline"] + 0.25
    assert acc["head"] > acc["baseline"] + 0.25


def test_evaluate_needs_enough_data(tmp_path):
    p = write_csv(tmp_path / "d.csv", [["a", "x"], ["b", "y"]], ["text", "label"])
    with pytest.raises(ValueError, match="at least 10"):
        evaluate(read_rows(p))


def test_cli_full_workflow(tmp_path, capsys):
    m = str(tmp_path / "model")
    main(["build", str(SAMPLE), "-o", m])
    assert "Built" in capsys.readouterr().out

    main(["decide", m, "--json", "Where is my parcel"])
    out = json.loads(capsys.readouterr().out)
    assert out["answer"] == "shipping" and out["evidence"]

    main(["record", m, "The courier lost my box", "shipping"])
    capsys.readouterr()
    main(["info", m])
    assert "Cases        : 85" in capsys.readouterr().out

    inp = write_csv(tmp_path / "in.csv", [["1", "I was charged twice"], ["2", ""],
                                          ["3", "the app crashes on launch"]], ["id", "text"])
    out_csv = tmp_path / "out.csv"
    main(["batch", m, str(inp), "-o", str(out_csv)])
    with open(out_csv, newline="") as f:
        got = list(csv.DictReader(f))
    assert [r["id"] for r in got] == ["1", "2", "3"]
    assert got[0]["decision"] == "billing" and got[1]["decision"] == ""
    assert got[2]["decision"] == "technical"


def test_cli_errors_are_friendly(tmp_path):
    with pytest.raises(SystemExit) as e:
        main(["decide", str(tmp_path / "missing"), "hello"])
    assert "not a vecdecide model folder" in str(e.value)
