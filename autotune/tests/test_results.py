"""Result persistence — where autotune results land."""
import json
import os

from ampere_autotune.half_a.results import save_report, state_dir


def test_state_dir_respects_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert state_dir() == os.path.join(str(tmp_path), "ampere-autotune")


def test_save_report_to_explicit_file(tmp_path):
    p = save_report("mtp-sweep", "hello report", data={"k": [0, 1]}, output=str(tmp_path / "r.txt"))
    assert p == str(tmp_path / "r.txt")
    assert open(p).read().strip() == "hello report"
    assert json.load(open(str(tmp_path / "r.json")))["k"] == [0, 1]   # sibling .json


def test_save_report_to_dir_gets_timestamped_name(tmp_path):
    p = save_report("auto", "x", output=str(tmp_path))
    assert p.startswith(str(tmp_path)) and p.endswith("-auto.txt")


def test_save_report_default_is_xdg_state(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    p = save_report("sweep", "y")
    assert p.startswith(os.path.join(str(tmp_path), "ampere-autotune", "results")) and os.path.exists(p)
