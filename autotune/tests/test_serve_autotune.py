"""`vllm serve --autotune` glue — pure (argv reconstruction, no server)."""
from ampere_autotune.serve_autotune import strip_autotune, base_serve_restart_cmd, AUTOTUNE_FLAGS


def test_strip_autotune_separate_values():
    base = strip_autotune(["--model", "X", "--tensor-parallel-size", "2",
                           "--autotune", "--autotune-objective", "latency", "--autotune-mtp"])
    assert base == ["--model", "X", "--tensor-parallel-size", "2"]   # autotune flags + their values gone


def test_strip_autotune_equals_form():
    base = strip_autotune(["--model=X", "--autotune-scenario=code", "--autotune", "--gpu-memory-utilization=0.9"])
    assert base == ["--model=X", "--gpu-memory-utilization=0.9"]


def test_strip_autotune_keeps_unrelated_flags():
    base = strip_autotune(["--autotune", "--max-model-len", "4096", "--enforce-eager"])
    assert base == ["--max-model-len", "4096", "--enforce-eager"]


def test_base_restart_cmd_relaunches_serve_with_flags_placeholder():
    cmd = base_serve_restart_cmd(["--model", "X", "--tensor-parallel-size", "2"])
    assert "vllm serve --model X --tensor-parallel-size 2 {flags}" in cmd
    assert "{flags}" in cmd                          # the autotuner appends swept flags here
    assert "kill $(cat" in cmd and "echo $! >" in cmd  # pidfile: kills only the prior CHILD, not the orchestrator
    assert "pkill" not in cmd                         # never pkill 'vllm serve' (would kill the orchestrator)


def test_autotune_master_flag_present():
    assert AUTOTUNE_FLAGS["--autotune"] is False     # store_true (no value)
    assert AUTOTUNE_FLAGS["--autotune-objective"] is True   # takes a value
