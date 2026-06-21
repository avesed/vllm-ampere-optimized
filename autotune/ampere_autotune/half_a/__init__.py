"""HALF-A — vLLM flag tuner (NO privilege, recommend-only).

Hard rule: nothing in half_a/ may import half_b/ or any NVML-write code. A CI test asserts
this. See DESIGN.md "Why HALF-A is offline / recommend-only".
"""
