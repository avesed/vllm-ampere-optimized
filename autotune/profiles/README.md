# profiles/

Per-GPU validated clock profiles are an **OUTPUT**, written at runtime to
`~/.config/ampere-autotune/<gpu-uuid>.json` — **not** here, and **never** shipped as a default.

## Silicon lottery — do not copy another card's profile

A validated `max_stable_mem_offset` is specific to **one physical card** at a **given
temperature and driver**. Copying it to another card (even the same SKU) can silently
corrupt output on no-ECC GDDR6X. Profiles are re-validated on temp-delta / driver-change /
UUID-mismatch.

Any file committed here is **documentation only** (e.g. `example-3090-gddr6x.json`) and is
**never auto-loaded**.
