#!/usr/bin/env python3
# Multimodal probe for the release matrix, over the OpenAI API.
# Stdlib only: the host that drives the matrix has no PIL, so the test images are drawn here.
# Every answer is a colour, shape or count the model cannot get right without seeing the pixels,
# and two cases share a prompt byte-for-byte but differ in the image -- a prefix cache that ignored
# the image would answer the second from the first.
#
# Usage: mm_probe.py PORT battery [--video FILE]   (FILE = base64 of an mp4)
#        mm_probe.py PORT concurrency [WORKERS] [REQUESTS]
import base64
import concurrent.futures as cf
import json
import math
import struct
import sys
import urllib.error
import urllib.request
import zlib

RED, BLUE, GREEN = (220, 30, 30), (30, 60, 220), (30, 170, 60)
YELLOW, WHITE, BLACK = (240, 210, 20), (255, 255, 255), (0, 0, 0)


def _span(kind, y, cx, cy, r):
    dy = y - cy
    if abs(dy) > r:
        return None
    if kind == "circle":
        dx = math.isqrt(r * r - dy * dy)
    elif kind == "square":
        dx = r
    else:  # upward triangle, apex at cy-r, base at cy+r
        dx = (dy + r) // 2
    return cx - dx, cx + dx + 1


def png(w, h, bg, shapes=()):
    raw = bytearray()
    for y in range(h):
        row = bytearray(bytes(bg) * w)
        for kind, col, cx, cy, r in shapes:
            s = _span(kind, y, cx, cy, r)
            if s:
                x0, x1 = max(0, s[0]), min(w, s[1])
                if x1 > x0:
                    row[3 * x0:3 * x1] = bytes(col) * (x1 - x0)
        raw += b"\x00" + row

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b""))


def img(data, mime="image/png"):
    return {"type": "image_url",
            "image_url": {"url": f"data:{mime};base64," + base64.b64encode(data).decode()}}


def one_word(q):
    return q + " Answer with one word."


# The cache pair must span several KV blocks (hybrid Qwen3.5 blocks are ~784 tokens), or prefix
# caching never engages and the pair proves nothing. They differ only in the circle's colour.
CACHE_A = png(1792, 1344, RED, [("circle", BLUE, 1100, 600, 300)])
CACHE_B = png(1792, 1344, RED, [("circle", GREEN, 1100, 600, 300)])
BLUE_ON_RED = png(448, 448, RED, [("circle", BLUE, 224, 224, 120)])
TRIANGLE = png(448, 448, WHITE, [("triangle", GREEN, 224, 224, 140)])
THREE_SQUARES = png(448, 448, BLACK, [("square", YELLOW, x, 224, 45) for x in (84, 224, 364)])
RED_SQUARE = png(448, 448, WHITE, [("square", RED, 224, 224, 110)])
BLUE_CIRCLE = png(448, 448, WHITE, [("circle", BLUE, 224, 224, 120)])
TINY = png(32, 32, GREEN)
WIDE = png(1024, 96, WHITE, [("circle", RED, 512, 48, 40)])

# (tag, content parts, accepted answers). Order matters: cache-b must follow cache-a.
CIRCLE_Q = one_word("What color is the circle?")
CASES = [
    ("cache-a", [img(CACHE_A), {"type": "text", "text": CIRCLE_Q}], ["blue"]),
    ("cache-b", [img(CACHE_B), {"type": "text", "text": CIRCLE_Q}], ["green"]),
    ("cache-a-again", [img(CACHE_A), {"type": "text", "text": CIRCLE_Q}], ["blue"]),
    ("background", [img(BLUE_ON_RED), {"type": "text", "text": one_word("What color is the background?")}], ["red"]),
    ("shape", [img(TRIANGLE), {"type": "text", "text": one_word("What shape is drawn in this image?")}], ["triangle"]),
    ("count", [img(THREE_SQUARES), {"type": "text", "text": "How many squares are in this image? Answer with a number."}], ["3", "three"]),
    ("two-images-2nd", [img(RED_SQUARE), img(BLUE_CIRCLE), {"type": "text", "text": one_word("What color is the shape in the second image?")}], ["blue"]),
    ("two-images-1st", [img(RED_SQUARE), img(BLUE_CIRCLE), {"type": "text", "text": one_word("What color is the shape in the first image?")}], ["red"]),
    ("tiny-32x32", [img(TINY), {"type": "text", "text": one_word("What color is this image?")}], ["green"]),
    ("wide-1024x96", [img(WIDE), {"type": "text", "text": CIRCLE_Q}], ["red"]),
    ("text-only", [{"type": "text", "text": "What is 17 times 23? Answer with the number only."}], ["391"]),
]


def ask(port, content, max_tokens=48, timeout=300):
    body = json.dumps({
        "model": "test", "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    c = d["choices"][0]
    return (c["message"].get("content") or ""), c["finish_reason"], d["usage"]["prompt_tokens"]


def check(port, tag, content, want):
    try:
        text, finish, ptok = ask(port, content)
    except Exception as e:
        return "ERROR", f"{tag}: {type(e).__name__}: {str(e)[:120]}"
    ok = any(w in text.lower() for w in want)
    return ("PASS" if ok else "FAIL"), f"{tag}: ptok={ptok} finish={finish} {text.strip()[:60]!r}"


def healthy(port):
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=30).status == 200
    except Exception:
        return False


def prefix_hits(port):
    try:
        t = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30).read().decode()
    except Exception:
        return None
    return sum(float(l.split()[-1]) for l in t.splitlines()
               if l.startswith("vllm:prefix_cache_hits_total"))


def battery(port, video_b64=None):
    cases = list(CASES)
    if video_b64:
        video = {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + video_b64}}
        cases.append(("video", [video, {"type": "text", "text": one_word("What color is the circle in this video?")}], ["yellow"]))
    tally = {"PASS": 0, "FAIL": 0, "ERROR": 0}
    reused = {}
    for tag, content, want in cases:
        before = prefix_hits(port)
        verdict, line = check(port, tag, content, want)
        after = prefix_hits(port)
        if before is not None and after is not None:
            reused[tag] = int(after - before)
        tally[verdict] += 1
        print(f"  {verdict} {line}")
    # cache-b differs from cache-a only in the image: it must reuse nothing. cache-a-again must reuse,
    # otherwise prefix caching never engaged and the pair tested nothing.
    b, again = reused.get("cache-b"), reused.get("cache-a-again")
    if b is None or again is None:
        cache = "UNKNOWN (no /metrics)"
    elif b > 0:
        cache = "POISONED"
    elif again == 0:
        cache = "NOT EXERCISED"
    else:
        cache = "PASS"
    print(f"  prefix cache {cache}: cache-b reused {b} tokens (must be 0),"
          f" cache-a-again reused {again} (must be > 0)")
    # A corrupt image must be rejected per request, not take the engine down.
    bad = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,bm90IGFuIGltYWdl"}},
           {"type": "text", "text": "Describe this image."}]
    try:
        ask(port, bad)
        status = "200 (accepted a corrupt image)"
    except urllib.error.HTTPError as e:
        status = str(e.code)
    except Exception as e:
        status = f"{type(e).__name__}"
    alive = healthy(port) and check(port, "after-bad-input", CASES[-1][1], CASES[-1][2])[0] == "PASS"
    print(f"  corrupt image -> HTTP {status}; engine healthy afterwards: {alive}")
    n = len(cases)
    print(f"  mm battery: pass {tally['PASS']}/{n}  wrong {tally['FAIL']}/{n}  errors {tally['ERROR']}/{n}")


def concurrency(port, workers=8, requests=48):
    # Mixed text / single-image / two-image / large-image traffic so continuous batching folds image
    # prefills together with decode rows of other requests.
    mix = [c for c in CASES if c[0] in ("cache-a", "cache-b", "shape", "count", "two-images-2nd",
                                        "tiny-32x32", "text-only")]
    jobs = [mix[i % len(mix)] for i in range(requests)]
    with cf.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda c: check(port, *c), jobs))
    errs = [line for v, line in res if v == "ERROR"]
    wrong = [line for v, line in res if v == "FAIL"]
    print(f"  concurrency W={workers}: {requests} requests, {len(errs)} errors, {len(wrong)} wrong,"
          f" healthy afterwards: {healthy(port)}")
    for line in (errs + wrong)[:4]:
        print(f"    {line}")


if __name__ == "__main__":
    port, mode = sys.argv[1], sys.argv[2]
    if mode == "battery":
        video = None
        if "--video" in sys.argv:
            try:
                video = open(sys.argv[sys.argv.index("--video") + 1]).read().strip() or None
            except OSError:
                pass
            if not video:
                print("  video: SKIPPED (no clip was generated)")
        battery(port, video)
    elif mode == "concurrency":
        concurrency(port, *[int(a) for a in sys.argv[3:5]])
    else:
        sys.exit(f"unknown mode {mode}")
