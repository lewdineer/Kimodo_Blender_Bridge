"""Drive ardy_bridge.py over its real stdin/stdout JSON protocol.

Runs the bridge as a real subprocess against a mock ``ardy`` package that
reproduces the released model's call contract — including the asserts
``Ardy.__call__`` makes about first_heading_angle, crop_history_length and
init_history_sequence — so the rollout, constraint windowing and frame-count
arithmetic are exercised without a GPU or 20 GB of weights.

Needs torch (an ARDY dependency); skipped when absent.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import Checker, Skip, REPO_DIR, TESTS_DIR

try:
    import torch  # noqa: F401  (only needed inside the subprocess, but the
                  #             mock model is built on it)
except ImportError as exc:                                    # pragma: no cover
    raise Skip(f"needs torch ({exc})")

PY = sys.executable
LOG = os.path.join(tempfile.mkdtemp(), "mock_calls.jsonl")
open(LOG, "w").close()
env = dict(os.environ,
           PYTHONPATH=os.path.join(TESTS_DIR, "mocks"),
           ARDY_MOCK_LOG=LOG)


def events(kind=None):
    out = []
    for line in open(LOG):
        line = line.strip()
        if line:
            e = json.loads(line)
            if kind is None or e["kind"] == kind:
                out.append(e)
    return out
proc = subprocess.Popen(
    [PY, os.path.join(REPO_DIR, "ardy_bridge.py"),
     "--model", "core", "--device", "cpu"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, bufsize=1, env=env,
)

def send(obj):
    proc.stdin.write(json.dumps(obj) + "\n"); proc.stdin.flush()

def read_until(*stop):
    msgs = []
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        m = json.loads(line)
        msgs.append(m)
        if m.get("status") in stop:
            return msgs
    raise RuntimeError(f"bridge closed before {stop}; stderr:\n{proc.stderr.read()}")

def bvh_frames(path):
    txt = open(path).read()
    return int(re.search(r"Frames:\s*(\d+)", txt).group(1)), \
           float(re.search(r"Frame Time:\s*([\d.]+)", txt).group(1))

check = Checker("ardy_bridge: JSON protocol, rollout and error handling")

# --- startup -------------------------------------------------------------
msgs = read_until("ready", "error")
ready = msgs[-1]
check(ready["status"] == "ready", "bridge reports ready")
check(ready["fps"] == 20.0, "reports the model's real FPS (20, not 30)")
check(ready["skeleton"] == "cskel27", "reports the skeleton name")
check(ready["supports_tpose"] is False, "cskel27 correctly reports no standard T-pose")
check(ready["max_diffusion_steps"] == 32, "reports the diffusion schedule ceiling")

# --- ping ----------------------------------------------------------------
send({"cmd": "ping"})
check(read_until("pong")[-1]["status"] == "pong", "ping/pong")

# --- single clip ---------------------------------------------------------
# single clip
send({"cmd": "generate", "prompt": "a person jogs in a circle", "duration": 3.0,
      "seed": 7, "output_format": "bvh", "diffusion_steps": 999,
      "bvh_standard_tpose": True})
done = read_until("done", "error")[-1]
check(done["status"] == "done", f"generate succeeded ({done.get('message','')})")
n, ft = bvh_frames(done["path"])
check(n == 60, f"3 s at 20 FPS -> 60 frames (got {n})")
check(abs(ft - 0.05) < 1e-9, f"Frame Time is 1/20 s (got {ft})")

CALLS = events("call")
check(CALLS[-1]["steps"] == 32, "diffusion_steps clamped to the schedule ceiling")
check(CALLS[-1]["crop"] == 160, f"history cropped to the 10 s window budget (got {CALLS[-1]['crop']})")
check(CALLS[-1]["cfg"] == [2.0, 2.0], f"default cfg_weight is (2.0, 2.0) (got {CALLS[-1]['cfg']})")

# --- constraints ---------------------------------------------------------
# constraints
cons = json.dumps([{"type": "root2d", "frame_indices": [0, 30],
                    "smooth_root_2d": [[0, 0], [1, 1]]}])
send({"cmd": "generate", "prompt": "walk", "duration": 2.0, "seed": 1,
      "output_format": "bvh", "constraints_json": cons})
done = read_until("done", "error")[-1]
check(done["status"] == "done", "generate with constraints succeeded")
check(events("call")[-1]["constrained"] is True, "constraints reached the model as motion_mask")
PP = events("postprocess")
check(len(PP) > 0 and PP[-1]["constraints"] == 1, "post-processing received the constraint list")

# --- heading units --------------------------------------------------------
# The add-on emits headings as [cos, sin] (Kimodo's format); ARDY wants
# radians and stacks cos/sin itself. Passing the pairs through untouched made
# the model die with "value tensor of shape [N, 2, 2] cannot be broadcast to
# indexing result of shape [N, 2]" -- for streaming and for a plain generate
# with a heading-bearing Root XZ constraint alike.
import math as _math
angles = [0.0, _math.pi / 2, -_math.pi / 3]
cons = json.dumps([{"type": "root2d",
                    "frame_indices": [0, 10, 20],
                    "smooth_root_2d": [[0, 0], [1, 1], [2, 2]],
                    "global_root_heading": [[_math.cos(a), _math.sin(a)] for a in angles]}])
send({"cmd": "generate", "prompt": "walk", "duration": 2.0, "seed": 1,
      "output_format": "bvh", "constraints_json": cons})
done = read_until("done", "error")[-1]
check(done["status"] == "done",
      f"[cos, sin] headings are accepted ({done.get('message','')})")
check(events("call")[-1]["constrained"] is True,
      "  and still reach the model as conditioning")

# Radians must pass through untouched -- the live stream already sends those.
cons = json.dumps([{"type": "root2d",
                    "frame_indices": [0, 10],
                    "smooth_root_2d": [[0, 0], [1, 1]],
                    "global_root_heading": [0.0, 1.5]}])
send({"cmd": "generate", "prompt": "walk", "duration": 2.0, "seed": 1,
      "output_format": "bvh", "constraints_json": cons})
check(read_until("done", "error")[-1]["status"] == "done",
      "radian headings pass through unchanged")

# --- bad constraint JSON must not kill the job ---------------------------
# malformed constraints
send({"cmd": "generate", "prompt": "walk", "duration": 1.0, "seed": 1,
      "output_format": "bvh", "constraints_json": "{not json"})
msgs = read_until("done", "error")
check(msgs[-1]["status"] == "done", "malformed constraint JSON degrades to a warning")
check(any("constraints skipped" in m.get("message", "") for m in msgs),
      "…and the warning is reported to the user")

# --- timeline ------------------------------------------------------------
# timeline
before = len(events("call"))
send({"cmd": "generate_multi",
      "prompts": ["a person walks", "a person jumps", "a person waves"],
      "durations": [2.0, 1.5, 2.5], "seed": 3, "output_format": "bvh"})
done = read_until("done", "error")[-1]
check(done["status"] == "done", f"generate_multi succeeded ({done.get('message','')})")
n, _ = bvh_frames(done["path"])
check(n == 40 + 30 + 50, f"segment durations preserved exactly -> 120 frames (got {n})")
chain = events("call")[before:]
check(chain[0]["hist"] == 0, "first segment starts with no history")
check(all(c["hist"] > 0 for c in chain[1:]),
      "later segments carry history in (init_history_sequence)")
check([c["text"] for c in chain][:1] == ["a person walks."],
      "prompts are normalised with a trailing period")
check(all(c["crop"] is None for c in chain[1:]),
      "crop_history_length omitted when init history is used (they are exclusive)")

# --- npz path ------------------------------------------------------------
# npz
send({"cmd": "generate", "prompt": "walk", "duration": 1.0, "seed": 1,
      "output_format": "npz"})
done = read_until("done", "error")[-1]
check(done["status"] == "done" and done["path"].endswith(".npz"), "npz output written")

# --- errors --------------------------------------------------------------
# error handling
send({"cmd": "generate_multi", "prompts": ["a"], "durations": []})
check(read_until("error")[-1]["status"] == "error", "mismatched prompts/durations -> error")
send({"cmd": "nonsense"})
check(read_until("error")[-1]["status"] == "error", "unknown command -> error")
proc.stdin.write("{bad json\n"); proc.stdin.flush()
check(read_until("error")[-1]["status"] == "error", "bad JSON line -> error, loop survives")
send({"cmd": "ping"})
check(read_until("pong")[-1]["status"] == "pong", "bridge still alive after errors")

# --- live streaming ------------------------------------------------------
# A stream_step before stream_begin must be refused, not crash the loop.
send({"cmd": "stream_step", "frame_idx": 0})
check(read_until("error")[-1]["status"] == "error",
      "stream_step without stream_begin -> error")

send({"cmd": "stream_begin", "prompt": "a person walks forward", "seed": 3,
      "diffusion_steps": 6, "replan_buffer": 4, "max_speed": 1.6})
begun = read_until("stream_ready", "error")[-1]
check(begun["status"] == "stream_ready", f"stream opens ({begun.get('message','')})")
check(begun["gen_horizon_len"] == 40, "reports the generation horizon")
check(begun["fps"] == 20.0, "reports the streaming frame rate")
check(len(begun["bone_names"]) == len(begun["parents"]) == len(begun["rest_offsets"]),
      "rest skeleton tables agree in length")
check(begun["bone_names"][begun["root_idx"]] == "Hips",
      "root index points at the root bone")
check(begun["parents"][begun["root_idx"]] < 0, "the root has no parent")

# First step: no history yet, so it starts at frame 0.
send({"cmd": "stream_step", "frame_idx": 0, "target_xz": [2.0, 3.0]})
first = read_until("frames", "error")[-1]
check(first["status"] == "frames", f"first step returns frames ({first.get('message','')})")
check(first["start_index"] == 0, "the first window starts at frame 0")
check(first["frames"] == 40, f"a window is one horizon of frames (got {first['frames']})")
check(len(first["local_rot_aa"]) == first["frames"], "one rotation row per frame")
check(len(first["local_rot_aa"][0]) == len(begun["bone_names"]) * 3,
      "each row is axis-angle per joint")
check(len(first["root_positions"]) == first["frames"], "one root position per frame")

step1 = events("ar_step")[-1]
check(step1["hist"] == 0, "the first step has no history")
# If steering emitted [cos, sin] pairs the mock constraint would have raised
# before the model was ever called, exactly as the real one did.
check(step1["constrained"] is True,
      "  steering headings are radians, so conditioning was built")
check(step1["steps"] == 6, "the requested denoising steps are honoured")
check(step1["constrained"] is True, "a follow target reaches the model as conditioning")

encodes = len(events("encode_text"))

# Second step from a playhead inside the generated range: the replan must
# start just ahead of the playhead, not at the end of what exists.
send({"cmd": "stream_step", "frame_idx": 10, "target_xz": [2.0, 3.0]})
second = read_until("frames", "error")[-1]
check(second["status"] == "frames", "second step returns frames")
check(second["start_index"] == 15,
      f"replan starts at playhead + commit buffer (got {second['start_index']})")
step2 = events("ar_step")[-1]
check(step2["hist"] > 0, "the second step feeds history back in")
check(step2["hist"] % 4 == 0, "history is a whole number of tokens")
check(len(events("encode_text")) == encodes,
      "an unchanged prompt is not re-encoded")

# A new prompt must re-encode; an identical one must not.
send({"cmd": "stream_step", "frame_idx": 20, "prompt": "a person runs"})
read_until("frames", "error")
check(len(events("encode_text")) == encodes + 1, "a changed prompt is re-encoded")

# No target: the model runs unconditioned rather than being pinned in place.
send({"cmd": "stream_step", "frame_idx": 30, "target_xz": None})
read_until("frames", "error")
check(events("ar_step")[-1]["constrained"] is False,
      "no follow target means no root conditioning")

send({"cmd": "stream_end"})
check(read_until("stream_closed", "error")[-1]["status"] == "stream_closed",
      "stream closes")
send({"cmd": "stream_step", "frame_idx": 0})
check(read_until("error")[-1]["status"] == "error",
      "stepping after stream_end -> error, state really was dropped")

send({"cmd": "ping"})
check(read_until("pong")[-1]["status"] == "pong", "bridge alive after a stream")

send({"cmd": "quit"})
read_until("bye")
proc.wait(timeout=10)


def run():
    return check.done()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
