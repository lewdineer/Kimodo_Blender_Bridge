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

send({"cmd": "quit"})
read_until("bye")
proc.wait(timeout=10)


def run():
    return check.done()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
