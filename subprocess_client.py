"""
Kimodo Blender Bridge — Subprocess Client

Runs inside Blender's Python. Manages the bridge_server.py subprocess
which runs under the Kimodo venv Python (with PyTorch, Kimodo, etc.).

Communication: newline-delimited JSON over stdin / stdout.

Threading model
---------------
A dedicated reader thread drains the bridge's stdout into a queue so that
waiting for messages never blocks indefinitely (readline() on a silent
process would otherwise hang forever and defeat every timeout).  All
public waiting functions poll the queue with short timeouts.

One request may be in flight on the pipe at a time (_busy).  Cancelling
does not abort the bridge's computation (diffusion can't be interrupted
mid-step); instead the in-flight response is drained and discarded in the
background so the next request never reads a stale message.
"""

import json
import os
import queue
import subprocess
import threading
import time


# ---------------------------------------------------------------------------
# Module-level process state (one bridge process per Blender session)
# ---------------------------------------------------------------------------

_proc: "subprocess.Popen | None" = None
_stdout_queue: "queue.Queue | None" = None
_lock = threading.Lock()
_status = "Not started"
_ready  = False
_busy   = False              # a request is in flight on the pipe
_cancel_requested = False

# Prevent a console window from flashing up for the subprocess on Windows
# (Blender is a GUI process; child console apps get their own window
# unless CREATE_NO_WINDOW is passed).
_NO_WINDOW = (
    {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
)


def _bridge_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bridge_server.py")


def _send(obj: dict) -> None:
    proc = _proc
    if proc is None or proc.poll() is not None:
        raise RuntimeError("Bridge is not running")
    proc.stdin.write(json.dumps(obj) + "\n")
    proc.stdin.flush()


def _recv(timeout: float = 0.1) -> "dict | None":
    """Pop one parsed JSON message from the stdout queue, or None on timeout."""
    q = _stdout_queue
    if q is None:
        return None
    try:
        raw = q.get(timeout=timeout)
    except queue.Empty:
        return None
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return None


def _read_stdout(pipe, q: "queue.Queue") -> None:
    """Reader thread: drain bridge stdout lines into the queue until EOF."""
    for line in pipe:
        if line.strip():
            q.put(line)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start(python_exe: str, model_name: str, progress_callback=None) -> "tuple[bool, str]":
    """
    Launch bridge_server.py and block until the model reports ready.
    Must be called from a background thread — model loading takes 1-3 min.
    Returns (success, status_message).
    """
    global _proc, _stdout_queue, _status, _ready, _busy, _cancel_requested

    with _lock:
        if _proc is not None and _proc.poll() is None:
            return True, _status  # already running

        bridge = _bridge_path()
        if not os.path.isfile(bridge):
            return False, f"bridge_server.py not found at: {bridge}"

        python = _resolve_python(python_exe)
        _ready  = False
        _busy   = False
        _cancel_requested = False
        _status = "Launching…"

        try:
            _proc = subprocess.Popen(
                [python, bridge, "--model", model_name],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,      # line-buffered
                env=_bridge_env(python),
                **_NO_WINDOW,
            )
        except FileNotFoundError:
            _proc = None
            return False, f"Python executable not found: {python}"
        except Exception as exc:
            _proc = None
            return False, f"Failed to launch bridge: {exc}"

        _stdout_queue = queue.Queue()

    # Drain stdout into the queue so waiting below can use real timeouts
    # (a plain readline() would block forever on a silent, hung process).
    threading.Thread(
        target=_read_stdout, args=(_proc.stdout, _stdout_queue), daemon=True,
    ).start()

    # Stream stderr to the console in a background thread so errors from
    # PyTorch / Kimodo are visible without blocking the stdout reader.
    def _drain(pipe):
        for line in pipe:
            line = line.rstrip()
            if line:
                print(f"[Kimodo Bridge] {line}", flush=True)
    threading.Thread(target=_drain, args=(_proc.stderr,), daemon=True).start()

    # Wait for "ready" or "error"
    deadline = time.monotonic() + 420   # 7-min ceiling (large models, slow GPU)
    while time.monotonic() < deadline:
        msg = _recv(timeout=0.5)

        if msg is None:
            if _proc.poll() is not None:
                # Give the stderr drain thread a moment to flush remaining lines
                time.sleep(0.2)
                _status = f"Process exited early (code {_proc.returncode}) — see console for details"
                print(f"[Kimodo Bridge] ERROR: {_status}", flush=True)
                return False, _status
            continue

        s = msg.get("status", "")

        if s == "loading":
            _status = msg.get("message", "Loading…")
            print(f"[Kimodo Bridge] {_status}", flush=True)
            if progress_callback:
                progress_callback(_status)

        elif s == "ready":
            _ready  = True
            _status = (
                f"Ready — {msg.get('model', model_name)} "
                f"on {msg.get('device', '?')} "
                f"({msg.get('fps', '?')} fps)"
            )
            print(f"[Kimodo Bridge] {_status}", flush=True)
            return True, _status

        elif s == "error":
            err = msg.get("message", "Unknown error")
            _status = f"Failed: {err}"
            print(f"[Kimodo Bridge] ERROR: {_status}", flush=True)
            stop()
            return False, _status

        else:
            print(f"[Kimodo Bridge] {msg}", flush=True)

    stop()
    return False, "Timed out waiting for Kimodo (>7 min)"


def stop() -> None:
    global _proc, _stdout_queue, _ready, _status, _busy, _cancel_requested
    with _lock:
        if _proc is not None:
            try:
                _send({"cmd": "quit"})
            except Exception:
                pass
            try:
                _proc.terminate()
                _proc.wait(timeout=5)
            except Exception:
                try:
                    _proc.kill()
                except Exception:
                    pass
            _proc = None
        _stdout_queue = None
        _ready  = False
        _busy   = False
        _cancel_requested = False
        _status = "Stopped"


def is_running() -> bool:
    proc = _proc
    return proc is not None and proc.poll() is None


def is_busy() -> bool:
    """True while a request is in flight on the pipe (including a cancelled
    one that is still being drained in the background)."""
    return _busy


def get_status() -> str:
    return _status


def request_cancel() -> None:
    """Abandon the in-flight request. The bridge cannot abort mid-diffusion,
    so the eventual response is drained and discarded in the background;
    the pipe stays busy until then."""
    global _cancel_requested
    _cancel_requested = True


def _drain_until_done() -> None:
    """After a cancel, keep consuming responses for the abandoned job so the
    next request never reads its stale 'done'/'error' message."""
    global _busy
    while is_running():
        msg = _recv(timeout=0.5)
        if msg is None:
            continue
        if msg.get("status") in ("done", "error"):
            # Discard the abandoned output file if one was produced.
            path = msg.get("path", "")
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            break
    with _lock:
        _busy = False


def _recv_until_done(progress_callback) -> "tuple[bool, str]":
    """Read responses from bridge_server until 'done' or 'error'. Shared by generate functions."""
    global _busy
    while True:
        if _cancel_requested:
            # Hand the rest of this job's messages to a background drainer
            # and return immediately so the UI is released.
            threading.Thread(target=_drain_until_done, daemon=True).start()
            return False, "Cancelled by user"

        if not is_running():
            with _lock:
                _busy = False
            return False, "Kimodo process died during generation."

        msg = _recv(timeout=0.2)
        if msg is None:
            continue

        s = msg.get("status", "")

        if s == "progress":
            if progress_callback:
                progress_callback(msg.get("message", ""))

        elif s == "done":
            with _lock:
                _busy = False
            path = msg.get("path", "")
            if not path or not os.path.isfile(path):
                return False, f"Output file not found: {path}"
            return True, path

        elif s == "error":
            with _lock:
                _busy = False
            return False, msg.get("message", "Generation failed")


def _begin_request(req: dict) -> "str | None":
    """Mark the pipe busy and send the request. Returns an error message on
    failure, None on success."""
    global _busy, _cancel_requested
    with _lock:
        if _busy:
            return ("Kimodo is still finishing the previous request — "
                    "wait for it to complete and try again.")
        _busy = True
        _cancel_requested = False
    try:
        _send(req)
    except Exception as exc:
        with _lock:
            _busy = False
        return f"Failed to send request: {exc}"
    return None


def generate_motion(
    prompt: str,
    duration: float,
    seed: int,
    output_format: str,
    constraints_json: "str | None" = None,
    diffusion_steps: int = 100,
    bvh_standard_tpose: bool = False,
    progress_callback=None,
) -> "tuple[bool, str]":
    """
    Send one generation request. Blocks until done or error.
    Must be called from a background thread.
    Returns (success, file_path_or_error_message).
    """
    if not is_running():
        return False, "Kimodo is not running — click 'Start Kimodo' first."

    req = {
        "cmd": "generate",
        "prompt": prompt,
        "duration": duration,
        "seed": seed if seed >= 0 else None,
        "output_format": output_format,
        "constraints_json": constraints_json,
        "diffusion_steps": diffusion_steps,
        "bvh_standard_tpose": bvh_standard_tpose,
    }

    err = _begin_request(req)
    if err:
        return False, err

    return _recv_until_done(progress_callback)


def generate_motion_multi(
    prompts: "list[str]",
    durations: "list[float]",
    seed: int,
    output_format: str,
    constraints_json: "str | None" = None,
    diffusion_steps: int = 100,
    num_transition_frames: int = 5,
    bvh_standard_tpose: bool = False,
    progress_callback=None,
) -> "tuple[bool, str]":
    """
    Generate a single continuous motion from multiple prompts in one model call.
    Kimodo transitions smoothly between prompts using num_transition_frames.
    Blocks until done or error. Must be called from a background thread.
    Returns (success, file_path_or_error_message).
    """
    if not is_running():
        return False, "Kimodo is not running — click 'Start Kimodo' first."

    req = {
        "cmd": "generate_multi",
        "prompts": prompts,
        "durations": durations,
        "seed": seed if seed >= 0 else None,
        "output_format": output_format,
        "constraints_json": constraints_json,
        "diffusion_steps": diffusion_steps,
        "num_transition_frames": num_transition_frames,
        "bvh_standard_tpose": bvh_standard_tpose,
    }

    err = _begin_request(req)
    if err:
        return False, err

    return _recv_until_done(progress_callback)


# ---------------------------------------------------------------------------
# Bridge environment
# ---------------------------------------------------------------------------

def _bridge_env(python_exe: str) -> dict:
    """
    Build the environment dict for the bridge subprocess.
    When the managed venv is in use and its LLM2Vec model has been downloaded,
    set the HuggingFace offline flags so the bridge never tries to reach the
    internet (load_model calls snapshot_download unconditionally; the weights
    were pre-downloaded into the HF cache by the installer).
    """
    env = os.environ.copy()
    try:
        from . import setup_operator as _so
        managed = _so.MANAGED_VENV
        llmvec  = _so.LLMVEC_DIR
    except ImportError:
        managed = os.path.join(os.path.expanduser("~"), ".kimodo-venv")
        llmvec  = os.path.join(managed, "llm2vec-model")

    using_managed = os.path.realpath(python_exe).startswith(
        os.path.realpath(managed) + os.sep
    )
    if using_managed and os.path.isdir(llmvec):
        env["TRANSFORMERS_OFFLINE"]  = "1"
        env["HF_DATASETS_OFFLINE"]   = "1"
        env["HF_HUB_OFFLINE"]        = "1"

    return env


# ---------------------------------------------------------------------------
# Python executable resolution
# ---------------------------------------------------------------------------

# Relative paths of the Python binary inside a venv / conda env root.
# Conda on Windows puts python.exe at the env root, not in Scripts/.
_PYTHON_SUBPATHS = ("bin/python3", "bin/python", "Scripts/python.exe", "python.exe")


def _resolve_python(hint: str) -> str:
    """
    Find a Python executable from the user's hint, auto-detecting common
    patterns like venv roots, sibling venvs, and kimodo_gen on PATH.
    """
    import shutil

    hint = (hint or "").strip()

    # Direct path to an executable
    if hint and os.path.isfile(hint):
        return hint

    # Path to a venv / conda env root — pick the python inside
    if hint and os.path.isdir(hint):
        for rel in _PYTHON_SUBPATHS:
            p = os.path.join(hint, rel)
            if os.path.isfile(p):
                return p

    # Look for a venv sitting next to (or near) the addon directory
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    for rel_venv in ("../venv", "../../venv", "../.venv", "../../.venv"):
        venv_root = os.path.normpath(os.path.join(addon_dir, rel_venv))
        for sub in _PYTHON_SUBPATHS:
            p = os.path.join(venv_root, sub)
            if os.path.isfile(p):
                return p

    # kimodo_gen on PATH → its sibling Python is the right one
    kimodo_gen = shutil.which("kimodo_gen")
    if kimodo_gen:
        bin_dir = os.path.dirname(kimodo_gen)
        for name in ("python3", "python"):
            p = os.path.join(bin_dir, name)
            if os.path.isfile(p):
                return p

    # Last resort: whatever python3 / python is on PATH
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            return found

    return "python3"
