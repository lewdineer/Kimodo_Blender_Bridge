"""
Kimodo Blender Bridge — Subprocess Client

Runs inside Blender's Python. Manages the bridge subprocess, which runs under
a managed venv Python (with PyTorch and the generation model).

Two backends speak the same protocol and are therefore interchangeable here:

    kimodo -> bridge_server.py  in the Kimodo venv  (SOMA skeleton, 30 FPS)
    ardy   -> ardy_bridge.py    in the ARDY venv    (Core skeleton, 20 FPS)

Only which script and which Python get launched differs; everything below —
framing, threading, cancellation — is shared.

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
_backend = "kimodo"          # which bridge the running process is
_ready_info: dict = {}       # the bridge's "ready" message (fps, skeleton, …)

# Prevent a console window from flashing up for the subprocess on Windows
# (Blender is a GUI process; child console apps get their own window
# unless CREATE_NO_WINDOW is passed).
_NO_WINDOW = (
    {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
)


# Bridge script per backend, resolved next to this file.
_BRIDGE_SCRIPTS = {
    "kimodo": "bridge_server.py",
    "ardy":   "ardy_bridge.py",
}

# Shown in the system console so two backends' logs are never confused.
_LOG_PREFIX = {"kimodo": "[Kimodo Bridge]", "ardy": "[ARDY Bridge]"}


def _bridge_path(backend: str = "kimodo") -> str:
    script = _BRIDGE_SCRIPTS.get(backend, _BRIDGE_SCRIPTS["kimodo"])
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), script)


def get_backend() -> str:
    """Which backend the running (or last-run) bridge is."""
    return _backend


def get_ready_info() -> dict:
    """The bridge's 'ready' message: fps, skeleton, device, and backend extras.

    Empty until a bridge reports ready. The add-on reads 'fps' and 'skeleton'
    from here rather than assuming Kimodo's 30 FPS / SOMA.
    """
    return dict(_ready_info)


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

def start(python_exe: str, model_name: str, use_offload: bool = False,
          progress_callback=None, backend: str = "kimodo",
          extra_args: "list[str] | None" = None) -> "tuple[bool, str]":
    """
    Launch the backend's bridge and block until the model reports ready.
    Must be called from a background thread — model loading takes 1-3 min.
    Returns (success, status_message).
    """
    global _proc, _stdout_queue, _status, _ready, _busy, _cancel_requested
    global _backend, _ready_info

    with _lock:
        if _proc is not None and _proc.poll() is None:
            return True, _status  # already running

        backend = backend if backend in _BRIDGE_SCRIPTS else "kimodo"
        _backend = backend
        _ready_info = {}
        label = "ARDY" if backend == "ardy" else "Kimodo"

        bridge = _bridge_path(backend)
        if not os.path.isfile(bridge):
            return False, f"{os.path.basename(bridge)} not found at: {bridge}"

        python = _resolve_python(python_exe, backend)

        # Kimodo only: a moved/renamed venv leaves an absolute path baked into
        # llm2vec_wrapper.py pointing at the old location; generation then
        # fails with a HuggingFace "Repo id must be in the form…" error.
        # Repair it in place so a relocated Kimodo venv just works. ARDY needs
        # no equivalent — its wrapper reads TEXT_ENCODERS_DIR from the
        # environment instead of hardcoding a path.
        if backend == "kimodo":
            try:
                from . import setup_operator as _so
                if _so.is_kimodo_venv(python):
                    _so.heal_wrapper_path(python)
            except Exception:
                pass

        _ready  = False
        _busy   = False
        _cancel_requested = False
        _status = "Launching…"

        try:
            cmd = [python, bridge, "--model", model_name]
            if backend == "kimodo":
                if use_offload:
                    cmd.append("--offload")
            elif extra_args:
                cmd += list(extra_args)
            _proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,      # line-buffered
                env=_bridge_env(python, backend),
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
                print(f"{_LOG_PREFIX.get(backend, '[Bridge]')} {line}", flush=True)
    threading.Thread(target=_drain, args=(_proc.stderr,), daemon=True).start()

    prefix = _LOG_PREFIX.get(backend, "[Bridge]")

    # Wait for "ready" or "error"
    deadline = time.monotonic() + 420   # 7-min ceiling (large models, slow GPU)
    while time.monotonic() < deadline:
        msg = _recv(timeout=0.5)

        if msg is None:
            if _proc.poll() is not None:
                # Give the stderr drain thread a moment to flush remaining lines
                time.sleep(0.2)
                _status = f"Process exited early (code {_proc.returncode}) — see console for details"
                print(f"{prefix} ERROR: {_status}", flush=True)
                return False, _status
            continue

        s = msg.get("status", "")

        if s == "loading":
            _status = msg.get("message", "Loading…")
            print(f"{prefix} {_status}", flush=True)
            if progress_callback:
                progress_callback(_status)

        elif s == "ready":
            _ready  = True
            _ready_info = dict(msg)
            _status = (
                f"Ready — {msg.get('model', model_name)} "
                f"on {msg.get('device', '?')} "
                f"({msg.get('fps', '?')} fps)"
            )
            print(f"{prefix} {_status}", flush=True)
            return True, _status

        elif s == "error":
            err = msg.get("message", "Unknown error")
            err_message = f"Failed: {err}"
            print(f"{prefix} ERROR: {err_message}", flush=True)
            stop()  # resets _status to "Stopped" — use local var
            return False, err_message

        else:
            print(f"{prefix} {msg}", flush=True)

    stop()
    return False, f"Timed out waiting for {label} (>7 min)"


def stop() -> None:
    global _proc, _stdout_queue, _ready, _status, _busy, _cancel_requested
    global _ready_info
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
        _ready_info = {}
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
            label = "ARDY" if _backend == "ardy" else "Kimodo"
            return False, f"{label} process died during generation."

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


def _recv_until_status(terminal, progress_callback=None,
                       timeout: float = 300.0) -> "tuple[bool, dict | str]":
    """Read until a message whose status is in *terminal*; returns (ok, msg).

    The sibling of _recv_until_done for commands whose reply is data rather
    than a file path — the streaming ones. On failure the second element is an
    error string instead of the message.

    Unlike _recv_until_done this does not watch _cancel_requested: cancellation
    belongs to a long generate, whereas a stream is ended by its own operator
    calling stream_end, and a step is short enough to simply finish.
    """
    global _busy
    deadline = time.monotonic() + timeout
    while True:
        if not is_running():
            with _lock:
                _busy = False
            label = "ARDY" if _backend == "ardy" else "Kimodo"
            return False, f"{label} process died."

        if time.monotonic() > deadline:
            with _lock:
                _busy = False
            return False, f"Timed out after {timeout:.0f}s waiting for the bridge."

        msg = _recv(timeout=0.2)
        if msg is None:
            continue

        s = msg.get("status", "")
        if s == "progress":
            if progress_callback:
                progress_callback(msg.get("message", ""))
        elif s == "error":
            with _lock:
                _busy = False
            return False, msg.get("message", "Bridge reported an error")
        elif s in terminal:
            with _lock:
                _busy = False
            return True, msg


def _begin_request(req: dict) -> "str | None":
    """Mark the pipe busy and send the request. Returns an error message on
    failure, None on success."""
    global _busy, _cancel_requested
    with _lock:
        if _busy:
            label = "ARDY" if _backend == "ardy" else "Kimodo"
            return (f"{label} is still finishing the previous request — "
                    f"wait for it to complete and try again.")
        _busy = True
        _cancel_requested = False
    try:
        _send(req)
    except Exception as exc:
        with _lock:
            _busy = False
        return f"Failed to send request: {exc}"
    return None


def _not_running_message() -> str:
    label = "ARDY" if _backend == "ardy" else "Kimodo"
    return f"{label} is not running — click 'Start {label}' first."


def _ardy_extras(cfg_text_weight, cfg_constraint_weight, history_frames) -> dict:
    """ARDY-only request fields.

    Always sent: bridge_server.py reads its request with req.get(), so the
    Kimodo backend simply ignores keys it does not know.
    """
    return {
        "cfg_text_weight": cfg_text_weight,
        "cfg_constraint_weight": cfg_constraint_weight,
        # 0 / None means "let the bridge pick its window budget"
        "history_frames": history_frames or None,
    }


def generate_motion(
    prompt: str,
    duration: float,
    seed: int,
    output_format: str,
    constraints_json: "str | None" = None,
    diffusion_steps: int = 100,
    bvh_standard_tpose: bool = False,
    progress_callback=None,
    cfg_text_weight: float = 2.0,
    cfg_constraint_weight: float = 2.0,
    history_frames: int = 0,
) -> "tuple[bool, str]":
    """
    Send one generation request. Blocks until done or error.
    Must be called from a background thread.
    Returns (success, file_path_or_error_message).
    """
    if not is_running():
        return False, _not_running_message()

    req = {
        "cmd": "generate",
        "prompt": prompt,
        "duration": duration,
        "seed": seed if seed >= 0 else None,
        "output_format": output_format,
        "constraints_json": constraints_json,
        "diffusion_steps": diffusion_steps,
        "bvh_standard_tpose": bvh_standard_tpose,
        **_ardy_extras(cfg_text_weight, cfg_constraint_weight, history_frames),
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
    seeds: "list[int] | None" = None,
    cfg_text_weight: float = 2.0,
    cfg_constraint_weight: float = 2.0,
    history_frames: int = 0,
) -> "tuple[bool, str]":
    """
    Generate a single continuous motion from multiple prompts in one model call.
    Kimodo transitions smoothly between prompts using num_transition_frames.
    Blocks until done or error. Must be called from a background thread.
    Returns (success, file_path_or_error_message).
    """
    if not is_running():
        return False, _not_running_message()

    req = {
        "cmd": "generate_multi",
        "prompts": prompts,
        "durations": durations,
        "seed": seed if seed >= 0 else None,
        "seeds": seeds,
        "output_format": output_format,
        "constraints_json": constraints_json,
        "diffusion_steps": diffusion_steps,
        "num_transition_frames": num_transition_frames,
        "bvh_standard_tpose": bvh_standard_tpose,
        **_ardy_extras(cfg_text_weight, cfg_constraint_weight, history_frames),
    }

    err = _begin_request(req)
    if err:
        return False, err

    return _recv_until_done(progress_callback)


# ---------------------------------------------------------------------------
# Live streaming (ARDY only)
# ---------------------------------------------------------------------------
#
# One request, one reply — no server push. The bridge holds the motion state
# between steps, so all that crosses the pipe is "here is the playhead and
# where the target is now", answered with one window of frames.

def stream_begin(
    prompt: str,
    seed: int,
    diffusion_steps: int,
    replan_buffer: int,
    cfg_text_weight: "float | None" = None,
    cfg_constraint_weight: "float | None" = None,
    history_frames: "int | None" = None,
    max_speed: float = 1.6,
) -> "tuple[bool, dict | str]":
    """Open a stream. Returns (True, stream_ready message) with the rest skeleton."""
    if not is_running():
        return False, "ARDY is not running — click 'Start ARDY' first."
    if _backend != "ardy":
        return False, "Live streaming needs the ARDY backend."

    req = {
        "cmd": "stream_begin",
        "prompt": prompt,
        "seed": seed if seed >= 0 else None,
        "diffusion_steps": diffusion_steps,
        "replan_buffer": replan_buffer,
        "max_speed": max_speed,
        **_ardy_extras(cfg_text_weight, cfg_constraint_weight, history_frames),
    }
    err = _begin_request(req)
    if err:
        return False, err
    return _recv_until_status({"stream_ready"})


def stream_step(
    frame_idx: int,
    target_xz: "list[float] | None" = None,
    target_heading: "float | None" = None,
    target_velocity: "list[float] | None" = None,
    prompt: "str | None" = None,
    replan_buffer: "int | None" = None,
    max_speed: "float | None" = None,
    timeout: float = 120.0,
) -> "tuple[bool, dict | str]":
    """Advance the stream one window. Returns (True, frames message).

    Must be called from a background thread: a step is a diffusion pass and
    blocking Blender's main thread on it would freeze the viewport.
    """
    if not is_running():
        return False, "ARDY is not running."

    # replan_buffer and max_speed are re-sent every step so their sliders stay
    # live during a stream, rather than being frozen at stream_begin.
    req = {
        "cmd": "stream_step",
        "frame_idx": int(frame_idx),
        "target_xz": target_xz,
        "target_heading": target_heading,
        "target_velocity": target_velocity,
        "prompt": prompt,
        "replan_buffer": replan_buffer,
        "max_speed": max_speed,
    }
    err = _begin_request(req)
    if err:
        return False, err
    return _recv_until_status({"frames"}, timeout=timeout)


def stream_end() -> "tuple[bool, dict | str]":
    """Close the stream and let the bridge drop its motion state."""
    if not is_running():
        return True, {"status": "stream_closed"}
    err = _begin_request({"cmd": "stream_end"})
    if err:
        return False, err
    return _recv_until_status({"stream_closed"}, timeout=30.0)


# ---------------------------------------------------------------------------
# Bridge environment
# ---------------------------------------------------------------------------

def _bridge_env(python_exe: str, backend: str = "kimodo") -> dict:
    """
    Build the environment dict for the bridge subprocess.

    Both backends call snapshot_download unconditionally at load time, so once
    the installer has fetched everything the HuggingFace offline flags are set
    and the bridge never touches the network again.

    ARDY additionally needs TEXT_ENCODERS_DIR, which its LLM2Vec wrapper reads
    to find the locally downloaded encoder — no source patching required.
    """
    env = os.environ.copy()

    if backend == "ardy":
        try:
            from . import ardy_setup as _as
            return _as.bridge_env(python_exe, env)
        except Exception:
            return env

    try:
        from . import setup_operator as _so
        is_kimodo = _so.is_kimodo_venv(python_exe)
        llmvec    = _so.llmvec_dir_for(python_exe)
    except ImportError:
        # Fallback: derive the venv root relative to the given executable so a
        # relocated venv still gets offline mode even without setup_operator.
        d = os.path.dirname(os.path.abspath(python_exe))
        root = os.path.dirname(d) if os.path.basename(d).lower() in ("bin", "scripts") else d
        is_kimodo = os.path.isfile(os.path.join(root, ".kimodo_install_complete"))
        llmvec    = os.path.join(root, "llm2vec-model")

    if is_kimodo and llmvec and os.path.isdir(llmvec):
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


def _resolve_python(hint: str, backend: str = "kimodo") -> str:
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

    # ARDY has its own managed venv and none of the Kimodo-specific fallbacks
    # below apply to it — guessing a system Python would only produce a
    # confusing "ARDY not found in this Python environment" at load time.
    if backend == "ardy":
        try:
            from . import ardy_setup as _as
            found = _as.managed_python()
            if found:
                return found
        except Exception:
            pass
        return hint or "python3"

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
