"""
ARDY auto-installer

Creates a managed Python venv (default ~/.ardy-venv), installs NVIDIA ARDY and
a matching PyTorch build into it, downloads the LLM2Vec text encoder and the
default checkpoint, and points the add-on at the result — one button, no
terminal.

Two things make ARDY harder to install than Kimodo, and both are handled here
rather than being pushed onto the user:

* **The native extension.** ARDY's setup.py always builds `motion_correction`
  through CMake, and unlike the Aero-Ex Kimodo fork there is no environment
  variable to skip it. On a machine without CMake and a C++ compiler the whole
  install would fail. Instead we detect the toolchain up front and, when it is
  missing, install ARDY with the extension dropped — everything works except
  foot-skate correction, which the bridge is then told to skip. The panel says
  so plainly instead of failing.

* **The gated text encoder.** ARDY's LLM2Vec encoder descends from
  meta-llama/Meta-Llama-3-8B-Instruct, which needs an accepted licence and a
  HuggingFace token. A refusal is detected and turned into an actionable
  message ("request access here, paste a token there") rather than a raw 401.

Deletion safety, subprocess environment handling, download retries and progress
reporting are all reused from setup_operator so both installers behave the same
and the panel needs only one set of progress UI.
"""

import json
import os
import re
import shutil
import subprocess
import threading
import traceback

import bpy
from bpy.types import Operator
from bpy.props import BoolProperty, StringProperty

from . import setup_operator as so
from .setup_operator import (
    _build_env,
    _download_with_retry,
    _find_system_python,
    _is_protected_path,
    _log,
    _lock,
    _max_gpu_compute_capability,
    _NO_WINDOW,
    _run,
    _state,
    _validate_python,
    _venv_python,
    _wrap_path,
    python_minor_of,
    torch_index_for,
)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

_DEFAULT_VENV_NAME = ".ardy-venv"
# Written last; its absence means the venv is partial and safe to wipe.
_SENTINEL_NAME = ".ardy_install_complete"
# Records what the install actually managed to do (e.g. whether foot-skate
# correction is available) so the bridge is launched accordingly.
_MANIFEST_NAME = "ardy_install.json"
# ARDY's llm2vec wrapper joins TEXT_ENCODERS_DIR with the repo id, so the
# encoders must live at <dir>/<org>/<repo>.
_ENCODERS_NAME = "text-encoders"

ARDY_REPO = ("nv-tlabs", "ardy")

# The encoder ARDY's default preset asks for (ardy/model/load_model.py).
# Both of these hold only a LoRA adapter (adapter_model.safetensors) plus a
# config and tokenizer — no base weights.
ENCODER_REPOS = (
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
)

# ...so the base Llama-3 weights have to come from Meta's gated repo, and they
# belong in the SAME directory as the adapter.  LLM2Vec loads the pair from one
# folder — see "special case where config.json and adapter weights are in the
# same directory" in ardy/model/llm2vec/llm2vec.py — first reading the shards
# with LlamaBiModel.from_pretrained(dir) and then applying the adapter from
# that same dir.  This is the step whose absence made the encoder unloadable.
BASE_MODEL_REPO = "meta-llama/Meta-Llama-3-8B-Instruct"

# Take only the weights.  The adapter repo's own config.json carries
# _name_or_path = meta-llama/Meta-Llama-3-8B-Instruct, which LLM2Vec reads back
# to pick the Llama-3 prompt template (llm2vec.py, prepare_for_tokenization),
# and its tokenizer is the one the encoder expects — neither may be overwritten
# by the base repo's copies.
#
# Named precisely rather than "*.safetensors": the patterns are matched against
# the whole relative path, so a loose glob would also pull anything the repo
# keeps in subfolders (Meta ships a second full copy of the weights under
# original/) and double the download.
BASE_MODEL_PATTERNS = [
    "model-*.safetensors",          # the sharded checkpoint
    "model.safetensors",            # single-file layout, if it ever ships one
    "model.safetensors.index.json",
]

# What a directory needs before it can be loaded as a model: either a sharded
# index, or a single-file checkpoint.  The adapter alone never satisfies this,
# which is exactly the state a pre-fix install left behind.
_WEIGHT_MARKERS = ("model.safetensors.index.json", "model.safetensors",
                   "pytorch_model.bin.index.json", "pytorch_model.bin")

# Only the Core checkpoints suit a character-animation add-on: G1 is a Unitree
# robot skeleton, and no SOMA checkpoint has been released yet.
DEFAULT_MODEL_REPO = "nvidia/ARDY-Core-RP-20FPS-Horizon40"

LLAMA_MODEL_URL = "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"
HF_TOKEN_URL = "https://huggingface.co/settings/tokens"

# Substrings that mean "HuggingFace refused you", not "the network broke".
_GATED_MARKERS = (
    "401 client error", "403 client error", "gated repo", "gatedrepo",
    "awaiting a review", "access to model", "is restricted",
    "you must be authenticated", "not authorized",
)

_VENV_MARKERS = {"pyvenv.cfg", "bin", "Scripts", "lib", "lib64", "include"}
_MANAGED_VENV_BASENAMES = {_DEFAULT_VENV_NAME, "ardy-venv"}


def _default_venv() -> str:
    return os.path.join(os.path.expanduser("~"), _DEFAULT_VENV_NAME)


def managed_venv() -> str:
    """The configured ARDY venv location, or the default.

    Reads bpy.context — main thread only. Background threads use the install
    dir passed into them.
    """
    try:
        prefs = bpy.context.preferences.addons[__package__].preferences
        loc = (prefs.ardy_install_location or "").strip()
        if loc:
            return os.path.abspath(bpy.path.abspath(os.path.expanduser(loc)))
    except Exception:
        pass
    return _default_venv()


def managed_python() -> str:
    return _venv_python(managed_venv())


def venv_root_for(python_exe: str) -> str:
    return so.venv_root_for(python_exe)


def is_ardy_venv(python_exe: str) -> bool:
    """True if python_exe points into a completed ARDY venv."""
    root = venv_root_for(python_exe)
    return (
        bool(root)
        and os.path.isfile(python_exe)
        and os.path.isfile(os.path.join(root, _SENTINEL_NAME))
    )


def venv_exists() -> bool:
    return os.path.isdir(managed_venv())


def is_installed() -> bool:
    return bool(managed_python()) and os.path.isfile(
        os.path.join(managed_venv(), _SENTINEL_NAME)
    )


def manifest_for(python_exe: str) -> dict:
    """Read the install manifest next to a venv ({} when absent)."""
    root = venv_root_for(python_exe)
    if not root:
        return {}
    try:
        with open(os.path.join(root, _MANIFEST_NAME), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def has_postprocess(python_exe: str = "") -> bool:
    """True when foot-skate correction is available in this ARDY install."""
    return bool(manifest_for(python_exe or managed_python()).get("postprocess"))


def bridge_env(python_exe: str, env: dict) -> dict:
    """Environment for the ARDY bridge subprocess.

    ARDY's wrapper reads TEXT_ENCODERS_DIR and HUGGINGFACE_CACHE_DIR straight
    from the environment (ardy/model/llm2vec/llm2vec_wrapper.py), so unlike the
    Kimodo path nothing in site-packages has to be rewritten.
    """
    root = venv_root_for(python_exe)
    if not root:
        return env
    encoders = os.path.join(root, _ENCODERS_NAME)
    if os.path.isdir(encoders):
        env["TEXT_ENCODERS_DIR"] = encoders
    if os.path.isfile(os.path.join(root, _SENTINEL_NAME)):
        # Everything was fetched at install time; never reach for the network
        # again (load_model calls snapshot_download unconditionally).
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
    return env


# ---------------------------------------------------------------------------
# Safe deletion
# ---------------------------------------------------------------------------

def _looks_like_ardy_venv(path: str) -> bool:
    """True only if *path* is unambiguously an ARDY-managed venv directory.

    Same rule as the Kimodo guard: a positive identity signal is required, so a
    clean-retry wipe can never hit a folder that merely happens to be the
    chosen install location.
    """
    if os.path.isfile(os.path.join(path, _SENTINEL_NAME)):
        return True
    if os.path.basename(os.path.normpath(path)) in _MANAGED_VENV_BASENAMES:
        try:
            entries = set(os.listdir(path))
        except OSError:
            return False
        if not entries:
            return True
        if entries & _VENV_MARKERS:
            return True
    return False


def _safe_rmtree(path: str) -> None:
    """shutil.rmtree() that only ever deletes an ARDY-managed venv."""
    if not path or not str(path).strip():
        raise RuntimeError("refusing to delete an empty path")
    real = os.path.realpath(os.path.abspath(os.path.expanduser(str(path))))
    if not os.path.isdir(real):
        raise RuntimeError(f"not a directory: {real}")
    if _is_protected_path(real):
        raise RuntimeError(
            f"refusing to delete protected location '{real}' — the install "
            f"folder must be a dedicated ARDY venv directory, not a home, "
            f"system, or mount-point path"
        )
    if not _looks_like_ardy_venv(real):
        raise RuntimeError(
            f"refusing to delete '{real}': it is not an ARDY-managed venv "
            f"(no '{_SENTINEL_NAME}' marker and not a recognised ardy-venv "
            f"folder). If you are certain, delete it manually."
        )
    shutil.rmtree(real)


# ---------------------------------------------------------------------------
# Toolchain detection
# ---------------------------------------------------------------------------

def _tool_ok(cmd: list) -> bool:
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=10,
                           env=_build_env(), **_NO_WINDOW)
        return r.returncode == 0
    except Exception:
        return False


def _has_build_toolchain() -> bool:
    """True when ARDY's CMake extension can actually be compiled here.

    ARDY's setup.py needs CMake >= 3.15 and a C++17 compiler. On Windows it
    accepts MinGW's g++ or an MSVC generator; elsewhere any g++/c++/clang++.
    """
    if not _tool_ok(["cmake", "--version"]):
        return False
    for cxx in ("g++", "c++", "clang++"):
        if _tool_ok([cxx, "--version"]):
            return True
    if os.name == "nt":
        # MSVC is not on PATH unless a developer prompt is active; cl.exe
        # answering at all is enough to know a build could work.
        return _tool_ok(["cl"])
    return False


def _looks_gated(text: str) -> bool:
    low = (text or "").lower()
    return any(marker in low for marker in _GATED_MARKERS)


def _recent_log_text() -> str:
    """The tail of the install log, for classifying a subprocess failure.

    _run() only raises "step failed (exit N)"; the HTTP status that explains
    why is in the output it streamed to the log.
    """
    with _lock:
        return "\n".join(_state.get("lines", []))


def check_llama_access(token: str) -> "tuple[bool, str]":
    """Verify up front that this token can read the gated Llama repo.

    Without this the user would download PyTorch and wait several minutes only
    to be refused at the text-encoder step. Uses urllib so it needs nothing
    installed. A network failure is not treated as a refusal — the real
    download retries anyway.
    """
    import urllib.error
    import urllib.request

    if not token:
        return False, "No HuggingFace token was provided."

    url = "https://huggingface.co/api/models/meta-llama/Meta-Llama-3-8B-Instruct"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            if resp.status == 200:
                return True, ""
            return False, f"HuggingFace answered HTTP {resp.status}."
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return False, (
                "HuggingFace refused this token for "
                "meta-llama/Meta-Llama-3-8B-Instruct (HTTP "
                f"{exc.code}). Accept the model licence on its page, wait for "
                "approval, and make sure the token has read access."
            )
        return False, f"HuggingFace answered HTTP {exc.code}."
    except Exception as exc:
        # Offline or blocked: let the install proceed and fail later if needed.
        _log(f"Could not pre-check Llama access ({exc}) — continuing anyway.")
        return True, ""


# ---------------------------------------------------------------------------
# Source acquisition
# ---------------------------------------------------------------------------

def _download_encoder(venv_py: str, repo_id: str, local_dir: str,
                      hf_token: str, allow_patterns: "list[str] | None" = None) -> None:
    """Fetch one encoder repo, translating a refusal into an actionable error.

    Everything the encoder needs is gated behind Meta-Llama-3-8B-Instruct, so a
    401/403 here means "your token has no Llama access", not "the network
    broke" — and it deserves to say so rather than surfacing as a stack trace.
    """
    try:
        _download_with_retry(
            venv_py, f"Downloading {repo_id.split('/')[-1]}",
            repo_id=repo_id, local_dir=local_dir, hf_token=hf_token,
            allow_patterns=allow_patterns,
        )
    except RuntimeError as exc:
        if _looks_gated(str(exc)) or _looks_gated(_recent_log_text()):
            with _lock:
                _state["needs_llama_access"] = True
            raise RuntimeError(
                f"HuggingFace refused {repo_id}. The ARDY text encoder is "
                f"gated behind Meta-Llama-3-8B-Instruct: open {LLAMA_MODEL_URL}, "
                "accept the licence, wait for approval, then paste a token with "
                f"read access ({HF_TOKEN_URL}) and retry."
            ) from exc
        raise


def _require_weights(local_dir: str) -> None:
    """Raise unless *local_dir* holds base weights, not just a LoRA adapter.

    adapter_model.safetensors matches no marker on purpose: a directory with
    only the adapter is precisely the broken state this guard exists to catch.
    """
    try:
        names = set(os.listdir(local_dir))
    except OSError as exc:
        raise RuntimeError(f"Text-encoder directory is unreadable: {exc}") from exc

    if any(marker in names for marker in _WEIGHT_MARKERS):
        return
    if any(n.startswith("model-") and n.endswith(".safetensors") for n in names):
        return

    raise RuntimeError(
        "The text encoder is missing its base weights: no model.safetensors "
        f"or shards in {local_dir} (found: {', '.join(sorted(names)) or 'nothing'}). "
        f"The {BASE_MODEL_REPO} download did not complete — check the log above "
        "for a HuggingFace refusal, then retry the install."
    )


def _fetch_source(venv_py: str, work_dir: str) -> str:
    """Put the ARDY source tree on disk and return its path.

    A source checkout (rather than pip installing the URL directly) is what
    lets us drop the native extension when no compiler is present.
    """
    owner, repo = ARDY_REPO
    src = os.path.join(work_dir, repo)

    if so._git_available():
        _log("Cloning ARDY (shallow)…")
        _run(["git", "clone", "--depth", "1",
              f"https://github.com/{owner}/{repo}.git", src],
             "Downloading ARDY source")
        return src

    # No git: fetch the default-branch zip with the venv's own Python so we do
    # not depend on anything else being installed.
    _log("git not found — downloading the ARDY source archive instead…")
    zip_path = os.path.join(work_dir, "ardy.zip")
    _run([venv_py, "-c",
          "import os,urllib.request,zipfile;"
          "u=os.environ['_ARDY_URL'];z=os.environ['_ARDY_ZIP'];"
          "d=os.environ['_ARDY_DIR'];"
          "urllib.request.urlretrieve(u,z);"
          "zipfile.ZipFile(z).extractall(d)"],
         "Downloading ARDY source",
         env={"_ARDY_URL": f"https://github.com/{owner}/{repo}/archive/HEAD.zip",
              "_ARDY_ZIP": zip_path, "_ARDY_DIR": work_dir})
    for entry in sorted(os.listdir(work_dir)):
        full = os.path.join(work_dir, entry)
        if os.path.isdir(full) and entry.lower().startswith(repo):
            return full
    raise RuntimeError("ARDY archive downloaded but no source folder was found inside it.")


def _disable_native_extension(src: str) -> None:
    """Rewrite setup.py so pip installs ARDY without compiling MotionCorrection.

    Only the ext_modules list is emptied; the pure-Python `motion_correction`
    package still installs, and ardy/postprocess.py only imports its compiled
    half lazily — so nothing breaks until post-processing is requested, which
    the bridge is told not to do.
    """
    path = os.path.join(src, "setup.py")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    patched, n = re.subn(r"ext_modules\s*=\s*\[[^\]]*\]", "ext_modules=[]", text)
    if not n:
        raise RuntimeError(
            "Could not disable ARDY's native extension: setup.py no longer has "
            "the expected ext_modules list. Install CMake and a C++ compiler, "
            "then retry."
        )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(patched)
    _log("Native extension disabled for this install.")


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

def _do_install(hf_token: str = "", system_python: str = "",
                install_dir: str = "") -> None:
    work_dir = ""
    try:
        venv = install_dir or _default_venv()
        encoders = os.path.join(venv, _ENCODERS_NAME)
        sentinel = os.path.join(venv, _SENTINEL_NAME)

        # 0 — fail fast on the one thing that cannot be worked around, before
        #     spending twenty minutes downloading PyTorch.
        if not hf_token:
            with _lock:
                _state["needs_llama_access"] = True
            raise RuntimeError(
                "A HuggingFace token is required. ARDY's text encoder is built "
                "on the gated Meta-Llama-3-8B-Instruct model: accept its "
                "licence, create a read token, paste it into the token field "
                "below, then click Retry."
            )
        _log("Checking HuggingFace access to the gated Llama-3 text encoder…")
        ok, why = check_llama_access(hf_token)
        if not ok:
            with _lock:
                _state["needs_llama_access"] = True
            raise RuntimeError(why)
        _log("HuggingFace access confirmed.")

        # 1 — system Python
        sys_py = ""
        if system_python:
            if os.path.isfile(system_python) and _validate_python(system_python):
                _log(f"Using user-specified Python: {system_python}")
                sys_py = system_python
            else:
                _log(f"User-specified Python is not a valid 3.10–3.12 executable: "
                     f"{system_python}")
                _log("Falling back to auto-detection…")
        if not sys_py:
            _log("Searching for system Python 3.10+…")
            sys_py = _find_system_python()
        if not sys_py:
            with _lock:
                _state["needs_python"] = True
            raise RuntimeError(
                "No Python 3.10+ found. Install it from python.org (tick "
                "'Add Python to PATH'), then click Retry Install."
            )
        _log(f"Found: {sys_py}")

        # 2 — venv
        _log(f"Install location: {venv}")
        os.makedirs(os.path.dirname(venv) or ".", exist_ok=True)
        _run([sys_py, "-m", "venv", venv], "Creating venv")

        venv_py = _venv_python(venv)
        if not venv_py:
            raise RuntimeError("Venv was created but Python binary not found.")
        pip = [venv_py, "-m", "pip"]

        # 3 — pip
        _run([*pip, "install", "--upgrade", "pip"], "Upgrading pip")

        # 4 — PyTorch, matched to the GPU exactly as the Kimodo installer does
        gpu_cap = _max_gpu_compute_capability()
        _log(f"Detected GPU compute capability: {gpu_cap[0]}.{gpu_cap[1]}")
        torch_index, cuda_label = torch_index_for(python_minor_of(venv_py), gpu_cap)
        _log(f"Installing PyTorch with CUDA {cuda_label} support — "
             f"this may take several minutes…")
        _run([*pip, "install", "torch", "--index-url", torch_index],
             "Installing PyTorch")

        # 5 — ARDY source
        work_dir = os.path.join(venv, "_src")
        os.makedirs(work_dir, exist_ok=True)
        src = _fetch_source(venv_py, work_dir)

        # 6 — decide whether the native extension can be built
        postprocess = _has_build_toolchain()
        if postprocess:
            _log("CMake and a C++ compiler found — building motion correction "
                 "(enables foot-skate cleanup).")
        else:
            _log("No CMake / C++ compiler found. Installing ARDY without the "
                 "motion-correction extension: everything works except "
                 "foot-skate cleanup. Install CMake + a C++ compiler and "
                 "reinstall to enable it.")
            _disable_native_extension(src)

        # 7 — ARDY itself (core inference only; the viser demo and TensorRT
        #     extras are for the standalone app, not for the bridge)
        try:
            _run([*pip, "install", src], "Installing ARDY")
        except RuntimeError:
            if not postprocess:
                raise
            # The build was the most likely thing to fail — retry without it
            # rather than losing the whole install over a toolchain problem.
            _log("Build failed — retrying without the motion-correction "
                 "extension (foot-skate cleanup will be unavailable).")
            _disable_native_extension(src)
            postprocess = False
            _run([*pip, "install", src], "Installing ARDY (no native extension)")

        # 8 — text encoder, into <venv>/text-encoders/<org>/<repo> so ARDY's
        #     TEXT_ENCODERS_DIR lookup finds it without patching any source
        os.makedirs(encoders, exist_ok=True)
        for repo_id in ENCODER_REPOS:
            local_dir = os.path.join(encoders, *repo_id.split("/"))
            os.makedirs(local_dir, exist_ok=True)
            _log(f"Downloading text-encoder adapter {repo_id}…")
            _download_encoder(venv_py, repo_id, local_dir, hf_token)

        # 8b — the base weights the adapters are trained against.  They land in
        #      the adapter's own directory so LLM2Vec finds both together, and
        #      this is the download that actually needs Llama-3 access.
        base_dir = os.path.join(encoders, *ENCODER_REPOS[0].split("/"))
        _log(f"Downloading base weights {BASE_MODEL_REPO} (~16 GB)…")
        _download_encoder(venv_py, BASE_MODEL_REPO, base_dir, hf_token,
                          allow_patterns=BASE_MODEL_PATTERNS)

        # Never write the sentinel over an encoder that cannot load: without
        # this the failure surfaces much later, as an OSError from deep inside
        # transformers when the bridge starts.
        _require_weights(base_dir)

        # 9 — checkpoint into the HF cache (load_model resolves it from there)
        _log(f"Downloading {DEFAULT_MODEL_REPO} weights…")
        _download_with_retry(
            venv_py, "Downloading ARDY Core weights",
            repo_id=DEFAULT_MODEL_REPO, hf_token=hf_token,
        )

        # 10 — record what this install can do, then mark it complete
        with open(os.path.join(venv, _MANIFEST_NAME), "w", encoding="utf-8") as fh:
            json.dump({
                "postprocess": postprocess,
                "model": DEFAULT_MODEL_REPO,
                "encoders_dir": encoders,
            }, fh, indent=2)

        # The source tree is only needed during the build.
        shutil.rmtree(work_dir, ignore_errors=True)
        work_dir = ""

        def _set_path():
            try:
                for scene in bpy.data.scenes:
                    if not scene.kimodo.ardy_python_executable:
                        scene.kimodo.ardy_python_executable = venv_py
                prefs = bpy.context.preferences.addons[__package__].preferences
                prefs.ardy_install_location = venv
                bpy.ops.wm.save_userpref()
            except Exception:
                pass
        bpy.app.timers.register(_set_path, first_interval=0.1)

        open(sentinel, "w").close()

        with _lock:
            _state["done"] = True
        _log("Installation complete! Switch the backend to ARDY and click Start ARDY.")

    except Exception as exc:
        print(f"[ARDY Install] FAILED:\n{traceback.format_exc()}", flush=True)
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
        with _lock:
            _state["error"] = str(exc)
    finally:
        with _lock:
            _state["running"] = False


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class ARDY_OT_ConfirmDeleteDir(Operator):
    """Confirmation dialog shown before any ARDY venv directory is deleted"""
    bl_idname = "kimodo.ardy_confirm_delete_dir"
    bl_label = "Delete ARDY Virtual Environment?"
    bl_options = {'INTERNAL'}

    directory: StringProperty(default="", options={'SKIP_SAVE'})
    action: StringProperty(default="RESET", options={'SKIP_SAVE'})

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=460)

    def draw(self, context):
        layout = self.layout
        layout.label(text="This will permanently delete:", icon='ERROR')
        box = layout.box()
        for line in _wrap_path(self.directory):
            box.label(text=line)
        layout.label(text="Everything inside this folder goes with it.")
        layout.label(text="This cannot be undone.")

    def execute(self, context):
        if self.action == 'RETRY_INSTALL':
            return bpy.ops.kimodo.install_ardy(
                'EXEC_DEFAULT', prompt_location=False, confirmed=True)
        return bpy.ops.kimodo.reset_ardy_venv('EXEC_DEFAULT', confirmed=True)


def _request_delete_confirmation(directory: str, action: str) -> bool:
    """Open the confirmation dialog. False when no UI can show it."""
    try:
        bpy.ops.kimodo.ardy_confirm_delete_dir(
            'INVOKE_DEFAULT', directory=directory, action=action)
        return True
    except Exception:
        return False


class KIMODO_OT_InstallArdy(Operator):
    bl_idname = "kimodo.install_ardy"
    bl_label = "Install ARDY (Auto)"
    bl_description = (
        "Choose a folder, then create an ARDY virtual environment there, "
        "install ARDY and a matching PyTorch build, download the text encoder "
        "and the Core checkpoint, and configure the add-on automatically. "
        "Requires a HuggingFace token with Llama-3 access, internet, and "
        "~25 GB of disk space"
    )

    directory: StringProperty(subtype='DIR_PATH', options={'SKIP_SAVE'})
    prompt_location: BoolProperty(default=True, options={'SKIP_SAVE'})
    confirmed: BoolProperty(default=False, options={'SKIP_SAVE'})

    def invoke(self, context, event):
        if so.is_installing() or not self.prompt_location:
            return self.execute(context)
        parent = os.path.dirname(managed_venv()) or os.path.expanduser("~")
        self.directory = parent + os.sep
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        if so.is_installing():
            self.report({"WARNING"}, "An installation is already in progress.")
            return {"CANCELLED"}

        chosen = (self.directory or "").strip()
        if self.prompt_location and chosen:
            install_dir = os.path.join(
                os.path.abspath(bpy.path.abspath(chosen)), "ardy-venv")
            try:
                prefs = context.preferences.addons[__package__].preferences
                prefs.ardy_install_location = install_dir
            except Exception:
                pass

        install_dir = managed_venv()
        real_dir = os.path.realpath(
            os.path.abspath(os.path.expanduser(install_dir)))
        if _is_protected_path(real_dir):
            self.report(
                {"ERROR"},
                f"Refusing to use '{real_dir}' as the install location. Pick a "
                f"dedicated folder (an 'ardy-venv' subfolder is created for "
                f"you) — not your home, a system, or a mount-point directory.",
            )
            return {"CANCELLED"}

        if venv_exists() and not is_installed():
            if not self.confirmed:
                if not _request_delete_confirmation(install_dir, 'RETRY_INSTALL'):
                    self.report(
                        {"ERROR"},
                        f"Refusing to delete '{install_dir}' without "
                        f"confirmation. Remove it manually to retry.",
                    )
                return {"CANCELLED"}
            _log(f"Removing partial venv for clean retry: {install_dir}")
            try:
                _safe_rmtree(install_dir)
            except Exception as exc:
                self.report({"ERROR"}, f"Could not remove partial venv: {exc}")
                return {"CANCELLED"}

        if is_installed():
            self.report({"INFO"}, "Managed ARDY venv already exists.")
            return {"CANCELLED"}

        hf_token = ""
        system_python = ""
        try:
            prefs = context.preferences.addons[__package__].preferences
            hf_token = (prefs.hf_token or "").strip()
            system_python = (prefs.system_python_override or "").strip()
        except Exception:
            pass

        with _lock:
            _state.update(running=True, lines=[], error="", done=False,
                          needs_python=False, dl_progress=0.0, dl_label="",
                          target="ardy", needs_llama_access=False)

        threading.Thread(
            target=_do_install,
            args=(hf_token, system_python, install_dir),
            daemon=True,
        ).start()

        def _redraw():
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == "VIEW_3D":
                        area.tag_redraw()
            return 0.5 if so.is_installing() else None

        bpy.app.timers.register(_redraw, first_interval=0.5)
        self.report({"INFO"}, "ARDY installation started — watch the Connection panel.")
        return {"FINISHED"}


class KIMODO_OT_UseInstalledArdy(Operator):
    bl_idname = "kimodo.use_installed_ardy"
    bl_label = "Use Installed ARDY"
    bl_description = "Point the add-on at the managed ARDY venv Python"

    def execute(self, context):
        py = managed_python()
        if not py:
            self.report({"ERROR"}, f"Managed ARDY venv not found at {managed_venv()}")
            return {"CANCELLED"}
        context.scene.kimodo.ardy_python_executable = py
        self.report({"INFO"}, f"ARDY Python path set to: {py}")
        return {"FINISHED"}


class KIMODO_OT_ResetArdyVenv(Operator):
    bl_idname = "kimodo.reset_ardy_venv"
    bl_label = "Delete ARDY Virtual Environment"
    bl_description = (
        "Delete the ARDY venv and allow a fresh install. Use this when a "
        "previous install failed, is stuck, or you need to reinstall for a "
        "different GPU or Python version"
    )

    confirmed: BoolProperty(default=False, options={'SKIP_SAVE'})

    def execute(self, context):
        if so.is_installing():
            self.report({"WARNING"}, "Cannot reset while installation is in progress.")
            return {"CANCELLED"}
        if not venv_exists():
            self.report({"INFO"}, "No ARDY venv found — nothing to reset.")
            return {"CANCELLED"}
        install_dir = managed_venv()
        if not self.confirmed:
            if not _request_delete_confirmation(install_dir, 'RESET'):
                self.report(
                    {"ERROR"},
                    f"Refusing to delete '{install_dir}' without confirmation.",
                )
            return {"CANCELLED"}
        try:
            _safe_rmtree(install_dir)
        except Exception as exc:
            self.report({"ERROR"}, f"Could not remove venv: {exc}")
            return {"CANCELLED"}
        with _lock:
            _state.update(running=False, lines=[], error="", done=False,
                          needs_llama_access=False)
        self.report({"INFO"}, f"Removed {install_dir} — ready for a fresh install.")
        return {"FINISHED"}


class KIMODO_OT_OpenURL(Operator):
    bl_idname = "kimodo.open_url"
    bl_label = "Open Link"
    bl_description = "Open this page in your browser"

    url: StringProperty(default="", options={'SKIP_SAVE'})

    def execute(self, context):
        import webbrowser
        if not self.url:
            return {"CANCELLED"}
        webbrowser.open(self.url)
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_classes = [
    ARDY_OT_ConfirmDeleteDir,
    KIMODO_OT_InstallArdy,
    KIMODO_OT_UseInstalledArdy,
    KIMODO_OT_ResetArdyVenv,
    KIMODO_OT_OpenURL,
]


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
