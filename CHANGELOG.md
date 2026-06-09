# Changelog

## [1.5.2] — 2026-06-09

### Fixed

- **Cancel now actually cancels**: Clicking *Cancel* previously only updated the UI — the in-flight result was still imported when it arrived, and a second generation could be started on the same pipe, causing the new request to consume the old request's response (wrong file imported). Cancel now discards the abandoned result (including its temp file), the modal operators report "Cancelled" instead of importing, and the bridge pipe refuses new requests until the abandoned job has drained.
- **Startup/generation timeouts could never fire**: Waiting for bridge messages used a blocking `readline()`, so a bridge process that hung without printing (e.g. stuck CUDA init) froze the start thread forever and the UI stayed on "Starting…" past the 7-minute ceiling. Bridge stdout is now drained by a dedicated reader thread into a queue that is polled with real timeouts.
- **`nvidia-smi` ran on every viewport redraw**: The Connection panel called the GPU check from `draw()`, spawning an `nvidia-smi` subprocess (5 s timeout) dozens of times per second while Kimodo was not installed — UI stutter on Linux and a console-window flash per redraw on Windows. The result is now detected once and cached for the session.
- **Console windows flashing on Windows**: All subprocesses (bridge server, pip, venv, git, nvidia-smi, Python probes) are now launched with `CREATE_NO_WINDOW` so no console windows pop up over Blender.
- **Bridge still hit the network on every start**: `HF_HUB_OFFLINE=1` is now set for the bridge subprocess when the managed venv is used (alongside the existing `TRANSFORMERS_OFFLINE`/`HF_DATASETS_OFFLINE`), so `load_model`'s unconditional `snapshot_download` uses the pre-downloaded cache instead of contacting HuggingFace — faster starts, no rate-limiting, works offline.
- **Random seed overflow**: `random.randint(0, 2**31)` could (rarely) produce `2**31`, which overflows Blender's 32-bit `IntProperty` when written to history/segment seeds. Upper bound corrected to `2**31 - 1`.
- **Silent constraint drops**: Segment and multi-prompt generation swallowed constraint-build errors and silently generated *unconstrained* motion. A warning is now reported when constraints fail to build, matching the single-generate path.
- **Conda env roots on Windows**: Pointing the *Kimodo Python* field at a conda env root now works — `python.exe` at the env root is probed in addition to `Scripts/python.exe`.
- **Stop/receive race**: Stopping the bridge while a generation was waiting for a response could raise `AttributeError` instead of reporting "process died" (fixed as part of the queue-based reader).

### Changed

- **Python download button is OS-aware**: On Linux/macOS it now opens the python.org downloads page instead of a Windows `.exe` installer.
- **Help panel** quick-start updated to reference the auto-installer instead of the pre-1.2.0 manual venv setup.

### Removed

- **`gradio_client.py`** (dead code): leftover from the pre-subprocess Gradio REST architecture; nothing imported it. Module docstrings updated to describe the bridge architecture.
- **Committed `__pycache__/*.pyc` files** removed from version control; `.gitignore` now ignores `__pycache__/` as a whole instead of individual files.

## [1.5.1] — 2026-06-09

### Fixed

- **UnicodeDecodeError during install**: Subprocesses launched with `text=True` decoded their output strictly, so a single non-UTF-8 byte from a child process (e.g. a `huggingface_hub`/`tqdm` progress bar written in a non-UTF-8 Windows console code page) aborted the whole install with `'utf-8' codec can't decode byte 0xa2`. All subprocess and text-file reads now pin `encoding="utf-8"` with `errors="replace"`.

### Added

- **HuggingFace token field**: An optional masked "HF Token" input now appears in the Connection panel before install. Entering a read token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) removes anonymous rate-limits that could cause model downloads to stall at 0%.
- **Download progress bar**: A live progress bar is shown in the installing panel during both HuggingFace model downloads, updated from tqdm output in real time.
- **Download retry with backoff**: Both `snapshot_download` calls (LLM2Vec encoder and SOMA weights) now retry up to 3 times with 15 s / 30 s backoff on failure, with retry messages visible in the install status UI.
- **Per-request download timeout**: `HF_HUB_DOWNLOAD_TIMEOUT=120` is set for download subprocesses so stalled HTTP connections are killed after 2 minutes instead of hanging forever.

### Changed

- **Computer restart reminder**: The Python install prompt now tells Windows users to restart their computer (not just Blender) after installing Python 3.12, so the updated PATH takes effect.

## [1.5.0] — 2026-05-26

### Added

- **Curve-path following**: Draw any Bezier or NURBS curve in the viewport, pick it in the Motion Constraints panel, set a waypoint count (2–30) and a frame range, then click *Sample N Waypoints*. The curve is evaluated via the depsgraph (modifiers supported) and walked to place evenly arc-length-spaced Empties (`Kimodo_Path_01…N`), each registered as a `root2d` constraint so it flows straight through to Kimodo.
- **Generate N Variations**: Run the current prompt 2–5 times with random seeds via a sequential modal operator; each result is imported as `Kimodo_Var_N` and recorded in history.
- **Generation history**: A rolling log (max 20, newest-first) of every generated motion — prompt, seed, duration, BVH path, and timestamp — shown as a collapsible list in the Quick Generate panel with detail view, Re-import BVH, and Clear History actions.
- **Transition frames control**: `num_transition_frames` (1–30, default 5) is now exposed in the Motion Segments panel to control crossfade sharpness between segments, instead of being hardcoded.

## [1.4.0] — 2026-05-19

### Fixed

- **PyTorch install on Python 3.13**: The cu121 index has no Python 3.13 wheels; the installer now detects the Python version and uses cu124 (PyTorch 2.6+) for Python 3.13.
- **PyTorch install on Blackwell GPUs (RTX 50xx)**: cu121/cu124 wheels only compile kernels for up to sm_90. RTX 50xx cards (sm_120+) caused a `no kernel image is available` CUDA error at runtime. The installer now detects GPU compute capability via `nvidia-smi` and selects cu128 (PyTorch 2.7+) for Blackwell.

### Added

- **Delete Venv button**: A trash icon button now appears at the bottom of the Connection panel after a successful install, so the venv can be wiped and reinstalled from the UI without going to the terminal. A confirmation popup prevents accidental deletion.

### Removed

- **`kimodo_textencoder --device cpu` tip**: The command did not work reliably and Kimodo runs fine on lower-VRAM cards without it.

---

## [1.3.3] — 2026-05-15

### Fixed

- **Rest-orientation-invariant joint rotations**: Joint rotation extraction now accounts for the rest orientation of each bone, so retargeted motion is correct even when the source armature is not in a canonical rest pose (previously rotations could be offset by the bone's rest transform).
- **Hand / foot effector constraints targeting hips**: Hand and foot spatial constraints were incorrectly sending the effector target to the hips bone instead of the wrist/ankle end-effector. Fixed so each constraint type maps to the correct bone.
- **Full-Body constraint re-enabled**: The Full-Body pose constraint (pose a reference armature to set a full joint-pose keyframe) was inadvertently disabled; it is now re-enabled. Duplicate pose markers are also frozen to prevent accidental edits.

### Changed

- **Default retarget mode is now "Child Of"**: New bone-mapping entries default to the *Child Of* constraint instead of *Copy Rotation*, which produces better full-body results out of the box for most rigs.

### Added

- **`blender_manifest.toml`**: Added the Blender 4.2+ extension-system manifest so the addon can be installed via the new Extensions platform (`.zip` install and legacy Add-ons path both still work).

---

## [1.3.1] — 2026-05-12

### Fixed

- **Desktop-launch compatibility**: Blender launched from a desktop environment (via `.desktop` file, app menu, or file manager) inherits a stripped PATH from the display manager — standard tools like `pip`, `venv`, `git`, and `nvidia-smi` were not found, breaking the installer. A `_build_env()` helper now ensures every subprocess spawned by the plugin receives a complete PATH (including `/usr/local/bin`, `/usr/bin`, `/bin`, etc.) and a valid `HOME`, regardless of how Blender was started.
- **Python detection in desktop sessions**: `_find_system_python()` now probes known install directories (`/usr/bin`, `/usr/local/bin`, `~/.local/bin`, `~/.pyenv/shims`) in addition to `shutil.which`, so Python interpreters installed outside the display manager's minimal PATH are still discovered.

---

## [1.3.0] — 2026-05-12

### Windows improvements

- **Python detection**: Skip Windows App Execution Alias stubs (`%LOCALAPPDATA%\Microsoft\WindowsApps`) that point to the Store instead of a real interpreter, preventing broken venv creation.
- **Python install prompt**: When no Python 3.10+ is found, the Connection panel shows a prominent download button linking directly to the Python 3.12 Windows installer, with instructions to tick "Add Python to PATH" and run as Administrator.
- **Python 3.13 support**: Python 3.13 is now accepted alongside 3.10–3.12.
- **Architecture-aware installer link**: The download button serves the `arm64` installer on ARM machines and `amd64` everywhere else.
- **Git-less install fallback**: When `git` is not on `PATH`, Kimodo and kimodo-viser are installed via GitHub's zip-archive endpoint instead of `git+https://`, so the installer works on stock Windows without Git.
- **Partial/broken venv recovery**: A sentinel file (`.kimodo_install_complete`) is written only on successful install. On the next session, any venv missing the sentinel is treated as broken and wiped on retry — no more silent failures after a mid-install crash.
- **NVIDIA GPU gate**: The install button is disabled and a clear error is shown when no NVIDIA GPU is detected (`nvidia-smi` not found or returns no GPUs). AMD/Intel users see an explicit "not supported" message instead of a cryptic CUDA error later.
- **GPU check on Start**: Clicking *Start Kimodo* also checks for a CUDA-capable GPU and fails fast with a readable error if none is found.
- **Blender restart reminder**: After the Python installer download prompt, the panel reminds users to restart Blender before clicking *Retry Install*.

---

## [1.2.0] — 2026-05-11

### Added

- **One-click auto-installer** (`setup_operator.py`): A new *Install Kimodo (Auto)* button in the Connection panel handles the full setup without any terminal work:
  - Creates a managed Python venv at `~/.kimodo-venv/`
  - Installs PyTorch (CUDA 12.1) and the [Aero-Ex offline fork](https://github.com/Aero-Ex/kimodo) of Kimodo, including a pre-built `motion_correction` wheel (no MSVC / CMake required on Windows)
  - Installs the [NVIDIA kimodo-viser fork](https://github.com/nv-tlabs/kimodo-viser) which provides `viser._timeline_api` (not available in PyPI viser)
  - Installs all undeclared Kimodo dependencies discovered by source audit: `bitsandbytes`, `safetensors`, `psutil`
  - Downloads the `Aero-Ex/KIMODO-Meta3_llm2vec_NF4` LLM2Vec text-encoder model locally and patches `llm2vec_wrapper.py` so it loads from disk
  - Downloads `nvidia/Kimodo-SOMA-RP-v1` model weights into the HF cache
  - Auto-fills the Python path field on completion
  - Shows live progress in the Connection panel; full log printed to the system console with `[Kimodo Install]` prefix
  - Failed installs show a *Retry Install* button that wipes the partial venv and starts clean

- **Offline operation**: After the initial install, Kimodo runs with no internet access. The bridge subprocess is launched with `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` when the managed venv is detected.

- **Bridge console logging**: All output from `bridge_server.py` (PyTorch errors, loading progress, model ready) is now streamed to the system console with a `[Kimodo Bridge]` prefix, making startup failures easy to diagnose.

- **Use Installed Kimodo** button: If the managed venv exists but the Python path is not set, a one-click button sets it automatically.

### Changed

- Connection panel now shows a contextual install section at the top: install prompt → live progress → completion/error state, depending on installer state.
- Failed bridge startup now reports the process exit code and directs the user to the console instead of showing a truncated stderr snippet.

---

## [1.1.0]

- Multi-segment generation: **Generate All** sends all enabled segments in a single model call with smooth transitions.
- Segment frame ranges auto-link (end of segment N locks to start of segment N+1).
- Duplicate / reorder segment operators.
- Seed control per segment.

## [1.0.0]

- Initial release.
- Subprocess bridge architecture (Blender ↔ bridge_server.py over stdin/stdout JSON).
- BVH import into `Kimodo_Source` armature.
- Constraint-based retargeting with bake.
- Bone mapping presets (save/load).
- Motion constraints: Root XZ, Hand, Foot waypoints.
