# Changelog

## [1.7.3] — 2026-08-21

### Fixed

- **Any ARDY generation using a Root XZ constraint with a heading crashed**: `shape mismatch: value tensor of shape [N, 2, 2] cannot be broadcast to indexing result of shape [N, 2]`. Headings are one of the few places the two models genuinely disagree: Kimodo takes them as `[cos θ, sin θ]` pairs, which is what the add-on emits, while ARDY's `Root2DConstraintSet` takes **radians** and does the cos/sin conversion itself. Handing it pairs made it stack cos/sin over an already-2D tensor. The ARDY bridge now converts on the way in, so the add-on keeps emitting one format rather than branching on the backend. This hit Live Stream on every step — steering always sets a heading — and equally hit a plain *Generate* whenever a Root XZ waypoint had its heading enabled, including waypoints produced by *Sample Curve as Waypoints*.

## [1.7.2] — 2026-08-21

### Fixed

- **Live Stream would not start on Blender 5.x**: `Could not build the stream armature: 'Action' object has no attribute 'fcurves'`. Blender 4.4 introduced slotted Actions and kept `Action.fcurves` working as a compatibility shim; 5.0 removed it, and the curves now live one level down and per slot — `action.layers[] → .strips[] → .channelbag(slot) → .fcurves`. The stream writes keyframes through F-curves directly (per-key `keyframe_insert` cannot keep up with playback), so it failed before a single frame was generated. The lookup now handles all three layouts, and the action's slot is bound *before* the curves are collected, so on 5.x the right channelbag is indexed rather than whichever one happened to be first.
- **Scaffolding keyframes no longer empty every curve**: the F-curves are created by inserting one throwaway key per channel, which used to be deleted immediately — leaving every curve keyless, which Blender may garbage-collect, invalidating the references the stream then writes through. The throwaway keys are now placed on the stream's start frame instead, where the first generated window overwrites them as part of its normal truncation.

## [1.7.1] — 2026-08-21

### Fixed

- **Generating on one backend, switching, and generating again produced a broken armature**: the new motion landed on the previous backend's rig with mangled rotations and the character sunk into the floor. *Reuse Armature* keeps pointing at the last generated source armature across a backend switch, and the import path transferred the freshly generated Action onto it without checking the two were the same skeleton. They never are: Kimodo emits somaskel77 (77 bones) and ARDY emits cskel27 (27). Worse, the names that overlap mean different joints — SOMA's `LeftLeg` is the hip, Core27's is the knee — so a partial, wrong-looking pose was applied rather than nothing at all, and because the root location channel is relative to whichever rest pose the BVH import built, the character was also offset by the difference in rest hip height. Reuse now happens only between armatures with identical bone sets; a mismatch keeps the newly imported armature instead and says so in the system console. Switching backend also clears the *Reuse Armature* field, so it never shows a rig that is silently never going to be reused. Deleting the old armature by hand between backends was the workaround; it is no longer needed.

## [1.7.0] — 2026-08-21

### Added

- **Live Stream: ARDY generates while the timeline plays** (#52): a new **Live Stream** panel (ARDY only) starts an open-ended generation instead of a fixed-length clip. Pick an object to follow, press *Start Live Stream*, and the character walks toward wherever that object currently is — move it during playback and the motion re-plans to follow. This is not a bake: ARDY is autoregressive, so the bridge holds the motion state and each step regenerates the near future against the target's live position. Frames land as ordinary keyframes on an ordinary armature, so when you stop the stream the result is a normal Action and the **Retarget** panel works on it unchanged. Kimodo has no autoregressive entry point, so the panel is hidden on that backend.
- **Streaming controls**: *Steps* is the latency dial — a live window has to finish generating faster than it plays, so it defaults far lower (8) than one-shot generation; raise it if motion looks noisy and the GPU has headroom. *Walk Speed* sets how fast the waypoint path chases the target (the prompt still decides the gait). *Commit Frames* is how far ahead of the playhead is locked in: lower reacts sooner, too low and motion about to play gets rewritten under you. *Buffer Frames* is how far ahead to keep generating. The panel reports the buffer depth and measured seconds-per-step, and flags when the GPU is the limit.
- **Graceful backpressure**: when generation falls behind the playhead, playback pauses with "Waiting for ARDY…" and resumes once the buffer refills, rather than playing into frames that have no keyframes yet and freezing the character mid-stride.
- **Streaming bridge protocol**: `stream_begin` / `stream_step` / `stream_end` alongside the existing commands. Each step is still one request and one reply — no server push — so the client's threading, cancellation and single-in-flight rules are unchanged, and the Kimodo path is untouched. The prompt encoding is cached between steps (it is an 8B model), so editing the prompt mid-stream re-encodes and leaving it alone does not.
- **Tests for all of it**: the steering geometry (`tests/test_ardy_steer.py`) and the streaming protocol driven over the real JSON pipe against a mock model, covering the replan arithmetic, history alignment, prompt-cache behaviour and that state is really dropped on `stream_end`. The suite now also pins the transient-state reset, so a saved `.blend` can no longer come back stuck mid-stream — the same trap that produced #43.

### Fixed

- **`cfg_text_weight: null` could crash a request**: the client always sends the ARDY guidance keys so the Kimodo bridge can ignore them, but the ARDY bridge read them with `float(req.get(..., 2.0))`, which raises on an explicit `null` rather than falling back to the default. Only reachable from the new streaming path today, but it was wrong for `generate` too.

## [1.6.1] — 2026-08-20

### Fixed

- **ARDY could not start: the text encoder was installed without its base weights**: the installer downloaded the two `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp*` repos and stopped there, but those repos hold only a LoRA adapter (`adapter_model.safetensors`, ~168 MB) plus a config and tokenizer — the ~16 GB of Llama-3 base weights they adapt live in the separate, gated `meta-llama/Meta-Llama-3-8B-Instruct`. The encoder directory therefore looked complete, the install reported success and wrote its sentinel, and *Start ARDY* then died with `OSError: Error no file named model.safetensors, or pytorch_model.bin` from inside transformers — pointing at a folder that plainly did contain a `.safetensors` file, which made the real cause hard to see. The base weights are now downloaded into the adapter's own directory, which is the layout LLM2Vec loads from (it reads the checkpoint from that folder, then applies the adapter from the same folder). Only the shards and their index are taken, so the adapter repo's `config.json` — which carries the `_name_or_path` LLM2Vec reads back to select the Llama-3 prompt template — and its tokenizer are left intact, and the second full copy of the weights that Meta ships under `original/` is not pulled.
- **A failed encoder download no longer passes as a finished install**: the install now verifies that real base weights are present before writing the completion sentinel, and refuses to finish if all it finds is the adapter. The error names the directory, lists what was actually in it, and names the repo whose download is missing, instead of surfacing twenty minutes later as a stack trace at startup. Because the encoder is gated, a HuggingFace refusal on *any* of the three repos now also reports which repo was refused, alongside the licence and token links.

## [1.6.0] — 2026-08-20

### Added

- **NVIDIA ARDY as a second backend** (#52): the Connection panel now starts with a Kimodo / ARDY switch, and everything downstream — generation, motion constraints, the timeline, retargeting, the frame-rate warning — follows whichever is selected. Kimodo remains the default and is unchanged; nothing about an existing scene behaves differently until the switch is moved. ARDY is autoregressive rather than one-shot, so it responds faster and handles long sequences better, and its Core skeleton uses Mixamo-style bone names (with toes) that auto-map onto typical rigs more cleanly than SOMA's. What it costs today: 20 FPS instead of 30, no standard-T-pose export (the Core skeleton ships no T-pose reference), and a gated text encoder. NVIDIA has announced an ARDY SOMA checkpoint but not released it — only Core and G1 exist, so Core is what the addon offers. The full analysis, including what changes when SOMA lands, is in `docs/ardy-feasibility.md`.
- **One-click ARDY installer**: *Install ARDY (Auto)* creates its own venv (default `~/.ardy-venv`, never shared with Kimodo — ARDY pins `transformers` exactly), installs PyTorch matched to your GPU using the same detection as the Kimodo installer, installs ARDY, downloads the text encoder and the Core checkpoint, and sets the Python path. Two things that would otherwise be dead ends are handled directly: ARDY's text encoder is built on the gated Meta-Llama-3-8B-Instruct, so the panel links the licence page and the token page, and access is verified *before* anything is downloaded — a missing licence fails in seconds instead of after twenty minutes. And ARDY's `setup.py` always compiles a CMake extension with no way to skip it, which would fail outright on a machine with no compiler; the installer detects the toolchain and, when it is absent, installs without the extension and says so in the panel. Everything works in that state except foot-skate cleanup. All deletion, subprocess-environment, download-retry and progress machinery is shared with the Kimodo installer, including the 1.5.7 `_safe_rmtree` guards.
- **BVH writer for ARDY** (`ardy_bvh.py`): ARDY has no BVH exporter — it writes `.npz`, and `ardy/skeleton/bvh.py` only *reads* BVH — so the bridge writes its own from the skeleton's rest joints, hierarchy and per-frame local rotations. Output goes through the addon's existing BVH import, reuse-armature, history and retarget paths unchanged.
- **Blender-free test suite** (`tests/`): `python3 tests/run_tests.py`. Covers the BVH writer against forward kinematics, the ARDY bridge driven over its real JSON protocol as a subprocess, add-on registration and property wiring, and every panel drawn in both backends. `bpy` and the ARDY model are mocked, so no Blender and no GPU are needed; modules whose optional dependency is missing are skipped rather than failed.

### Changed

- **The bridge protocol is now backend-neutral**: `subprocess_client.py` launches `bridge_server.py` or `ardy_bridge.py` depending on the backend and reads the model's real frame rate and skeleton out of its `ready` message instead of assuming Kimodo's 30 FPS and SOMA. The wire format, every status value, the threading model and cancellation are untouched, so the Kimodo path is byte-for-byte what it was.
- **The frame-rate warning follows the model**: it read "Kimodo needs 30 FPS" and set 30 unconditionally; it now names the active model's own rate (Kimodo 30, ARDY Core 20) and sets that.
- **Constraint JSON is no longer SOMA-only**: `constraints.py` gained a skeleton registry, and `build_constraints_json` takes the joint order, parents and end-effector indices from whichever model is loaded. The JSON layout itself did not change — ARDY registers the same constraint types and accepts the same keys — so the Motion Constraints panel, curve waypoints and full-body pose authoring all work against either backend with no user-visible difference.
- **Bone auto-matching knows the Core skeleton**: the SOMA hint table already covered most of Core27, so only the joints SOMA lacks (Spine3, hand ends, thumbs) were added.

## [1.5.8] — 2026-08-20

### Changed

- **Quick Generate and Motion Segments are now one Generate panel with a Single Clip / Timeline switch**: The two panels drew the same model picker, reuse-armature row, T-pose option, FPS warning, Generate button and history list independently — identically, in two separate collapsed panels, with nothing indicating they were related. A new user had no way to tell which one to open first. Single-prompt and multi-segment generation are now two modes of one **Generate** panel: everything that was duplicated between them is drawn once, and only the parts that actually differ — the prompt field vs. the segment list — switch with the mode. The panel also opens expanded by default, since generating motion is the point of the addon rather than something to discover behind a collapsed header. Nothing about generation itself changed — every operator and property this touches is unchanged, only how it's drawn.
- **Motion Constraints rows no longer clip their type label**: each row packed a checkbox, a type icon, the type dropdown, an `F:` label, the frame number, a goto-frame button, a select-object button and a remove button into a single row, which left no room for a label like "Full-Body Pose" alongside the rest — Blender clipped it to `Full-Bo…` and abbreviated the frame field down to a bare `F:`. Each row is now two: what the constraint is (enabled, type icon, type dropdown, remove) on the first line, when it fires (frame, jump-to-frame, select object) on the second.

## [1.5.7] — 2026-07-24

### Changed

- **Deleting a venv now always asks first, and names the directory**: *Reset Venv* / *Delete Venv* previously showed a generic "OK?" confirmation that never said what would be removed, and the *Retry Install* clean-up deleted the partial venv with no prompt at all. Both paths now open the same dialog, which spells out the full path of the directory about to be deleted (wrapped across lines so a long path is never cut off) and warns that its entire contents go with it. Nothing is removed unless that dialog is confirmed — in headless Blender, where no dialog can be shown, the deletion is refused instead of proceeding unconfirmed.

### Fixed

- **Critical: a failed install could delete the chosen folder's entire contents**: The *Retry Install* clean-up and the *Delete Virtual Environment* button called `shutil.rmtree()` directly on the configured install location. Because that location comes from a free-text *Install Location* preference field (and folder browser), pointing it at `$HOME`, `/`, `/home`, a mount point, or any populated directory and then retrying would recursively delete everything under it — there was no guard against non-venv paths, no ownership/marker requirement, and no protection for system or home directories. All deletions now go through a `_safe_rmtree()` guard that refuses to remove anything unless it resolves to a dedicated Kimodo-managed venv (identified by the install-complete sentinel, or a `kimodo-venv`/`.kimodo-venv` folder that is empty or contains real venv artefacts) *and* is not a protected location. Protected covers every platform: POSIX roots (`/`, `/home`, `$HOME` and its parent, `/usr`, …), Windows drive roots (`C:\`), the Windows user profile (`C:\Users\…`) and system folders (`C:\Windows`, `Program Files`, `ProgramData`, resolved from the environment), and any mount point. The installer additionally refuses up front to use a protected path as the install location, so a mis-chosen folder fails loudly and early instead of being written into or wiped.

## [1.5.6] — 2026-07-06

### Added

- **ComfyUI-style seed control** (#40): The seed field in *Quick Generate* and on each *Motion Segment* now has four mode buttons — pin (don't edit), plus / minus (step the seed by 1 after every successful generation), and randomize (new random seed each time). Exactly one mode is always active. Selecting randomize grays the seed field out at -1 and remembers the previous seed; switching back to any other mode restores it. New / duplicated segments and *Sync Seeds* carry the mode over, and the seed actually used is printed to the System Console for reproducibility (it was already recorded in the generation history).

### Changed

- **Generation history rehaul**: Each history row now shows the seed next to the duration (with a `…` marker for multi-segment entries), and the row's stats hug the right so the prompt gets the remaining width. The detail view wraps long prompts across multiple lines instead of cutting them off, and multi-segment generations are listed one numbered line per segment *with the seed each segment actually used* — previously only the first segment's seed was recorded. The duplicated history UI in the Quick Generate and Motion Segments panels was consolidated into one shared implementation.

### Fixed

- **Generate button permanently stuck on "Cancelling…"** (#43): `is_generating` is a scene property, so a file saved mid-generation (or an undo step restoring that state) baked the "generating" flag into the .blend — on reload the Generate button stayed grayed behind a Cancel that had nothing to cancel, and restarting Blender/Kimodo couldn't fix it. The transient generation state is now cleared whenever a file is loaded and once when the addon is enabled (so files locked by older versions heal on the spot), and the Cancel button itself resets the state when no job is actually running.

## [1.5.5] — 2026-06-24

### Fixed

- **Generating motion a second time stopped the animation on Blender 4.x** (#37): Blender 4.4 introduced "slotted" Actions, where an object's `animation_data` must be bound to a specific Action *slot* for the keyframes to play. Each BVH import produced an Action whose slot had a randomized name, so when Kimodo swapped the Action on the reused source armature (i.e. on the second and later generations) Blender couldn't match the previous slot identifier and left no slot bound — the armature went still until the slot was reloaded by hand. Kimodo now gives every Action slot a stable name and binds it explicitly when transferring the Action, so playback survives repeated generations. Blender versions without slots (pre-4.4) fall back to the plain assignment as before.
- **Re-importing piled up duplicate Actions/armature data**: Each time motion was imported onto the reused source armature (re-import from history, repeat generation), the previously assigned Action and the temporary import armature's data-block were left behind as unused orphans, accumulating in the .blend as `Action.001`, `Kimodo_Source.001`, `.002`, and so on. The orphaned Action and armature data are now removed when nothing else references them, so re-importing keeps the file clean.
- **Deleting a constraint removed the wrong row**: The `X` button on a constraint always deleted the *active* row (usually the first one) instead of the row whose `X` was clicked, because it relied on the list's selection index. Each `X` now passes its own row index, so the clicked constraint is the one removed.

## [1.5.4] — 2026-06-18

### Added

- **Choose where the Kimodo venv is installed**: *Install Kimodo* now opens a folder browser so you can pick the install location instead of it silently going to `~/.kimodo-venv`. The venv is created in a `kimodo-venv` subfolder of the chosen directory and the choice is saved to the addon preferences (remembered across restarts). *Retry Install* reuses the chosen location without re-prompting.

### Changed

- **Sampled path waypoints now face along the curve**: *Sample Curve as Waypoints* sets each Root waypoint's heading from the curve's forward direction, so the character walks along the drawn path instead of always facing +Y. The arrow empties are rotated to show the heading in the viewport.
- **Tidier Connection tab**: the Python override, HuggingFace token, and install-location fields moved into a collapsible *Advanced* section for a cleaner default Install → Use → Start flow. Both override capabilities (Python executable path and venv install location) remain available one click away.

## [1.5.3] — 2026-06-15

### Fixed

- **BVH import failed on a fresh Blender 5.0+ install** (#25): Blender 5.0+ ships the legacy BVH importer (`io_anim_bvh`) disabled by default, so `bpy.ops.import_anim.bvh` was "could not be found" and *Generate Motion* crashed with a traceback (originally reported on an RTX 3080ti, but the GPU was unrelated). BVH import now goes through a single helper that enables the importer add-on on demand (detecting registration via `bpy.types.IMPORT_ANIM_OT_bvh`, which—unlike `hasattr(bpy.ops.import_anim, "bvh")`—is accurate, and also enabling extension-namespaced module ids). If the importer genuinely cannot be enabled, callers report a clear, actionable error in the UI instead of leaking a traceback, and the generate path no longer claims success when the import failed.

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
