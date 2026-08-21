# Feasibility study: adding NVIDIA ARDY support to Kimodo Blender Bridge

**Status:** research complete — and since implemented. ARDY now ships as a second
backend (v1.6.0); this document is kept as the reasoning behind that design and
as the reference for what is still unverified. Where it says "would need" below,
read it as what the implementation did.
**Answers:** issue [#52 — "NV ardy support"](https://github.com/lewdineer/Kimodo_Blender_Bridge/issues/52).
**Sources:** add-on source at `main`; [`nv-tlabs/ardy`](https://github.com/nv-tlabs/ardy)
cloned and read at commit `693f74d13b3d04a0a22ce127ee79c929dd89756b`; the ARDY and Kimodo
READMEs. Every code citation below was checked against those trees.

---

## Verdict

**Yes — feasible, and the add-on's architecture already fits.** The two-process design
(Blender ⇄ venv subprocess over newline-delimited JSON) is exactly what ARDY needs, ARDY's
Python API is a near-sibling of Kimodo's, and — the biggest single piece of good news — the
constraint JSON the add-on already writes is **directly loadable by ARDY without changes**.

Four concrete gaps stand between here and a working backend. All are solvable; none is a
blocker. Ranked by effort:

| # | Gap | Severity |
|---|---|---|
| 1 | ARDY has **no BVH writer** — it emits `.npz` only, and the add-on imports BVH only | Largest code item (~150–250 lines) |
| 2 | **No SOMA checkpoint is released** — only Core and G1, so every SOMA-specific table in the add-on needs a sibling | Medium, and possibly throwaway work |
| 3 | **Core runs at 20 FPS**, not 30 — the hardcoded 30 FPS assumptions must become per-backend | Small but touches several files |
| 4 | **Install is harder than Kimodo's** — gated Llama-3-8B, ~14 GB encoder VRAM, CMake/MSVC build with no skip hatch | Highest support-burden risk |

**Recommendation: add ARDY as a second backend, do not replace Kimodo.** Reasoning in
[§6](#6-recommended-shape-second-backend-not-a-replacement).

---

## 1. How the add-on is built, and where a backend seam would go

The add-on is ~6,900 lines of Python in a flat package. There are no tests, no CI and no
`pyproject.toml`; quality control is code review plus a disciplined `CHANGELOG.md`.

| Piece | Where | Why it matters here |
|---|---|---|
| Bridge protocol | `bridge_server.py` — `main()` :247, command switch :317–333 | Four commands: `ping` / `generate` / `generate_multi` / `quit`. **All model-specific code lives in this one file.** This is the natural seam. |
| Blender-side client | `subprocess_client.py` — `start()` :88, `generate_motion()` :334 | Launches `[python, bridge_server.py, --model, NAME]` and speaks JSON. Backend-agnostic apart from the `--model` string and the `[Kimodo Bridge]` log prefix. |
| Installer | `setup_operator.py` — `_do_install()` :788 | 13 imperative steps: venv → PyTorch (cu121/cu124/cu128 chosen by GPU compute capability) → prebuilt `motion_correction` wheel → Kimodo (Aero-Ex fork) → LLM2Vec encoder → weights → sentinel file. |
| Scene → constraints | `constraints.py` — `build_constraints_json()` :345 | Emits blocks of `{type, frame_indices, smooth_root_2d, root_positions, local_joints_rot, global_root_heading}`. Hardcoded `SOMA_JOINT_ORDER` :49, `SOMA_JOINT_PARENTS` :84, `EFFECTOR_IDX` :130. |
| Pose extraction | `constraints.py` — `get_armature_joint_rots()` :212 | Rest-orientation-invariant local-rotation extraction from any posed armature. **Skeleton-agnostic — reusable unchanged.** |
| Frame mapping | `constraints.py` — `blender_frame_to_kimodo()` :187 | Already takes an FPS argument rather than assuming 30. |
| Import | `operators.py` — `_import_bvh()` :71, `KIMODO_OT_ImportBVH` :524 | BVH only (`axis_forward='-Z'`, `axis_up='Y'`, `global_scale=0.01`). NPZ is explicitly "import manually". |
| Retarget | `retarget.py` — `_SOMA_BONE_MAP_HINTS` :49 | Fuzzy-matches SOMA bone names onto Mixamo/Rigify rigs. |
| Model picker | `properties.py` — `kimodo_model` :291 | Already an `EnumProperty` with two entries marked "Unsupported atm" — the natural home for ARDY entries. |
| FPS warning | `panels.py` :354 | Hardcodes the string *"Kimodo needs 30 FPS"*. |

**Licensing.** The add-on is GPL-2.0-or-later; ARDY's code is Apache-2.0 and its weights are
under the NVIDIA Open Model Agreement. Because ARDY would run in a separate process and a
separate venv — exactly as Kimodo does today — the same arrangement carries over with no new
licence interaction.

---

## 2. What ARDY actually is

ARDY (*Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion
Generation*, SIGGRAPH) is NVIDIA's real-time sibling to Kimodo. Kimodo's own README describes
it as "real-time motion generation with Kimodo's controllability".

**It is genuinely released, not a placeholder.** The repo contains the full inference stack,
a CLI (`scripts/generate.py`), an interactive viser demo, an ONNX/TensorRT export path, and
four HuggingFace checkpoints that download automatically.

### 2.1 The API is a near-sibling of Kimodo's

Same names, same shapes, in many cases the same function signatures:

| Kimodo (used by `bridge_server.py`) | ARDY equivalent |
|---|---|
| `from kimodo import load_model` | `from ardy.model import load_model` |
| `kimodo.constraints.load_constraints_lst(path, skeleton)` | `ardy.constraints.load_constraints_lst(path, skeleton)` (`ardy/constraints.py` :425) |
| `kimodo.tools.seed_everything` | `ardy.tools.seed_everything` |
| `kimodo.skeleton.SOMASkeleton30`, `global_rots_to_local_rots` | `ardy.skeleton.SOMASkeleton30`, `global_rots_to_local_rots`, `to_standard_tpose` |
| `kimodo.exports.bvh.save_motion_bvh` | **no equivalent — see §3.1** |

### 2.2 The call shape differs

Kimodo does everything in one call. ARDY splits it into three steps
(`ardy/model/ardy_model.py` `__call__` :531, and `scripts/generate.py` :251–279):

```python
motion = model(texts, num_frames,
               num_denoising_steps=…, pad_mask=…, first_heading_angle=…,
               motion_mask=…, observed_motion=…,
               cfg_weight=(text_weight, constraint_weight),
               crop_history_length=…)
output = model.motion_rep.inverse(motion, is_normalized=True)
output.update(post_process_motion(output["local_rot_mats"], output["root_positions"],
                                  output["foot_contacts"], model.skeleton, constraint_lst))
```

Constraints are not a `constraint_lst=` kwarg; they go through
`model.motion_rep.create_conditions_from_constraints_batched(...)` to produce
`observed_motion` / `motion_mask`. Post-processing is an explicit, **optional** call rather
than a `post_processing=True` flag.

Two new knobs are worth exposing in the UI: `cfg_weight` (separate text and constraint
guidance scales) and `crop_history_length` (how much history each autoregressive step sees —
smaller adapts to new prompts faster, larger gives smoother transitions).

### 2.3 The constraint JSON is already compatible

This is the finding that makes the whole integration cheap. `ardy/constraints.py`
`TYPE_TO_CLASS` (:414) registers exactly:

```
root2d · fullbody · left-hand · right-hand · left-foot · right-foot · end-effector
```

— the same type strings the add-on's `build_constraints_json()` emits. Better still,
`Root2DConstraintSet.from_dict`, `FullBodyConstraintSet.from_dict` and
`EndEffectorConstraintSet.from_dict` all accept **`smooth_root_2d` as an alias for
`root_2d`**, and read `frame_indices`, `local_joints_rot`, `root_positions` and
`global_root_heading` — precisely the keys `constraints.py` already writes.

So the Motion Constraints panel, the curve-sampling waypoints, the full-body pose authoring
and the `KIMODO_` coordinate conversions all carry over. Only the *joint index tables*
change, because those depend on which skeleton the model uses.

---

## 3. The four gaps, in detail

### 3.1 No BVH writer — the largest code item

`ardy/exports/` contains only `__init__.py` and `mujoco.py`. `ardy/skeleton/bvh.py` is a BVH
**parser**, not a writer. ARDY writes `.npz` containing `posed_joints`, local/global joint
rotations, root positions and foot contacts — plus a MuJoCo qpos `.csv` for G1. There is no
FBX, glTF or USD path either.

The add-on's entire import route is `bpy.ops.import_anim.bvh`.

**This is solvable and not especially hard.** Everything a BVH writer needs is present on the
skeleton object (`ardy/skeleton/base.py`):

- `neutral_joints` (:73) — rest joint positions, i.e. the BVH `OFFSET` values
- `joint_parents` (:94) — the hierarchy
- `bone_order_names` — the `JOINT` names
- `fk()` (:229) and `global_rots_to_local_rots()` for channel data
- plus `output["local_rot_mats"]` and `output["root_positions"]` from generation

Kimodo's `save_motion_bvh` is Apache-2.0 and can be adapted with attribution. Estimate:
**150–250 lines** in the bridge, plus a round-trip test.

The alternative — reading `.npz` in Blender and building the armature directly — is worse: it
throws away the existing import, reuse-armature, history and retarget plumbing for no gain.

### 3.2 No SOMA checkpoint — Core27 is what exists today

`ardy/model/registry.py` :23 defines:

```python
MODELS_BY_SKELETON = {
    "core": {40: "ARDY-Core-RP-20FPS-Horizon40", 8: "ARDY-Core-RP-20FPS-Horizon8"},
    "g1":   {52: "ARDY-G1-RP-25FPS-Horizon52",  8: "ARDY-G1-RP-25FPS-Horizon8"},
}
```

`"soma"` appears in `DEFAULT_HORIZON` (:35) and in docstrings, but **not in
`MODELS_BY_SKELETON`** — so `resolve_model_name("soma")` raises `ValueError`. The README
lists the SOMA variant as "coming soon".

The practical target is therefore **`CoreSkeleton27`** (`ardy/skeleton/definitions.py` :338).
Its bone names are Mixamo-style:

```
Hips · Spine · Spine1 · Spine2 · Spine3 · Neck · Head
RightShoulder · RightArm · RightForeArm · RightHand · RightHandEnd · RightHandThumb1
LeftShoulder  · LeftArm  · LeftForeArm  · LeftHand  · LeftHandEnd  · LeftHandThumb1
RightUpLeg · RightLeg · RightFoot · RightToeBase
LeftUpLeg  · LeftLeg  · LeftFoot  · LeftToeBase
```

That is **more convenient than SOMA for retargeting**, because those names are already the
right-hand side of most entries in `_SOMA_BONE_MAP_HINTS`. But it means new
`CORE_JOINT_ORDER` / `CORE_JOINT_PARENTS` / `EFFECTOR_IDX` tables in `constraints.py` and a
`_CORE_BONE_MAP_HINTS` table in `retarget.py`.

Also note: `ardy/assets/skeletons/cskel27/` contains only `joints.p` and
`skin_standard.npz` — **no `standard_t_pose_global_offsets_rots.p`**, which
`to_standard_tpose()` requires. So the existing **Use Standard T-Pose** option (already
documented as SOMA-only) must be hidden or disabled for Core27.

G1 is a Unitree humanoid robot skeleton with a MuJoCo export path. It is out of scope for a
character-animation add-on and should not be offered.

### 3.3 FPS is not 30

Core runs at **20 FPS**, G1 at 25 FPS. Kimodo is 30. The add-on hardcodes 30 in the panel
warning (`panels.py` :354) and defaults `kimodo_fps` to 30. `blender_frame_to_kimodo()`
(:187) is already parameterised, so the fix is confined to making the FPS a per-backend
property that the warning, the **Set to 30 FPS** operator and the constraint builder all read.

### 3.4 Install friction — the biggest support risk

**Gated text encoder.** ARDY's default encoder preset (`ardy/model/load_model.py` :21) is
`McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp` in **bfloat16**, which depends on the
gated `meta-llama/Meta-Llama-3-8B-Instruct`. The README quotes **~14 GB VRAM** for the
default `cuda / bfloat16` mode (CPU modes available, slower). Today the add-on avoids the
gate entirely by pulling Aero-Ex's ungated NF4 mirror, and `bitsandbytes` is not an ARDY
dependency — so an HF token and an accepted Llama licence would become **mandatory** for ARDY
users. The add-on already has an HF-token field, so the UI cost is small; the user-friction
cost is not.

**The encoder is two downloads, not one.** Both McGill repos contain *only* a LoRA adapter
(`adapter_config.json` + `adapter_model.safetensors`, ~168 MB) together with a config and
tokenizer. The base weights are not in them: they come from the gated
`meta-llama/Meta-Llama-3-8B-Instruct` (~16 GB) and must land in the **same directory** as the
adapter, which is the layout `LLM2Vec.from_pretrained` expects — it loads the checkpoint from
that folder with `LlamaBiModel.from_pretrained(dir)` and then applies the adapter from the same
folder (see the "special case where config.json and adapter weights are in the same directory"
branch in `ardy/model/llm2vec/llm2vec.py`). This is what ARDY's README means by requiring Llama
access, and it is easy to miss: fetching only the McGill repos produces a directory that looks
complete and fails at load time with `OSError: no file named model.safetensors`.

> **Corrected 2026-08-20.** An earlier revision of this study recommended
> `TEXT_ENCODERS_DIR` as a "cleaner than `_patch_wrapper()`" way to stage the encoder, and
> treated the McGill repo as self-contained. The first ARDY install built on that advice shipped
> without base weights and could not start. `TEXT_ENCODERS_DIR` is also undocumented — it appears
> only in `ardy/model/llm2vec/llm2vec_wrapper.py` :29–31 and nowhere in ARDY's README or demo — so
> the layout it expects has to be inferred from the loader. It does work, but only with the base
> weights alongside the adapter, and the installer now verifies that before declaring success
> (`ardy_setup.py` `_require_weights()`).

> **Still unverified:** whether the existing ungated NF4 checkpoint can stand in for the gated
> base weights, which would remove the licence step entirely. `bitsandbytes` is not an ARDY
> dependency and whether a pre-quantized checkpoint loads through `LLM2Vec.from_pretrained` with
> `torch_dtype=bfloat16` has not been tested.

**Native extension build.** ARDY's `setup.py` builds `motion_correction` via a
`CMakeExtension` (:14) on every install, requiring CMake ≥ 3.15 and a C++17 compiler. Crucially
there is **no `SKIP_MOTION_CORRECTION_IN_SETUP` escape hatch** — that flag is a feature of the
Aero-Ex *Kimodo* fork, which is why the current installer can drop in a prebuilt wheel and
skip the build. Options:

1. **Fork ARDY** and add the skip variable, then reuse a prebuilt wheel — the pattern this
   repo already depends on for Kimodo. Adds a fork to maintain.
2. **Ship without post-processing.** `post_process_motion` is optional: the import is lazy
   (`ardy/postprocess.py` :309), the CLI has `--no-postprocess`, and it is disabled outright
   for G1. Cost: more foot skating. Good enough for a v1 / experimental backend.

**Dependency pinning.** ARDY pins `transformers==5.8.1` exactly and caps `numpy<2`. ARDY must
get **its own venv** (`~/.ardy-venv`), never a shared one with Kimodo.

**Platform.** ARDY is tested on Ubuntu 22.04 / RTX 4090 / Python 3.11 / driver 575. `setup.py`
does contain Windows MinGW handling, so Windows is contemplated but not the tested path.

---

## 4. Timeline mode is feasible, but not free

Kimodo stitches multiple prompts in a single call (`multi_prompt=True`,
`num_transition_frames=5`) — that is what `bridge_server._generate_multi()` :169 uses.

ARDY's `__call__` takes **one prompt and one length**. However it accepts
`init_history_sequence=` (:548, handled by `_encode_init_history()` :367), which takes a
normalized motion-rep tensor. So segments chain: generate segment *n*, feed its tail in as the
history for segment *n+1* under the next prompt. Transitions come from the autoregressive
history rather than an explicit blend length.

One caveat to design around: `crop_history_length` and `init_history_sequence` are mutually
exclusive (assert at :609), so history must be cropped before it is passed in.

The interactive demo (`scripts/interactive_demo/generation.py` `_generate_step` :180, calling
`model.autoregressive_step()`) is the reference implementation, and its GUI already models a
timeline of per-frame-range prompts — conceptually the same thing as the add-on's Motion
Segments.

---

## 5. What ARDY buys the add-on

Being honest about the upside, since issue #52 claims it is "better than kimodo":

- **Streaming / interactive generation.** ARDY is autoregressive, so motion can extend
  indefinitely and react to prompt changes mid-stream. Kimodo generates a fixed clip.
  **Shipped in 1.7.0** as the Live Stream panel — see §5.1.
- **Faster response.** Real-time on an RTX 4090, with TensorRT and `torch.compile`
  acceleration paths. Kimodo's 100-step diffusion is offline by comparison.
- **Toes and Mixamo-style names** out of the box on Core27 — relevant to issue #49.
- **Genuinely long sequences** without the transition-frame stitching Kimodo needs.

Against that, today: no SOMA checkpoint, no BVH export, 20 FPS, gated encoder, harder install.
"Better" is not yet true *for this add-on's use case* — it is true for the interactive one.

### 5.1 Live streaming, as built (1.7.0)

The Live Stream panel drives `Ardy.autoregressive_step()` from Blender's playhead. The design
follows the interactive demo closely, because the demo is the only reference for how ARDY is
meant to be run continuously.

**The replan is the whole trick.** `GenerationMixin._get_history_motion`
(`scripts/interactive_demo/generation.py` :157-178) sets
`history_end_idx = min(cur_motion_len - 1, frame_idx + replan_buffer_size)` — every step
*discards already-generated future motion* just ahead of the playhead and regenerates it against
the current constraints. That is what makes moving a target feel live rather than queued, and it
maps onto Blender as `frame_idx = scene.frame_current - stream_start_frame`, with the discarded
frames' keyframes overwritten in place. `ardy_stream_replan_buffer` is that commit window.

**Steering is a path, not a waypoint.** Pinning one constraint at the end of the horizon asks
the model to teleport and fights the motion prior. The demo instead projects future root
positions from a target velocity, easing out of the current velocity
(`_update_root_constraints_from_target_velocity`, `gen_constraints.py` :156). `ardy_steer.py` is
that idea reduced to "steer toward a point", kept free of torch/numpy/bpy so it is unit-testable.

**Protocol.** `stream_begin` / `stream_step` / `stream_end`, one request and one reply each — no
server push, so the client's threading, cancellation and single-in-flight rules did not change
and Kimodo's path was untouched. Frames cross the pipe as axis-angle (joints × 3 floats per
frame, a few KB per window).

**The rest pose is deliberately plain.** The stream armature's bones all point +Y with zero
roll, which makes every bone's rest rotation the identity — so a pose bone's quaternion *is*
ARDY's local joint rotation, with no per-bone change of basis. An oriented rest (as a BVH import
produces) would need every rotation conjugated by `bone.matrix_local`, and an error there is
silent and subtly wrong rather than loud. The cost is cosmetic; the rig's job is to drive a
retarget.

**Known limits.** Latency is the binding constraint: a window must generate faster than it
plays (0.4 s for `core8`, 2 s for the 40-frame default). When it does not, playback pauses
rather than running into un-keyframed frames. `ardy_stream_diffusion_steps` defaults to 8 for
this reason. Whether a given GPU keeps up is not something the test suite can answer — the
panel reports measured seconds-per-step so the user can see it directly.

---

## 6. Recommended shape: second backend, not a replacement

Keep Kimodo as the default and add ARDY alongside it, selectable per scene.

Kimodo currently has the released SOMA checkpoint, 30 FPS, a working BVH exporter, an
ungated encoder mirror and a prebuilt native wheel. ARDY has none of those. Replacing Kimodo
would be a downgrade for existing users today, and the two models suit different jobs — Kimodo
for authored clips with constraints, ARDY for long or interactive motion.

### Suggested phasing

**Phase 1 — backend abstraction, zero behaviour change.**
Add `--backend {kimodo,ardy}` to `bridge_server.py` and move the Kimodo-specific pieces
(`_save_output`, `_load_constraints`, `_generate`, `_generate_multi`, the model load in
`main()`) behind a small interface. Keep the JSON protocol and every `status` value byte-identical
so `subprocess_client.py` needs no edits at all. Parameterise the SOMA tables in
`constraints.py` into a per-skeleton table passed into `build_constraints_json()`.
*Acceptance: Kimodo output is bit-identical before and after.*

**Phase 2 — ARDY bridge backend.**
`load_model` → `model(...)` → `motion_rep.inverse` → optional `post_process_motion` → the new
BVH writer. Constraints reuse the existing JSON, then go through
`create_conditions_from_constraints_batched`. `generate_multi` via chained
`init_history_sequence`. Disable `bvh_standard_tpose` for Core27.

**Phase 3 — installer and UI.**
A second install path building `~/.ardy-venv`, reusing `_run()`, `_download_with_retry()`,
`_safe_rmtree()`, `_is_protected_path()` and `_max_gpu_compute_capability()` unchanged — the
1.5.7 deletion-safety work applies as-is. Add a `backend` `EnumProperty`, ARDY model entries,
a per-backend `native_fps`, `cfg_weight` / `history_frames` controls, and
`_CORE_BONE_MAP_HINTS` in `retarget.py`.

### Verification an implementation would need

1. **Regression:** same prompt and seed through `--backend kimodo` before and after Phase 1;
   diff the BVH. Must be identical.
2. **Smoke-test ARDY outside Blender first:**
   `python scripts/generate.py "a person walks in a circle." --model core --seed 0` in the
   ARDY venv. This proves the environment, weights, HF token and VRAM before any add-on code
   is involved. **Do this before writing anything.**
3. **BVH round-trip:** generated `.npz` → BVH → `bpy.ops.import_anim.bvh` → compare imported
   joint world positions against `output["posed_joints"]` to ~1 mm, after the `0.01` scale and
   the Z-up ↔ Y-up conversion.
4. **Constraints:** author a Root XZ waypoint and a Full-Body keyframe, generate, confirm the
   armature passes through them at the requested frames.
5. **Timeline:** three chained segments; inspect the seams for pops.
6. **Manual clean install** on Windows and Linux. The install path is the likeliest failure
   and has no automated coverage in this repo.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| ARDY-SOMA ships and obsoletes the Core27 table work | Sequence the BVH writer, backend split and installer first — none of those are wasted. See §8. |
| CMake / MSVC requirement drives Windows support tickets | Decide early: fork ARDY for a skip flag, or ship v1 without post-processing. |
| ~14 GB encoder VRAM makes 8 GB cards CPU-encoder-only and slow | Surface the CPU encoder option prominently, as the Kimodo README already does. |
| Gated Llama licence blocks users at install time | Detect the 401/403 and give a direct link to the model page plus the token field, rather than a raw traceback. |
| Two venvs (~10 GB each) on disk | Make the ARDY install opt-in, never part of the default flow. |

---

## 8. What changes if NVIDIA ships ARDY-SOMA

The README promises a SOMA variant. If it lands:

- **Gap 3.2 largely disappears.** `SOMASkeleton30` and `somaskel77` assets are *already
  present* in the ARDY repo, including `standard_t_pose_global_offsets_rots.p` — so
  `to_standard_tpose()` would work, the **Use Standard T-Pose** option would apply, and the
  existing `SOMA_JOINT_ORDER` / `SOMA_JOINT_PARENTS` / `EFFECTOR_IDX` tables and
  `_SOMA_BONE_MAP_HINTS` would be reused verbatim.
- `scripts/generate.py` already calls `skeleton.output_to_SOMASkeleton77(output)` for SOMA
  models — the same 77-joint form Kimodo's BVH exporter consumes.
- **Everything else still applies.** The BVH writer (3.1), the FPS parameterisation (3.3) and
  the install work (3.4) are all needed regardless.

That is the argument for doing the backend split and the BVH writer first and the Core27
tables last: the first two are permanent, the third may be superseded.

---

## 9. What was actually built

Implemented in v1.6.0, following §6:

| Gap | How it was closed |
|---|---|
| No BVH writer | `ardy_bvh.py` — writes from `neutral_joints` / `joint_parents` / `local_rot_mats`, in centimetres with ZXY Euler channels and an End Site on every leaf. Verified against forward kinematics to ~4e-8 m. |
| No SOMA checkpoint | Core27 tables added to `constraints.py` behind a skeleton registry; `retarget.py` gained hints for the joints SOMA lacks. The tables are checked against ARDY's own `CoreSkeleton27` in the test suite. |
| FPS is not 30 | The bridge reports its real rate in the `ready` message; the warning, the set-FPS operator and constraint frame mapping all read it. |
| Install friction | `ardy_setup.py` — own venv, GPU-matched PyTorch, Llama access verified before any download, and a toolchain-aware install that drops the CMake extension when no compiler exists. |
| Timeline mode | Chained `init_history_sequence` in `ardy_bridge.py`, following the interactive demo's accumulation exactly, with each segment trimmed to its exact frame count. |

**Still unverified — needs a machine with an NVIDIA GPU.** Everything above was
tested against mocks that reproduce the released model's call contract, not
against real weights. Specifically untested: a real `load_model`, generation
quality, whether the written BVH imports cleanly into Blender, retarget results
on a real rig, and the installer end to end on Windows and Linux. The
verification list in §6 still stands, and the "the Aero-Ex NF4 encoder may load
under ARDY" idea in §3.4 was **not** pursued — the implementation uses ARDY's
own encoder preset and requires the token.

---

## 10. Bottom line

Implementing ARDY is a **medium-sized, well-bounded piece of work** — roughly: a backend split
in `bridge_server.py`, a BVH writer, a skeleton table, a second installer path, and some UI
plumbing. Nothing about it is architecturally awkward, and the constraint-JSON compatibility
removes what would otherwise have been the hardest part.

The honest caveat for issue #52 is that **ARDY is not a drop-in upgrade today**. Until the
SOMA checkpoint ships, ARDY means a different skeleton, a lower frame rate, no standard T-pose,
a gated 14 GB text encoder and a compiler on the install path. It is worth adding as a second
backend for its streaming and long-sequence strengths — not as a replacement.
