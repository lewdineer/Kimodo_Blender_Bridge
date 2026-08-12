# Kimodo Blender Bridge — Security & Data-Safety Audit

Scope: full source review of v1.5.7 (`__init__.py`, `operators.py`, `setup_operator.py`,
`subprocess_client.py`, `bridge_server.py`, `retarget.py`, `constraints.py`,
`properties.py`, `panels.py`, `ui_list.py`).

Findings are ranked by severity: how bad the outcome is, how likely a normal user hits
it, and whether it is recoverable.

---

## Severity ranking

| # | Severity | Finding | Impact |
|---|---|---|---|
| 1 | **Critical** | Arbitrary program execution from an untrusted `.blend` file | Code execution as the user |
| 2 | **High** | One-click installer fetches unpinned, unverified third-party code and binaries | Supply-chain code execution |
| 3 | **High** | *Bake & Remove Constraints* strips **every** constraint on the target rig and overwrites its action | Irreversible rig + animation destruction |
| 4 | **Medium** | *Apply Constraints* permanently rewrites `use_inherit_rotation` on the user's armature | Silent rig corruption, never restored |
| 5 | **Medium** | Unescaped path interpolated into generated Python source (`llm2vec_wrapper.py`) | Code injection / broken install |
| 6 | **Medium** | Destructive operators push no undo step | Data loss cannot be undone |
| 7 | **Medium-Low** | `reuse_armature` accepts any armature; Generate overwrites and can delete its Action | Loss of user animation |
| 8 | **Low** | Client deletes any filesystem path the bridge subprocess names | Arbitrary file deletion (post-compromise) |
| 9 | **Low** | Generated BVH/NPZ temp files are never cleaned up | Disk / RAM exhaustion |
| 10 | **Low** | Any add-on module named `*io_anim_bvh` is imported and permanently enabled | Activates dormant code |
| 11 | **Low** | Residual venv-deletion risk after the 1.5.7 guard | Deletion of a non-Kimodo directory |
| 12 | **Info** | Secret handling: HF token at rest, silent `save_userpref()`, token in README | Credential hygiene |

---

## 1. Critical — Arbitrary program execution from an untrusted `.blend` file

**Where:** `properties.py:282`, `operators.py:204-216`, `subprocess_client.py:104,123-137,463-465`

`python_executable` is a `StringProperty` on `KIMODO_SceneSettings`, which is registered as
`bpy.types.Scene.kimodo` (`properties.py:607`). Scene property groups are **serialised into the
`.blend` file** — the project already relies on this fact (see the `is_generating` bug in
CHANGELOG 1.5.6). So the value travels with any `.blend` a user opens.

That value is handed straight to `Popen` with no validation:

```python
# subprocess_client.py:463
if hint and os.path.isfile(hint):
    return hint                      # anything that is a file
...
# subprocess_client.py:126
_proc = subprocess.Popen([python, bridge, "--model", model_name], ...)
```

Nothing checks that the target is a Python interpreter, is inside a venv, is on an allowlist,
or has not changed since the user last set it. There is no confirmation prompt.

**Attack:** ship a `.blend` (asset pack, rig commission, marketplace freebie, tutorial file)
with `scene.kimodo.python_executable` pointing at a payload bundled alongside it — Blender asset
packs are routinely distributed as a zip of `.blend` + support folders, so the payload has a
predictable relative path. The victim opens the file and clicks **Start Kimodo**. The payload
runs as the user.

**Why it is invisible:** `show_advanced_connection` is *also* a scene property, so the attacker
sets it to `False`. With Kimodo installed and the malicious path pointing at a real file,
`panels.py:199` falls through to the `else` branch, which draws only the collapsed *Advanced*
header. The Connection panel looks completely normal — model selector, Start button, nothing else.

This breaks Blender's own security model, where `.blend` files are data and embedded scripts do
not auto-run. The add-on turns a data field into an exec path.

**Fix direction:** treat `python_executable` as untrusted on file load. Reset or re-confirm it in
the existing `load_post` handler (`properties.py:575`, which already sanitises transient scene
state); require the resolved interpreter to answer `sys.version_info` before launching; and show
the resolved path in the panel unconditionally, not behind a collapsed section.

---

## 2. High — Installer fetches unpinned, unverified third-party code and binaries

**Where:** `setup_operator.py:544-552, 891-896, 923-931, 938-943, 958-985`

The advertised one-click install path executes third-party code with **zero integrity checking**:

```python
# setup_operator.py:550-552
return f"git+https://github.com/{owner}/{repo}.git"        # default branch, unpinned
return f"https://github.com/{owner}/{repo}/archive/HEAD.zip"  # mutable HEAD
```

- `Aero-Ex/kimodo` and `nv-tlabs/kimodo-viser` are installed from the **default branch / mutable
  `HEAD.zip`** — no tag, no commit SHA, no hash. `pip install` runs the package's `setup.py`, so
  whatever is on that branch at install time executes as the user.
- `motion_correction-1.0.0-…​.whl` is a **pre-built binary C extension** pulled from a GitHub
  release asset (`setup_operator.py:891-896`). Release assets are mutable — a maintainer (or
  anyone who compromises the account) can delete and re-upload under the same tag and filename.
  No hash, no signature.
- `snapshot_download(repo_id=…)` (`setup_operator.py:960-985`) pins **no `revision=`** and passes
  no `allow_patterns=`, so it fetches whatever the HF repo currently contains. If a
  `pytorch_model.bin` appears there, loading it goes through `torch.load` → pickle → code
  execution.
- `Aero-Ex` is a third-party fork, not the NVIDIA upstream the README and manifest link to.

Every user who follows the documented install path is exposed. None of this is malicious today;
the problem is that there is no mechanism that would *notice* if it changed.

**Fix direction:** pin every install source to a commit SHA or release tag, add
`--require-hashes` / explicit `--hash=` for the wheel, pass `revision=` to both
`snapshot_download` calls, and add `allow_patterns=["*.safetensors", "*.json", …]` so `.bin`
pickles are never fetched.

---

## 3. High — *Bake & Remove Constraints* destroys the entire target rig

**Where:** `retarget.py:328-345`

```python
bpy.ops.pose.select_all(action='SELECT')
bpy.ops.nla.bake(
    frame_start=frame_start, frame_end=frame_end,
    only_selected=False,
    visual_keying=True,
    clear_constraints=True,     # ← Blender: "Remove all constraints from keyed object/bones"
    use_current_action=True,    # ← Blender: "Bake into current action, instead of creating a new one"
    bake_types={'POSE'},
)
```

Two separate destructive behaviours, neither of which matches what the UI promises:

1. **`clear_constraints=True` with `only_selected=False`** removes *all* constraints from *all*
   pose bones — not just the `KIMODO_`-prefixed ones. On any real character rig that means the
   IK chains, Limit Rotation, Damped Track, Stretch To, Copy Rotation twist setups — the whole
   control system — are deleted. The button is labelled "Bake & Remove Constraints" and the
   README says "all Kimodo constraints are removed, leaving a clean, self-contained animation".
   `remove_retargeting_constraints()` (`retarget.py:301`) already does the correct
   prefix-filtered removal; the bake path bypasses it.

2. **`use_current_action=True`** overwrites the rig's active action in place rather than creating
   a new one, so any existing hand-animated action on the target is destroyed within the bake
   range.

`KIMODO_OT_BakeRetargeting` declares no `bl_options`, so it pushes no undo step of its own
(see finding 6). Once the file is saved, the rig is gone.

**Fix direction:** bake with `clear_constraints=False`, then call the existing
`remove_retargeting_constraints(target_arm)`; and default `use_current_action=False` so the bake
lands in a new action.

---

## 4. Medium — *Apply Constraints* permanently rewrites the user's armature data

**Where:** `retarget.py:199-202`, `properties.py:213-223, 254-263`, `operators.py:596-615`

```python
# retarget.py:199-202
if inherit_rotation is not None:
    data_bone = target_arm.data.bones.get(tgt_name)
    if data_bone is not None:
        data_bone.use_inherit_rotation = inherit_rotation
```

`use_inherit_rotation` is a property of the **armature data-block** — part of the rig itself,
shared by every user of that armature, not a transient constraint.

`KIMODO_OT_AutoMapBones` (`operators.py:608-612`) creates each mapping row without setting
`inherit_rotation`, so every row inherits the PropertyGroup default of `True`. Clicking
**Auto-Match Bones → Apply Constraints** therefore forces `use_inherit_rotation = True` on every
matched bone of the user's rig, silently overwriting rigs that deliberately disable it (common
for hip/root/twist bones).

`remove_retargeting_constraints()` restores nothing — the change survives *Remove Constraints*,
*Bake*, and add-on removal. `_on_inherit_rotation_update` (`properties.py:213`) writes the same
property immediately on toggle, so merely clicking around the bone list mutates the rig.

**Fix direction:** snapshot the original value per bone before the first write and restore it in
`remove_retargeting_constraints`; or make the override tri-state (`unset` / `on` / `off`) so
`unset` never touches the rig.

---

## 5. Medium — Unescaped path interpolated into generated Python source

**Where:** `setup_operator.py:688-706` (`_patch_wrapper`), `setup_operator.py:752-765` (`heal_wrapper_path`)

```python
safe_dir = local_dir.replace("\\", "\\\\")
patched  = text.replace(_WRAPPER_PLACEHOLDER, safe_dir, 1)
```

`safe_dir` is dropped inside a Python string literal (`custom_path = r"…"`, per the regex at
`setup_operator.py:752`) in `llm2vec_wrapper.py`, a file the venv Python **imports on every
generation**. Only backslashes are escaped. A `"` in the path closes the literal and everything
after it is parsed as Python:

```
~/rigs/a"; __import__("os").system("…"); x="       →  executes on next generation
```

`local_dir` derives from the install location, which comes from the folder browser or the
free-text *Install Location* preference. Quotes and newlines are legal in POSIX directory names,
so this is a genuine injection primitive — reachable via social engineering ("unzip the pack and
install Kimodo into the included folder") rather than remotely, hence Medium not High.

Secondary correctness bug: the backslash doubling is wrong when the target literal is a **raw**
string, which `heal_wrapper_path`'s own regex (`r?"`) explicitly anticipates. `r"C:\\Users\\x"`
is the literal path `C:\\Users\\x` with doubled separators.

**Fix direction:** emit the value with `repr(local_dir)` (or `json.dumps`) and replace the whole
literal including its quotes, instead of hand-escaping the inside of one.

---

## 6. Medium — Destructive operators push no undo step

**Where:** `operators.py:524, 647, 679, 694, 743, 827, 1862, 1374`; `setup_operator.py`

Blender only pushes an undo step for operators declaring `bl_options = {'REGISTER', 'UNDO'}`.
Only seven operators do (`operators.py:1015, 1518, 1562, 1969, 1984, 2026, 2048`) — and they are
the harmless ones (Sync Seeds, Draw Curve, Set 30 FPS, Clear History…).

Every operator that actually destroys data omits it:

| Operator | Destroys |
|---|---|
| `KIMODO_OT_BakeRetargeting` | all rig constraints + current action (finding 3) |
| `KIMODO_OT_ApplyRetargeting` | `use_inherit_rotation` on armature data (finding 4) |
| `KIMODO_OT_RemoveRetargeting` | constraints |
| `KIMODO_OT_RemoveConstraint` | `bpy.data.objects.remove()` when `delete_object=True` (`operators.py:1886`) |
| `KIMODO_OT_ImportBVH` / `ImportBVHAtFrame` | object + action removal via `_apply_to_existing_source` |
| `KIMODO_OT_LoadPreset` / `ImportPresetFile` | clears `bone_mappings` before it can fail |

The preset importers are the sharpest of these: `operators.py:852` calls
`s.bone_mappings.clear()` **before** assigning parsed values, and the assignment can raise
(`p.get()` on a non-dict, or an out-of-vocabulary `mode` string rejected by the EnumProperty).
A malformed JSON file therefore wipes the user's bone mapping and leaves it unrecoverable.

**Fix direction:** add `bl_options = {'REGISTER', 'UNDO'}` to each; validate and build the full
preset list before clearing the existing one.

---

## 7. Medium-Low — `reuse_armature` accepts any armature and can delete its Action

**Where:** `operators.py:488-517`, `properties.py:368-377`

```python
existing = s.reuse_armature
...
_bind_action_with_slot(existing, new_action)
...
if old_action and old_action is not new_action and old_action.users == 0:
    bpy.data.actions.remove(old_action)
```

The `poll` only requires `obj.type == 'ARMATURE'` — there is no check for the `kimodo_source`
marker that `operators.py:567` sets on armatures the add-on owns. The panel exposes it as a bare
eyedropper labelled **"Reuse"** (`panels.py:408, 484`), which reads naturally as "apply the
motion to this rig". Pointing it at the user's character rig and clicking Generate replaces the
rig's action with a raw Kimodo BVH action (bone names won't match, so the rig collapses) and then
deletes the original action if nothing else references it — with no undo step (finding 6).

**Fix direction:** restrict the `poll` to `obj.get("kimodo_source")`, or warn when the chosen
armature is not Kimodo-generated.

---

## 8. Low — Client deletes any path the bridge subprocess names

**Where:** `subprocess_client.py:264-270`

```python
path = msg.get("path", "")
if path:
    try:
        os.unlink(path)
```

After a cancel, the drainer unlinks whatever path arrives in the bridge's `done` message, with no
check that it is the temp file we expected (no prefix check, no `tempfile.gettempdir()`
containment, no comparison against a recorded path). Today the bridge is our own code, so this is
only reachable once the subprocess is already attacker-controlled — which finding 1 makes
possible. Cheap to harden.

**Fix direction:** record the expected temp path, or require the path to sit under
`tempfile.gettempdir()` with the `kimodo_` prefix before unlinking.

---

## 9. Low — Generated motion files are never cleaned up

**Where:** `bridge_server.py:59`, `_save_output`

```python
fd, out_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
```

Every generation writes a BVH/NPZ to the system temp directory and nothing ever removes it —
only the cancel path unlinks (`subprocess_client.py:268`). *Generate Variations* produces up to 5
per click; history keeps 20 entries but files beyond that are simply orphaned. Each file is a few
MB for a 30-second clip.

On most Linux distributions `/tmp` is **tmpfs, i.e. RAM**, so a long session of iterating on
prompts consumes memory, not disk. There is no cap and no cleanup on `unregister()`.

**Fix direction:** move outputs to a managed subdirectory and prune to the history cap (20), or
delete files whose history entry has been evicted.

---

## 10. Low — Any module named `*io_anim_bvh` is imported and permanently enabled

**Where:** `operators.py:50-68`

```python
candidates += [m.__name__ for m in addon_utils.modules()
               if m.__name__ != "io_anim_bvh" and m.__name__.endswith("io_anim_bvh")]
...
addon_utils.enable(module_name, default_set=True, persistent=True)
```

`addon_utils.modules()` enumerates every module in every add-on search path, including **disabled**
ones — which Blender does not import at startup. `addon_utils.enable()` imports the module
(executing its top-level code) and `default_set=True` writes it into the user's preferences as
permanently enabled.

So a dormant file dropped into an add-ons directory with a name ending in `io_anim_bvh` is
imported and enabled the first time a BVH is generated. This upgrades a file-write into the
add-ons folder — otherwise inert — into code execution, and it silently modifies user preferences
without asking.

**Fix direction:** match against a known allowlist (`io_anim_bvh`,
`bl_ext.blender_org.io_anim_bvh`, `bl_ext.system.io_anim_bvh`) instead of a suffix wildcard.

---

## 11. Low — Residual venv-deletion risk after the 1.5.7 guard

**Where:** `setup_operator.py:152-175` (`_looks_like_kimodo_venv`), `104-121` (`managed_venv`)

The 1.5.7 `_safe_rmtree` guard is a solid improvement, but two gaps remain:

1. **Name + marker collision.** A directory qualifies for deletion if its basename is
   `kimodo-venv` / `.kimodo-venv` **and** it contains any of `{pyvenv.cfg, bin, Scripts, lib,
   lib64, include}`. The README explicitly documents a manual-install flow ("If you already have
   Kimodo installed in your own venv…"), and a hand-built Kimodo venv is very likely to be named
   exactly `kimodo-venv`. It has no sentinel, so `is_installed()` is `False` and *Retry Install*
   offers to wipe it. The confirmation dialog names the path, so this is a warning-shot rather
   than a silent wipe.

2. **The sentinel is a delete permit.** `managed_venv()` returns the free-text *Install Location*
   **verbatim** — the `kimodo-venv` subfolder is only appended on the folder-browser path
   (`setup_operator.py:1112-1114`). A location typed directly into the preference field therefore
   receives the venv *and* the `.kimodo_install_complete` sentinel at its root
   (`setup_operator.py:1004`). From then on `_looks_like_kimodo_venv` returns `True` for that
   directory unconditionally, and *Delete Venv* will `rmtree` the whole thing.

**Fix direction:** always append the `kimodo-venv` subfolder in `managed_venv()`, regardless of
how the location was set, so the sentinel can never land in a directory the user also uses for
anything else.

---

## 12. Informational — Secret handling

- **`hf_token`** (`properties.py:519-528`) uses `subtype='PASSWORD'`, which only masks the widget.
  It is stored in plaintext in Blender's userpref file. It is passed to subprocesses via the
  environment (`setup_operator.py:613`) — better than a command line, and `/proc/<pid>/environ` is
  owner-only, so this is acceptable; the at-rest storage is the weak point. Worth a note in the
  tooltip that a **read-only** token should be used.
- **`bpy.ops.wm.save_userpref()`** (`setup_operator.py:997`) fires from a timer after install and
  silently commits *all* pending preference edits the user may not have intended to persist.
- **`README.md:9-11`** embeds a star-history `sealed_token` in three image URLs. Low risk (these
  are scoped, read-only chart tokens) but it is a credential in version control; worth confirming
  it cannot be replayed for anything else.

---

## Not findings

Checked and clean:

- No `shell=True`, `eval`, `exec`, `pickle`, or `os.system` anywhere in the add-on.
- Subprocess argument lists are constructed as lists — no shell injection surface.
- Prompts and constraints reach the model as JSON via a temp file; no interpolation into code or
  commands.
- `tempfile.mkstemp` is used correctly (no predictable-name race).
- `_is_protected_path` correctly covers POSIX roots, `$HOME` and its parent, Windows drive roots,
  the user profile, environment-resolved system folders, and mount points.
