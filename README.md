

A Blender addon that generates AI-driven human motion via [NVIDIA Kimodo](https://github.com/nv-tlabs/kimodo) or [NVIDIA ARDY](https://github.com/nv-tlabs/ardy) and imports it directly into your scene — no copy-pasting, no manual BVH wrangling.

## Star History

<a href="https://www.star-history.com/?repos=lewdineer%2FKimodo_Blender_Bridge&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=lewdineer/Kimodo_Blender_Bridge&type=date&theme=dark&legend=top-left&sealed_token=gg3e8bjBVO1mASsY7F2FfwXezEB5zKIehjMw-s8K0AeGqeg94JCIRoPyBtxv55OxO9Pvi5cFwEyXXpA7wOwC8khA42N0EIz--WX4-WLBhTkFav2p97mv0gNKScQBzfaLta9xHnw9ELr6Ntm0RTGEbBvwvSp8RubVcK10aKSlsQs3pwmd4LuWqQp0kpux" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=lewdineer/Kimodo_Blender_Bridge&type=date&legend=top-left&sealed_token=gg3e8bjBVO1mASsY7F2FfwXezEB5zKIehjMw-s8K0AeGqeg94JCIRoPyBtxv55OxO9Pvi5cFwEyXXpA7wOwC8khA42N0EIz--WX4-WLBhTkFav2p97mv0gNKScQBzfaLta9xHnw9ELr6Ntm0RTGEbBvwvSp8RubVcK10aKSlsQs3pwmd4LuWqQp0kpux" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=lewdineer/Kimodo_Blender_Bridge&type=date&legend=top-left&sealed_token=gg3e8bjBVO1mASsY7F2FfwXezEB5zKIehjMw-s8K0AeGqeg94JCIRoPyBtxv55OxO9Pvi5cFwEyXXpA7wOwC8khA42N0EIz--WX4-WLBhTkFav2p97mv0gNKScQBzfaLta9xHnw9ELr6Ntm0RTGEbBvwvSp8RubVcK10aKSlsQs3pwmd4LuWqQp0kpux" />
 </picture>
</a>

## Links / Video guide
Youtube tutorial: https://youtu.be/nbiaS43Ncng
Superhive: https://superhivemarket.com/products/kimodoblenderbridge

---

## Tested on 
**Blender 5.1 / 4.4**

**Arch Linux RTX 3090 Python 3.12**

**Windows 11 RTX 1080 Python 3.12**

**Windows 11 RTX 5070 Python 3.13**

<img width="1237" height="1257" alt="image" src="https://github.com/user-attachments/assets/71a1666d-a460-40eb-af23-1dbb8ab750cb" />


---

## How it works

Blender's embedded Python cannot load PyTorch, Kimodo or ARDY directly. The addon solves this with a two-process bridge:

```
Blender (addon)                 model venv
  subprocess_client.py  ─────▶  bridge_server.py   (Kimodo)
                        ◀─────  ardy_bridge.py     (ARDY)
                                 model loaded once, handles requests
```

The bridge server loads the model once at startup and then responds to generation requests over stdin/stdout. Blender stays responsive while generation runs in a background thread.

---

## Choosing a backend

Pick one at the top of the **Connection** panel. They install into separate venvs and can both be present; only one runs at a time.

| | **Kimodo** (default) | **ARDY** |
|---|---|---|
| Skeleton | SOMA | Core — Mixamo-style bone names, with toes |
| Frame rate | 30 FPS | 20 FPS |
| How it generates | One-shot diffusion over the whole clip | Autoregressive — continues from what it already made |
| Timeline mode | Segments blended over N transition frames | Segments continue from generation history |
| Standard T-pose export | Yes | No — the Core skeleton ships no T-pose reference |
| Text encoder | Pre-quantized, no HuggingFace account needed | Gated Llama 3 — needs an account and a token |
| Install size | ~10 GB | ~25 GB |
| Best at | Authored clips with precise constraints | Long sequences and fast iteration |

> **Which should I use?** Start with Kimodo. Try ARDY when you want long or rapidly-iterated motion, or when your target rig uses Mixamo-style names — the Core skeleton auto-maps onto those more cleanly.

> NVIDIA has announced an ARDY SOMA checkpoint but not released it. Until it lands, ARDY means the Core skeleton at 20 FPS. See [`docs/ardy-feasibility.md`](docs/ardy-feasibility.md) for the full analysis.

---

## Requirements

| Requirement | Notes |
|---|---|
| Blender 4.0+ | Tested on 5.1 (Windows & Arch Linux) |
| Python 3.10–3.12 | System Python used to create the managed venv |
| NVIDIA GPU | 8 GB+ VRAM recommended; 16 GB+ for best results |
| CUDA | Must match your PyTorch build (installed automatically for your GPU) |
| ~10 GB disk | For the managed venv, model weights, and LLM2Vec encoder |

**ARDY additionally needs:**

| Requirement | Notes |
|---|---|
| HuggingFace account | Its text encoder is built on the gated Meta-Llama-3-8B-Instruct |
| ~25 GB disk | The encoder alone is ~16 GB |
| CMake + a C++ compiler | *Optional.* Only needed for foot-skate cleanup — the installer detects this and continues without it, saying so in the panel |

> **Low VRAM?** For Kimodo, run `kimodo_textencoder --device cpu` in a separate terminal with the Kimodo venv activated. For ARDY, set **Encoder** to *CPU (low VRAM)* in the Connection panel — its GPU encoder needs ~14 GB of VRAM on its own.

---

## Installation

As of v1.2.0 Kimodo installs itself automatically — no terminal required. The addon is also compatible with the Blender 4.2+ Extensions platform (includes `blender_manifest.toml`).

### 1 — Install the Blender addon

1. Download or clone this repository (top-right **Code → Download ZIP**).
2. Open Blender → **Edit → Preferences → Add-ons → Install from Disk…**
3. Select the downloaded zip and enable **"Kimodo Motion Generator"**.

### 2 — Click "Install Kimodo (Auto)"

Open the **Kimodo** N-Panel tab (press `N` in the 3D Viewport), expand **Connection**, and click **Install Kimodo (Auto)**.

The installer will:
- Create a managed Python venv at `~/.kimodo-venv/`
- Install PyTorch (CUDA 12.1), all Kimodo dependencies, and the [Aero-Ex offline fork](https://github.com/Aero-Ex/kimodo)
- Download the LLM2Vec text-encoder model locally and patch it for offline use
- Download the `Kimodo-SOMA-RP-v1` model weights into the HF cache
- Set the Python path automatically when done

Progress is shown live in the Connection panel. The full log is printed to the system console (launch Blender from a terminal, or on Windows use **Window → Toggle System Console**).

> **Requires:** Python 3.10–3.12 on your system PATH, internet access, and ~10 GB of free disk space. After the initial install, Kimodo runs fully offline.

### Installing ARDY

Switch the **Connection** panel to **ARDY** and follow the checklist it shows.

Before the Install button unlocks you need a HuggingFace token with Llama 3 access — the panel links both pages directly:

1. Open the **Meta-Llama-3-8B-Instruct** page and accept the licence (free; approval is usually quick).
2. Create a **read** token in your HuggingFace token settings.
3. Paste it into the token field in the panel.
4. Click **Install ARDY (Auto)** and pick a folder.

The installer then creates a venv, installs PyTorch matched to your GPU, installs ARDY, downloads the text encoder and the `ARDY-Core-RP-20FPS-Horizon40` checkpoint, and sets the Python path. Access is verified up front, so a missing licence fails in seconds rather than after a long download.

If CMake and a C++ compiler are not found, ARDY is installed without its motion-correction extension: everything works except foot-skate cleanup, and the panel tells you so. Install both and reinstall to enable it.

### Manual installation (advanced)

If you already have Kimodo or ARDY installed in your own venv, skip the auto-installer and paste the path to your venv Python into the **Kimodo Python** / **ARDY Python** field under *Connection ▸ Advanced*.

---

## Quick Start

### Generate motion from a text prompt

1. **Start** the bridge: click **Start Kimodo** in the Connection panel.  
   The status line will show *Loading model…* then *Ready* once the model is loaded (this takes 10–60 s the first time).

2. Open the **Generate** panel. It opens in **Single Clip** mode — type a prompt (e.g. `a person jogs in a circle`), set a duration, and click **Generate Motion**.

3. A `Kimodo_Source` armature will appear in your scene with the generated motion applied.

> **Frame-rate tip:** Kimodo always generates at 30 FPS and ARDY Core at 20. If your scene is set to a different frame rate, an alert appears above the Generate button with a one-click fix.

### Use multiple segments

Switch the **Generate** panel to **Timeline** mode for this. Each segment is an independent text prompt mapped to a frame range. **Generate Motion** sends every enabled segment to Kimodo in a single model call, producing one continuous animation with smooth transitions between prompts.

- Segments are listed in order. The **Start** frame of each segment after the first is automatically locked to the **End** frame of the previous segment — just drag the End frame and the next segment's Start updates automatically.
- Use **Duplicate** to copy a segment and place it immediately after.
- Use **↑ / ↓** to reorder segments.

<img width="2676" height="1181" alt="image" src="https://github.com/user-attachments/assets/a5d336e9-f32f-44c7-9aca-a09983e869d6" />


### Follow a moving target live (ARDY only)

Instead of generating a fixed-length clip, ARDY can generate *while the timeline plays* and steer toward an object you move around the scene.

1. Switch the backend to **ARDY** and click **Start ARDY**.
2. Add an Empty somewhere in the scene.
3. Open the **Live Stream** panel, type a prompt, and set **Follow** to your Empty.
4. Click **Start Live Stream**. Playback begins and the character walks toward the Empty.
5. Drag the Empty while it plays — the motion re-plans and the character changes course.
6. Press **Stop Stream** (or `Esc`) when you have what you want.

The frames are ordinary keyframes on an ordinary armature, so once you stop, the result behaves exactly like a generated clip — including **Retarget**.

Why this only exists for ARDY: it is autoregressive, generating one short window at a time on top of what came before. Kimodo generates a whole clip in one call, so there is nothing to steer mid-flight.

**If playback keeps stalling**, the GPU is not finishing a window faster than the window plays. In order of effect:

- Lower **Steps** — this is the main latency dial.
- Raise **Buffer Frames** so more motion is generated ahead of the playhead.
- Use the `core8` checkpoint (8-frame windows) instead of the default 40-frame one.

**If the character reacts sluggishly** to the target, lower **Commit Frames** — that is how much of the near future is locked in and cannot be re-planned.

### Retarget to your own rig

After generating motion you can drive any armature from the Kimodo source:

1. Open the **Retarget** panel.
2. Set **Source** to `Kimodo_Source` and **Target** to your character rig.
3. Click **Auto-Match Bones** — the addon fuzzy-matches Kimodo bone names against your rig.
4. Review the mapping, enable/disable pairs, choose a retarget mode per bone. Its recommended to also adjust the scale of the armature to match your character and then applying it with CTRL+A.
5. Choose the type of constraint the plugin should use, "Child of", "Copy Rotation" etc...
6. Click **Apply Constraints** — Blender constraint drivers are added to your rig.
7. Click **Bake & Remove Constraints** when you are happy — keyframes are baked onto your rig and all Kimodo constraints are removed, leaving a clean, self-contained animation.

Use **Save / Load Preset** to store bone mappings for a rig and reuse them later.

<img width="1233" height="839" alt="image" src="https://github.com/user-attachments/assets/d76290db-7662-4223-9cd6-7083f89b35ca" />

### Motion constraints

Spatial goals can be given to Kimodo so the generated motion passes through specific positions:

| Constraint | What it controls |
|---|---|
| Root XZ | Where the character's root lands on the ground plane |
| Full-Body | A full joint-pose keyframe (pose a reference armature) |
| Left / Right Hand | Wrist end-effector position |
| Left / Right Foot | Foot / heel end-effector position |

To add a constraint:
1. Move the 3D cursor (or select an armature) to the desired position.
2. Set the timeline to the target frame.
3. In the **Motion Constraints** panel, click the constraint type.

**Auto-Origin** (off by default) shifts all constraint positions so the earliest root waypoint lands at Kimodo's world origin — author constraints anywhere in your scene without worrying about absolute coordinates.

---

## Panel reference

| Panel | What's in it |
|---|---|
| **Connection** | Backend selector, install / Python path, model selector, Start / Stop bridge |
| **Generate** | Single Clip mode (one prompt, duration, seed) or Timeline mode (segment list, frame ranges) — one Generate Motion button either way |
| **Live Stream** | ARDY only: follow-target, streaming controls, Start / Stop |
| **Motion Constraints** | Spatial waypoints for the generated motion |
| **Retarget** | Bone mapping, Apply Constraints, Bake |
| **Help** | Quick-start checklist, VRAM tip |

---

## Troubleshooting

**Bridge won't start / "Failed to start"**
- Check the system console for `[Kimodo Bridge]` lines — the full Python/PyTorch error is printed there.
- Check that the Python path points to the venv Python that has Kimodo installed.
- If you used the auto-installer and it failed partway through, click **Retry Install** — it will wipe the partial venv and start clean.

**CUDA out of memory**
- Use a shorter duration or fewer segments.

**Retargeted rig is in the wrong pose**
- Try a different retarget mode per bone (Copy Rotation vs Copy Transforms vs Child Of).
- Make sure the source and target armatures are both in their rest pose / have the same pose before trying the retargeting, and have scale applied on the armature.

**Frames from imorted animation dont match**
- Kimodo generates at exactly 30 FPS, ARDY Core at 20. Use the frame-rate button that appears in the Generate panel when your scene does not match.

**ARDY: "HuggingFace refused this token"**
- The text encoder needs Meta-Llama-3-8B-Instruct access. Open the model page from the Connection panel, accept the licence, wait for approval, and check the token has read permission.

**ARDY: "Foot-skate cleanup unavailable"**
- ARDY was installed without its motion-correction extension because CMake and a C++ compiler were not found. Generation works; only the foot-skate pass is missing. Install CMake and a C++ compiler (`build-essential` on Ubuntu, Visual Studio Build Tools on Windows), then reinstall ARDY.

---

## File overview

| File | Role |
|---|---|
| `__init__.py` | Blender addon entry point |
| `bridge_server.py` | Subprocess: loads Kimodo, handles generation requests |
| `ardy_bridge.py` | Subprocess: same protocol, backed by ARDY |
| `ardy_bvh.py` | BVH writer for ARDY output (ARDY ships no BVH exporter) |
| `ardy_stream.py` | Live streaming: modal operator, armature build, keyframe writing |
| `ardy_steer.py` | Pure maths for streaming: target steering, ARDY ↔ Blender space |
| `subprocess_client.py` | Blender-side bridge manager (both backends) |
| `operators.py` | All `bpy.ops.kimodo.*` operators |
| `properties.py` | All `bpy.props` scene settings |
| `panels.py` | N-panel UI |
| `constraints.py` | Converts Blender constraint markers to Kimodo JSON |
| `retarget.py` | Applies / bakes retargeting constraints |
| `ui_list.py` | UIList helper for the bone mapping panel |
| `setup_operator.py` | One-click auto-installer for Kimodo and all dependencies |
| `ardy_setup.py` | One-click auto-installer for ARDY |
| `tests/` | Blender-free checks — see [`tests/README.md`](tests/README.md) |

---

## License

See [LICENSE](LICENSE).
