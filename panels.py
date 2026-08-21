"""
Kimodo Blender Bridge — Panels
All N-panel UI panels in the 3D Viewport → Kimodo tab.
"""

import os
import textwrap

import bpy
import json
from bpy.types import Panel


# ---------------------------------------------------------------------------
# Base class — common settings
# ---------------------------------------------------------------------------

class KIMODO_PanelBase:
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = 'Kimodo'


# ---------------------------------------------------------------------------
# Generation history (shared by both Generate panel modes)
# ---------------------------------------------------------------------------

def _label_wrapped(col, text, context, icon='NONE'):
    """Labels don't wrap on their own, so long prompts (e.g. 5+ joined
    segment prompts) were cut off. Emit the text as multiple labels sized
    to the panel width instead."""
    try:
        ui_scale = context.preferences.system.ui_scale
    except Exception:
        ui_scale = 1.0
    width_px = context.region.width if context.region else 320
    chars = max(24, int(width_px / (7 * ui_scale)) - 4)
    for i, line in enumerate(textwrap.wrap(text, width=chars) or [""]):
        col.label(text=line, icon=icon if i == 0 else 'BLANK1')


def _draw_history_detail(layout, context, entry):
    """Selected entry: full wrapped prompt(s) with the seed(s) used."""
    col = layout.column(align=True)
    prompts = [p.strip() for p in entry.prompt.split(" | ")] if entry.prompt else [""]
    seed_list = [x.strip() for x in entry.seeds.split(",") if x.strip()] if entry.seeds else []

    if len(prompts) > 1:
        # Multi-segment generation: one line per segment with its own seed
        for i, p in enumerate(prompts):
            seed_txt = f"   (seed {seed_list[i]})" if i < len(seed_list) else ""
            _label_wrapped(col, f"{i + 1}.  {p}{seed_txt}", context,
                           icon='TEXT' if i == 0 else 'BLANK1')
        meta = f"{entry.duration:.1f}s total  ·  {entry.timestamp.replace('T', ' ')}"
        if not seed_list:  # entry from a version that only stored the first seed
            meta = f"Seed: {entry.seed}  ·  {meta}"
    else:
        _label_wrapped(col, entry.prompt, context, icon='TEXT')
        meta = f"Seed: {entry.seed}  ·  {entry.duration:.1f}s  ·  {entry.timestamp.replace('T', ' ')}"

    meta_col = layout.column(align=True)
    _label_wrapped(meta_col, meta, context, icon='TIME')


def _draw_history(layout, context, s):
    layout.separator()
    hist_header = layout.row(align=True)
    hist_header.prop(
        s, "history_expanded",
        icon='DISCLOSURE_TRI_DOWN' if s.history_expanded else 'DISCLOSURE_TRI_RIGHT',
        icon_only=True, emboss=False,
    )
    hist_header.label(
        text=f"History ({len(s.generation_history)})", icon='TIME'
    )
    if not s.history_expanded:
        return
    if not s.generation_history:
        layout.label(text="No generations yet.", icon='INFO')
    else:
        layout.template_list(
            "KIMODO_UL_History", "",
            s, "generation_history",
            s, "history_index",
            rows=min(len(s.generation_history), 5),
        )
        if 0 <= s.history_index < len(s.generation_history):
            entry = s.generation_history[s.history_index]
            detail = layout.box()
            _draw_history_detail(detail, context, entry)
            op_row = detail.row(align=True)
            reimport_op = op_row.operator(
                "kimodo.reimport_from_history",
                text="Re-import BVH", icon='IMPORT',
            )
            reimport_op.index = s.history_index
    layout.operator("kimodo.clear_history", text="Clear History", icon='TRASH')


# ---------------------------------------------------------------------------
# Panel 1: Connection
# ---------------------------------------------------------------------------

class KIMODO_PT_Connection(KIMODO_PanelBase, Panel):
    bl_label    = "⚙  Connection"
    bl_idname   = "KIMODO_PT_Connection"
    bl_order    = 10

    def draw(self, context):
        layout = self.layout
        s = context.scene.kimodo
        running = s.is_connected

        # --- Backend selector ---
        # Which model generates the motion. Only one bridge runs at a time, so
        # switching stops the current one (see properties._on_backend_update).
        row = layout.row(align=True)
        row.prop(s, "backend", expand=True)
        layout.separator(factor=0.5)

        if s.backend == 'ARDY':
            self._draw_ardy(layout, context, s, running)
        else:
            self._draw_kimodo(layout, context, s, running)

    # ------------------------------------------------------------------ shared

    def _draw_progress(self, layout, backend: str) -> bool:
        """Live install progress. True when an install for *backend* is running.

        Both installers share setup_operator's state, so the box is drawn only
        for whichever backend actually started the install.
        """
        from . import setup_operator as so

        if not so.is_installing() or so.install_target() != backend:
            return False
        name = "ARDY" if backend == "ardy" else "Kimodo"
        box = layout.box()
        box.label(text=f"Installing {name}…", icon='TIME')
        box.label(text=so.install_status())
        dl_pct = so.download_progress()
        if dl_pct > 0.0:
            label = so.download_label()
            short = label.replace("Downloading ", "").replace(" (attempt 1/3)", "")
            box.progress(factor=dl_pct, text=f"{short}  {int(dl_pct * 100)}%")
        layout.separator(factor=0.5)
        return True

    def _draw_connection_controls(self, layout, s, running, label: str):
        """Start / Stop plus the status line — identical for both backends."""
        layout.separator(factor=0.5)
        if running:
            layout.operator("kimodo.stop_kimodo",
                            text=f"Stop {label}", icon='CANCEL')
        else:
            layout.operator("kimodo.start_kimodo",
                            text=f"Start {label}", icon='PLAY')

        status_row = layout.row()
        if running:
            status_row.label(text=s.connection_status, icon='CHECKMARK')
        elif s.connection_status in ("Not started", "Stopped"):
            status_row.label(text=s.connection_status, icon='RADIOBUT_OFF')
        else:
            is_err = s.connection_status.startswith("Failed") or \
                     s.connection_status.startswith("Error")
            status_row.label(
                text=s.connection_status,
                icon='ERROR' if is_err else 'TIME',
            )

    # ------------------------------------------------------------------ Kimodo

    def _draw_kimodo(self, layout, context, s, running):
        from . import setup_operator as so

        if self._draw_progress(layout, "kimodo"):
            pass

        elif so.install_failed() or (so.venv_exists() and not so.is_installed()):
            # install_failed()  → failed this session
            # venv_exists() but not is_installed() → partial venv from a
            # previous session (no sentinel file); treat it the same way.
            box = layout.box()
            if so.needs_python():
                box.label(text="Python 3.10–3.12 required!", icon='ERROR')
                box.separator(factor=0.3)
                box.label(text="No compatible Python was found on your system.")
                box.label(text="Click below to download the Python 3.12 installer (Windows 64-bit).")
                box.label(text='Run it, tick "Add Python to PATH".')
                box.label(text="On Windows: run the installer as Administrator.")
                box.label(text="Then restart Blender before clicking Retry Install.", icon='ERROR')
                box.label(text="On Windows, restart your computer (not just Blender)", icon='BLANK1')
                box.label(text="so the new PATH takes effect.", icon='BLANK1')
                box.separator(factor=0.3)
                box.label(text="Select your Python 3.10–3.12 executable:", icon='FILEBROWSER')
                box.label(text="e.g. /usr/bin/python3.12  or  C:\\Python312\\python.exe",
                          icon='BLANK1')
                try:
                    prefs = context.preferences.addons[__package__].preferences
                    box.prop(prefs, "system_python_override", text="")
                except Exception:
                    pass
                box.separator(factor=0.3)
                box.label(text="Or install Python from python.org:", icon='URL')
                box.operator("kimodo.open_python_download",
                             text="Download Python 3.12 Installer", icon='URL')
                box.separator(factor=0.3)
            else:
                box.label(text="Installation incomplete", icon='ERROR')
                if so.install_status():
                    box.label(text=so.install_status(), icon='BLANK1')
                try:
                    prefs = context.preferences.addons[__package__].preferences
                    box.label(text="Install location (blank = default ~/.kimodo-venv):",
                              icon='FILE_FOLDER')
                    box.prop(prefs, "install_location", text="")
                except Exception:
                    pass
            # Reuse the already-chosen location on retry — no folder prompt.
            op = box.operator("kimodo.install_kimodo",
                              text="Retry Install", icon='FILE_REFRESH')
            op.prompt_location = False
            box.operator("kimodo.reset_venv",
                         text="Reset Venv", icon='TRASH')
            layout.separator(factor=0.5)

        elif not so.is_installed() and not so.is_kimodo_venv(s.python_executable):
            box = layout.box()
            has_gpu = so.has_nvidia_gpu()
            if not has_gpu:
                box.label(text="NVIDIA GPU required!", icon='ERROR')
                box.label(text="Kimodo only works with NVIDIA GPUs (CUDA).")
                box.label(text="AMD and Intel GPUs are not supported.")
                box.separator(factor=0.3)
            else:
                box.label(text="Kimodo not installed", icon='INFO')
            box.label(text="Requires:  Python 3.10 - 3.12, ~8 GB disk, internet")
            box.label(text="Click Install, then pick a folder for the Kimodo venv.",
                      icon='FILE_FOLDER')
            row = box.row()
            row.scale_y = 1.3
            row.enabled = has_gpu
            row.operator("kimodo.install_kimodo", icon='IMPORT')
            # Advanced overrides (Python / HF token / explicit install location).
            self._draw_advanced(box, context, s, show_python=False)
            layout.separator(factor=0.5)

        elif not s.python_executable or not os.path.isfile(s.python_executable):
            box = layout.box()
            box.label(text="Kimodo venv ready", icon='CHECKMARK')
            box.operator("kimodo.use_installed_kimodo", icon='CONSOLE')
            self._draw_advanced(box, context, s, show_python=True)
            layout.separator(factor=0.5)
        else:
            # Installed and a Python executable is set — keep overrides one click
            # away under Advanced instead of always on screen.
            box = layout.box()
            self._draw_advanced(box, context, s, show_python=True)
            layout.separator(factor=0.5)

        # --- Model selector ---
        row = layout.row(align=True)
        row.label(text="Model:", icon='ARMATURE_DATA')
        row.prop(s, "kimodo_model", text="")
        row.enabled = not running

        # --- Offload toggle ---
        row = layout.row(align=True)
        row.prop(s, "use_offload", text="Enable Memory Offload")
        row.enabled = not running

        self._draw_connection_controls(layout, s, running, "Kimodo")

        # --- Delete venv (always shown when installed) ---
        if so.is_installed() and not so.is_installing():
            layout.separator(factor=0.5)
            row = layout.row()
            row.alignment = 'RIGHT'
            row.operator("kimodo.reset_venv", text="Delete Venv", icon='TRASH', emboss=False)

    # -------------------------------------------------------------------- ARDY

    def _draw_llama_access(self, box, context):
        """The one ARDY prerequisite that cannot be automated.

        ARDY's text encoder descends from meta-llama/Meta-Llama-3-8B-Instruct,
        which is gated: the user has to accept the licence on HuggingFace and
        supply a token. Spelling out both steps with direct links is the
        difference between a two-minute setup and a dead end.
        """
        box.label(text="1. Accept the Llama 3 licence (free, instant-ish):",
                  icon='URL')
        op = box.operator("kimodo.open_url",
                          text="Open Meta-Llama-3-8B-Instruct page", icon='URL')
        op.url = "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"
        box.label(text="2. Create a HuggingFace read token:", icon='LOCKED')
        op = box.operator("kimodo.open_url",
                          text="Open HuggingFace token settings", icon='URL')
        op.url = "https://huggingface.co/settings/tokens"
        box.label(text="3. Paste the token here:", icon='BLANK1')
        try:
            prefs = context.preferences.addons[__package__].preferences
            box.prop(prefs, "hf_token", text="")
        except Exception:
            pass

    def _draw_ardy(self, layout, context, s, running):
        from . import setup_operator as so
        from . import ardy_setup as aso

        has_token = False
        try:
            prefs = context.preferences.addons[__package__].preferences
            has_token = bool((prefs.hf_token or "").strip())
        except Exception:
            pass

        if self._draw_progress(layout, "ardy"):
            pass

        elif (so.install_target() == "ardy" and so.install_failed()) or \
                (aso.venv_exists() and not aso.is_installed()):
            box = layout.box()
            box.alert = True
            box.label(text="ARDY installation incomplete", icon='ERROR')
            if so.install_target() == "ardy" and so.install_status():
                _label_wrapped(box.column(align=True), so.install_status(),
                               context, icon='BLANK1')
            box.separator(factor=0.3)
            if so.needs_llama_access():
                self._draw_llama_access(box, context)
                box.separator(factor=0.3)
            op = box.operator("kimodo.install_ardy",
                              text="Retry Install", icon='FILE_REFRESH')
            op.prompt_location = False
            box.operator("kimodo.reset_ardy_venv", text="Reset ARDY Venv", icon='TRASH')
            layout.separator(factor=0.5)

        elif not aso.is_installed() and not aso.is_ardy_venv(s.ardy_python_executable):
            box = layout.box()
            has_gpu = so.has_nvidia_gpu()
            box.label(text="ARDY not installed", icon='INFO')
            box.separator(factor=0.3)

            # A plain checklist beats a paragraph: every row says whether this
            # machine already satisfies it.
            box.label(text="NVIDIA GPU (CUDA)",
                      icon='CHECKMARK' if has_gpu else 'ERROR')
            box.label(text="Python 3.10–3.12 on your system", icon='DOT')
            box.label(text="~25 GB free disk (encoder is ~16 GB)", icon='DOT')
            box.label(text="HuggingFace token with Llama 3 access",
                      icon='CHECKMARK' if has_token else 'ERROR')
            box.separator(factor=0.3)

            if not has_gpu:
                box.label(text="ARDY needs an NVIDIA GPU — AMD and Intel are",
                          icon='ERROR')
                box.label(text="not supported.", icon='BLANK1')
                box.separator(factor=0.3)

            if not has_token:
                self._draw_llama_access(box, context)
                box.separator(factor=0.3)

            box.label(text="Everything else is automatic: venv, PyTorch,", icon='INFO')
            box.label(text="ARDY, text encoder and the Core checkpoint.", icon='BLANK1')
            box.label(text="Click Install, then pick a folder for the ARDY venv.",
                      icon='FILE_FOLDER')

            row = box.row()
            row.scale_y = 1.3
            row.enabled = has_gpu and has_token
            row.operator("kimodo.install_ardy", icon='IMPORT')
            if has_gpu and not has_token:
                box.label(text="Paste a token above to enable Install.", icon='INFO')
            self._draw_ardy_advanced(box, context, s, show_python=False)
            layout.separator(factor=0.5)

        elif not s.ardy_python_executable or not os.path.isfile(s.ardy_python_executable):
            box = layout.box()
            box.label(text="ARDY venv ready", icon='CHECKMARK')
            box.operator("kimodo.use_installed_ardy", icon='CONSOLE')
            self._draw_ardy_advanced(box, context, s, show_python=True)
            layout.separator(factor=0.5)
        else:
            box = layout.box()
            self._draw_ardy_advanced(box, context, s, show_python=True)
            layout.separator(factor=0.5)

        installed = aso.is_installed() or aso.is_ardy_venv(s.ardy_python_executable)

        # --- Model + text encoder ---
        row = layout.row(align=True)
        row.label(text="Model:", icon='ARMATURE_DATA')
        row.prop(s, "ardy_model", text="")
        row.enabled = not running

        row = layout.row(align=True)
        row.label(text="Encoder:", icon='CONSOLE')
        row.prop(s, "ardy_text_encoder_device", text="")
        row.enabled = not running
        if s.ardy_text_encoder_device == 'AUTO' and not running:
            layout.label(text="GPU encoder needs ~14 GB VRAM — switch to CPU",
                         icon='INFO')
            layout.label(text="if you run out of memory.", icon='BLANK1')

        # Foot-skate cleanup needs the native extension, which only builds when
        # CMake and a C++ compiler were present at install time.
        if installed and not aso.has_postprocess(s.ardy_python_executable):
            box = layout.box()
            box.label(text="Foot-skate cleanup unavailable", icon='INFO')
            box.label(text="ARDY installed without its motion-correction", icon='BLANK1')
            box.label(text="extension (no CMake / C++ compiler found).", icon='BLANK1')
            box.label(text="Install both and reinstall to enable it.", icon='BLANK1')

        self._draw_connection_controls(layout, s, running, "ARDY")

        if installed and not so.is_installing():
            layout.separator(factor=0.5)
            row = layout.row()
            row.alignment = 'RIGHT'
            row.operator("kimodo.reset_ardy_venv", text="Delete ARDY Venv",
                         icon='TRASH', emboss=False)

    # ---------------------------------------------------------------- advanced

    def _draw_advanced(self, box, context, s, show_python=False):
        """Collapsible Advanced overrides: Python path, HF token, install location.

        Keeps the default Connection view clean while leaving both override
        capabilities (Python executable + venv install location) one click away.
        """
        expanded = s.show_advanced_connection
        box.prop(
            s, "show_advanced_connection",
            text="Advanced",
            icon='TRIA_DOWN' if expanded else 'TRIA_RIGHT',
            emboss=False,
        )
        if not expanded:
            return

        col = box.column(align=True)
        if show_python:
            col.label(text="Kimodo Python:", icon='CONSOLE')
            row = col.row(align=True)
            row.prop(s, "python_executable", text="")
            row.enabled = not s.is_connected
            col.label(text="Leave blank to auto-detect from PATH / sibling venv",
                      icon='INFO')
            col.separator(factor=0.5)

        try:
            prefs = context.preferences.addons[__package__].preferences
            col.label(text="HF Token (optional — set if model downloads stall):",
                      icon='LOCKED')
            col.prop(prefs, "hf_token", text="")
            col.label(text="System Python 3.10–3.12 (override auto-detect):",
                      icon='CONSOLE')
            col.prop(prefs, "system_python_override", text="")
            col.label(text="Install location (blank = default ~/.kimodo-venv):",
                      icon='FILE_FOLDER')
            col.prop(prefs, "install_location", text="")
        except Exception:
            pass

    def _draw_ardy_advanced(self, box, context, s, show_python=False):
        """Same idea as _draw_advanced, against the ARDY venv's own settings."""
        expanded = s.show_advanced_connection
        box.prop(
            s, "show_advanced_connection",
            text="Advanced",
            icon='TRIA_DOWN' if expanded else 'TRIA_RIGHT',
            emboss=False,
        )
        if not expanded:
            return

        col = box.column(align=True)
        if show_python:
            col.label(text="ARDY Python:", icon='CONSOLE')
            row = col.row(align=True)
            row.prop(s, "ardy_python_executable", text="")
            row.enabled = not s.is_connected
            col.label(text="Leave blank to use the managed ARDY venv", icon='INFO')
            col.separator(factor=0.5)

        try:
            prefs = context.preferences.addons[__package__].preferences
            col.label(text="HF Token (required — Llama 3 access):", icon='LOCKED')
            col.prop(prefs, "hf_token", text="")
            col.label(text="System Python 3.10–3.12 (override auto-detect):",
                      icon='CONSOLE')
            col.prop(prefs, "system_python_override", text="")
            col.label(text="ARDY install location (blank = default ~/.ardy-venv):",
                      icon='FILE_FOLDER')
            col.prop(prefs, "ardy_install_location", text="")
        except Exception:
            pass

        col.separator(factor=0.5)
        col.label(text="Generation tuning:", icon='PREFERENCES')
        col.prop(s, "ardy_cfg_text_weight")
        col.prop(s, "ardy_cfg_constraint_weight")
        col.prop(s, "ardy_history_frames")


# ---------------------------------------------------------------------------
# Panel 2: Generate — Single Clip or Timeline, one workflow
# ---------------------------------------------------------------------------
#
# Single Clip (one prompt -> one motion) and Timeline (multiple prompt
# segments stitched together) used to be two separate panels that duplicated
# the model picker, the reuse-armature row, the FPS warning, the T-pose
# option, the Generate button and the history list -- identically, in two
# places, both collapsed by default. A new user had no way to tell which
# panel to open first, or that the two were even related. One panel with a
# mode switch keeps everything that's actually shared in one place, drawn
# once, and open by default since generating motion is the point of the addon.

class KIMODO_PT_Generate(KIMODO_PanelBase, Panel):
    bl_label   = "🎬  Generate"
    bl_idname  = "KIMODO_PT_Generate"
    bl_order   = 14

    def draw(self, context):
        layout = self.layout
        s = context.scene.kimodo
        layout.enabled = not s.is_generating

        mode_row = layout.row(align=True)
        mode_row.prop(s, "generate_mode", expand=True)

        layout.separator(factor=0.5)

        if s.backend == 'ARDY':
            row = layout.row(align=True)
            row.label(text=f"Model: ARDY {s.active_skeleton or 'core'}"
                           f"  ·  {int(round(s.native_fps))} FPS")
        else:
            row = layout.row(align=True)
            row.label(text="Model:")
            row.prop(s, "model_type", expand=True)

        layout.separator(factor=0.5)

        if s.generate_mode == 'SINGLE':
            self._draw_single(layout, context, s)
        else:
            self._draw_timeline(layout, context, s)

        layout.separator()

        # --- Shared: output options ---
        # "Standard T-pose" needs standard_t_pose_global_offsets_rots.p, which
        # only the SOMA skeleton ships. ARDY's Core skeleton has no such asset,
        # so the option would silently do nothing — hide it instead.
        if s.backend != 'ARDY':
            layout.prop(s, "bvh_standard_tpose", icon='ARMATURE_DATA')
        reuse_row = layout.row(align=True)
        reuse_row.prop(s, "reuse_armature", text="Reuse", icon='ARMATURE_DATA')
        reuse_row.operator("kimodo.pick_latest_armature", text="", icon='SORTTIME')

        if s.generate_mode == 'TIMELINE':
            if s.backend == 'ARDY':
                # ARDY has no blend length: each segment continues from the
                # previous one's motion, so smoothness is set by how much
                # history the model sees (Connection ▸ Advanced).
                layout.label(text="Transitions come from generation history",
                             icon='INFO')
            else:
                trans_row = layout.row(align=True)
                trans_row.label(text="Transition Frames:")
                trans_row.prop(s, "num_transition_frames", text="")

        # --- Shared: FPS warning ---
        # Kimodo is 30 FPS; ARDY Core is 20 and G1 is 25, and the running
        # bridge reports the real value into native_fps.
        model_fps = float(s.native_fps) if s.backend == 'ARDY' else 30.0
        label = "ARDY" if s.backend == 'ARDY' else "Kimodo"
        scene_fps = context.scene.render.fps / context.scene.render.fps_base
        if abs(scene_fps - model_fps) > 0.01:
            fps_box = layout.box()
            fps_box.alert = True
            fps_box.label(
                text=f"Scene is {scene_fps:.4g} FPS — {label} needs "
                     f"{model_fps:.4g} FPS",
                icon='ERROR')
            fps_box.operator("kimodo.set_to_30fps",
                             text=f"Set to {int(round(model_fps))} FPS",
                             icon='RECOVER_LAST')

        layout.separator()

        # --- Shared: Generate / Cancel ---
        if s.is_generating:
            col = layout.column()
            col.enabled = True   # re-enable for cancel
            col.operator("kimodo.cancel_generation", text="⏹  Cancel", icon='X')
            box = layout.box()
            box.label(text=s.generation_progress or "Working…", icon='TIME')
        else:
            row = layout.column()
            row.enabled = s.is_connected
            row.scale_y = 2
            if s.generate_mode == 'SINGLE':
                connected_icon = 'PLAY' if s.is_connected else 'UNLINKED'
                row.operator("kimodo.generate", text="Generate Motion", icon=connected_icon)
                if s.generation_progress:
                    layout.label(text=s.generation_progress,
                                 icon='CHECKMARK' if "Done" in s.generation_progress else 'ERROR')
            else:
                row.operator("kimodo.generate_all_segments", text="Generate Motion", icon='PLAY')

        # --- Single-mode only: batch variations ---
        if s.generate_mode == 'SINGLE':
            layout.separator()
            var_row = layout.row(align=True)
            var_row.enabled = s.is_connected and not s.is_generating
            var_row.prop(s, "num_variations", text="Variations")
            var_row.operator(
                "kimodo.generate_variations",
                text=f"Generate {s.num_variations} Variations",
                icon='DUPLICATE',
            )

        # --- Shared: History ---
        _draw_history(layout, context, s)

        # --- Shared: Manual import fallback ---
        layout.separator()
        box = layout.box()
        box.label(text="Manual Import", icon='IMPORT')
        row = box.row()
        row.prop(s, "last_bvh_path", text="BVH Path")
        row.operator("kimodo.import_bvh", text="", icon='FILE_FOLDER').filepath = ""

    def _draw_single(self, layout, context, s):
        layout.label(text="Prompt:")
        layout.prop(s, "prompt", text="")

        layout.prop(s, "duration", slider=True)
        seed_row = layout.row(align=True)
        seed_field = seed_row.row(align=True)
        seed_field.enabled = s.seed_mode != 'RANDOM'
        seed_field.prop(s, "seed")
        seed_modes = seed_row.row(align=True)
        seed_modes.alignment = 'RIGHT'
        seed_modes.prop(s, "seed_mode", expand=True)

        row = layout.row(align=True)
        row.label(text="Output:")
        row.prop(s, "output_format", expand=True)

    def _draw_timeline(self, layout, context, s):
        row = layout.row(align=True)
        row.operator("kimodo.add_segment",    text="Add",    icon='ADD')
        row.operator("kimodo.remove_segment", text="Remove", icon='REMOVE')
        row.separator()
        row.operator("kimodo.duplicate_segment", text="", icon='DUPLICATE')
        row.operator("kimodo.move_segment_up",   text="", icon='TRIA_UP')
        row.operator("kimodo.move_segment_down", text="", icon='TRIA_DOWN')

        layout.separator(factor=0.5)

        if not s.motion_segments:
            col = layout.column()
            col.label(text="No segments yet.", icon='INFO')
            col.label(text="Click Add to create one.")
            return

        for i, seg in enumerate(s.motion_segments):
            is_generating = (i == s.generating_segment_index)

            box = layout.box()
            header = box.row(align=True)

            header.prop(seg, "enabled", text="", emboss=False,
                        icon='CHECKBOX_HLT' if seg.enabled else 'CHECKBOX_DEHLT')

            title = f"  {seg.prompt[:28]}{'…' if len(seg.prompt) > 28 else ''}"
            header.label(text=title)
            header.label(text=f"{seg.start_frame}–{seg.end_frame}")

            if is_generating:
                header.label(text="", icon='TIME')
            elif seg.generated:
                header.label(text="", icon='CHECKMARK')

            op_rem = header.operator("kimodo.remove_segment_by_index", text="", icon='X', emboss=False)
            op_rem.index = i

            col = box.column(align=True)
            col.prop(seg, "prompt", text="")

            row2 = col.row(align=True)
            start_sub = row2.row(align=True)
            start_sub.enabled = (i == 0)
            start_sub.prop(seg, "start_frame", text="Start")
            row2.prop(seg, "end_frame", text="End")

            fps = context.scene.render.fps / context.scene.render.fps_base
            dur = (seg.end_frame - seg.start_frame + 1) / fps

            row3 = col.row(align=True)
            row3.label(text=f"  {dur:.1f}s · {seg.end_frame - seg.start_frame + 1} frames",
                      icon='TIME')
            seed_field = row3.row(align=True)
            seed_field.enabled = seg.seed_mode != 'RANDOM'
            seed_field.prop(seg, "seed", text="Seed")
            seed_modes = row3.row(align=True)
            seed_modes.alignment = 'RIGHT'
            seed_modes.prop(seg, "seed_mode", expand=True)


# ---------------------------------------------------------------------------
# Panel 3: Live Stream (ARDY only)
# ---------------------------------------------------------------------------

class KIMODO_PT_LiveStream(KIMODO_PanelBase, Panel):
    bl_label   = "📡  Live Stream"
    bl_idname  = "KIMODO_PT_LiveStream"
    bl_order   = 20
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        # Kimodo is one-shot: it has no autoregressive entry point to stream
        # from, so the whole panel only makes sense on the ARDY backend.
        return context.scene.kimodo.backend == 'ARDY'

    def draw(self, context):
        layout = self.layout
        s = context.scene.kimodo

        if s.is_streaming:
            box = layout.box()
            box.label(text=s.stream_status or "Streaming…", icon='REC')
            col = layout.column()
            col.scale_y = 1.6
            col.operator("kimodo.ardy_stream_stop", text="⏹  Stop Stream", icon='X')
            layout.separator(factor=0.5)
            layout.label(text="Move the target and the motion re-plans to follow.",
                         icon='INFO')
            layout.label(text="Esc also stops the stream.", icon='BLANK1')
            return

        layout.label(text="Generate while the timeline plays.", icon='INFO')
        layout.separator(factor=0.5)

        layout.prop(s, "prompt", text="Prompt")
        target_row = layout.row(align=True)
        target_row.prop(s, "ardy_stream_target", text="Follow", icon='EMPTY_ARROWS')
        if s.ardy_stream_target is None:
            layout.label(text="No target: walks to the prompt alone.", icon='BLANK1')

        layout.separator(factor=0.5)
        col = layout.column(align=True)
        col.prop(s, "ardy_stream_diffusion_steps")
        col.prop(s, "ardy_stream_max_speed")
        col.prop(s, "ardy_stream_replan_buffer")
        col.prop(s, "ardy_stream_lead_frames")
        layout.prop(s, "ardy_stream_autoplay")

        layout.separator(factor=0.5)
        row = layout.column()
        row.enabled = s.is_connected and not s.is_generating
        row.scale_y = 2
        row.operator("kimodo.ardy_stream", text="Start Live Stream",
                     icon='PLAY' if s.is_connected else 'UNLINKED')

        if s.stream_status and not s.is_streaming:
            layout.label(text=s.stream_status, icon='INFO')

        # Streaming rewrites the near future every step, so it only looks right
        # when the scene plays at the model's own rate.
        scene_fps = context.scene.render.fps / context.scene.render.fps_base
        model_fps = float(s.native_fps)
        if abs(scene_fps - model_fps) > 0.01:
            warn = layout.box()
            warn.alert = True
            warn.label(text=f"Scene is {scene_fps:.4g} FPS — ARDY streams at "
                            f"{model_fps:.4g}", icon='ERROR')
            warn.operator("kimodo.set_to_30fps",
                          text=f"Set to {int(round(model_fps))} FPS",
                          icon='RECOVER_LAST')


# ---------------------------------------------------------------------------
# Panel 4: Motion Constraints
# ---------------------------------------------------------------------------

class KIMODO_PT_Constraints(KIMODO_PanelBase, Panel):
    bl_label   = "🎯  Motion Constraints"
    bl_idname  = "KIMODO_PT_Constraints"
    bl_order   = 25
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        s = context.scene.kimodo

        # --- Quick-add buttons ---
        layout.label(text="Add Constraint at Current Frame:", icon='ADD')
        grid = layout.grid_flow(row_major=True, columns=3, even_columns=True, align=True)

        add_types = [
            ('root2d',      "Root XZ",    'EMPTY_ARROWS'),
            ('fullbody',    "Full-Body",  'ARMATURE_DATA'),
            ('left_hand',   "L.Hand",     'VIEW_PAN'),
            ('right_hand',  "R.Hand",     'VIEW_PAN'),
            ('left_foot',   "L.Foot",     'SNAP_FACE'),
            ('right_foot',  "R.Foot",     'SNAP_FACE'),
        ]
        for ctype, label, icon in add_types:
            op = grid.operator("kimodo.add_constraint", text=label, icon=icon)
            op.constraint_type = ctype

        # Fullbody tip — shown when Full-Body button is in the grid
        has_fullbody = any(ci.constraint_type == 'fullbody' for ci in s.motion_constraints)
        if not has_fullbody:
            tip = layout.box()
            tip.label(text="Full-Body tip:", icon='INFO')
            tip.label(text="Select an armature first, then click Full-Body.")
            tip.label(text="Or generate once — source arm will be duplicated.")

        # --- Curve path waypoint sampler ---
        layout.separator()
        path_box = layout.box()
        path_box.label(text="Sample Curve as Waypoints", icon='CURVE_DATA')
        split = path_box.split(factor=0.8, align=True)
        split.prop(s, "path_curve", text="Curve")
        split.operator("kimodo.draw_freehand_curve", text="Draw", icon='GREASEPENCIL')
        if s.path_curve:
            prow = path_box.row(align=True)
            prow.prop(s, "path_waypoints", text="Points")
            prow.prop(s, "path_start_frame", text="Start F")
            prow.prop(s, "path_end_frame", text="End F")
            path_box.operator(
                "kimodo.sample_curve_as_waypoints",
                text=f"Sample {s.path_waypoints} Waypoints",
                icon='NORMALIZE_FCURVES',
            )

        layout.separator()
        n = len(s.motion_constraints)
        if n:
            layout.label(text=f"{n} constraint{'s' if n != 1 else ''}:", icon='SEQUENCE')
        else:
            layout.label(text="No constraints — motion is unconstrained", icon='INFO')

        for i, ci in enumerate(s.motion_constraints):
            box = layout.box()

            # --- Row 1: what it is ---
            row = box.row(align=True)
            row.prop(ci, "enabled", text="", emboss=False,
                     icon='CHECKBOX_HLT' if ci.enabled else 'CHECKBOX_DEHLT')

            type_icons = {
                'root2d': 'EMPTY_ARROWS', 'fullbody': 'ARMATURE_DATA',
                'left_hand': 'VIEW_PAN',  'right_hand': 'VIEW_PAN',
                'left_foot': 'SNAP_FACE', 'right_foot': 'SNAP_FACE',
            }
            row.label(text="", icon=type_icons.get(ci.constraint_type, 'DOT'))
            row.prop(ci, "constraint_type", text="")

            op_rem = row.operator("kimodo.remove_constraint", text="", icon='X')
            op_rem.index = i
            op_rem.delete_object = False

            # --- Row 2: when + jump/select. Type labels ("Full-Body Pose",
            # "Root Waypoint"...) are too long to share a row with these and
            # the type dropdown -- that's what crammed everything down to
            # single letters before. ---
            row2 = box.row(align=True)
            row2.prop(ci, "frame", text="Frame")

            op_goto = row2.operator("kimodo.goto_constraint_frame", text="", icon='TIME')
            op_goto.frame = ci.frame

            op_sel = row2.operator("kimodo.select_constraint_object", text="", icon='RESTRICT_SELECT_OFF')
            op_sel.index = i

            # --- Object picker row — type-aware ---
            sub = box.row(align=True)

            if ci.constraint_type == 'fullbody':
                # Armature picker with validation warning
                obj = ci.marker_object
                if obj and obj.type != 'ARMATURE':
                    # Wrong type — show error
                    sub.alert = True
                    sub.label(text="⚠ Not an armature!", icon='ERROR')
                    sub.prop(ci, "marker_object", text="Fix →")
                elif not obj:
                    # Nothing set — show hint
                    sub.alert = True
                    sub.label(text="Select an armature above then click Full-Body again", icon='INFO')
                    sub.prop(ci, "marker_object", text="Or set →")
                else:
                    # Valid armature — show normally with bone count hint
                    bone_n = len(obj.data.bones)
                    sub.prop(ci, "marker_object", text=f"Pose Ref  ({bone_n} bones)")
                    sub.operator(
                        "kimodo.select_constraint_object",
                        text="", icon='EDITMODE_HLT',
                    ).index = i
            else:
                # Regular Empty picker for all other types
                sub.prop(ci, "marker_object", text="Marker")

            # root2d heading extras
            if ci.constraint_type == 'root2d':
                sub2 = box.row(align=True)
                sub2.prop(ci, "include_heading", text="Heading")
                if ci.include_heading:
                    sub2.prop(ci, "heading_angle", text="")

        layout.separator()

        # --- Settings ---
        box = layout.box()
        box.label(text="Settings", icon='PREFERENCES')
        row = box.row(align=True)
        row.prop(s, "kimodo_fps")
        row.prop(s, "auto_canonicalize", toggle=True, text="Auto-Origin")

        layout.separator()

        # --- Actions ---
        row = layout.row(align=True)
        row.operator("kimodo.preview_constraints_json", icon='TEXT', text="Preview JSON")
        row.operator("kimodo.clear_constraints",        icon='TRASH', text="Clear All")


# ---------------------------------------------------------------------------
# Panel 4: Retarget
# ---------------------------------------------------------------------------

class KIMODO_PT_Retarget(KIMODO_PanelBase, Panel):
    bl_label   = "🦴  Retarget"
    bl_idname  = "KIMODO_PT_Retarget"
    bl_order   = 30
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        s = context.scene.kimodo

        # Armature pickers
        box = layout.box()
        box.label(text="Armatures", icon='ARMATURE_DATA')
        box.prop(s, "source_armature", text="Source (Kimodo)")
        box.prop(s, "target_armature", text="Target (Your Rig)")
        box.prop(s, "retarget_root_bone", text="Root Bone")

        layout.separator()

        # Bone mapping list
        layout.label(text="Bone Mapping:", icon='BONE_DATA')
        layout.label(text="Link toggle = target bone's Inherit Rotation", icon='LINKED')

        if s.source_armature and s.target_armature:
            # Auto-match button
            layout.operator("kimodo.auto_map_bones",
                            text="Auto-Match Bones", icon='SHADERFX')

        row = layout.row()
        row.template_list(
            "KIMODO_UL_BoneMappings", "",
            s, "bone_mappings",
            s, "bone_mapping_index",
            rows=6,
        )

        col = row.column(align=True)
        col.operator("kimodo.add_bone_mapping",    text="", icon='ADD')
        col.operator("kimodo.remove_bone_mapping", text="", icon='REMOVE')

        layout.separator()

        # Apply / Remove constraints
        row = layout.row(align=True)
        row.operator("kimodo.apply_retargeting",  text="Apply Constraints", icon='CONSTRAINT_BONE')
        row.operator("kimodo.remove_retargeting", text="",                  icon='X')

        layout.separator()

        # Bake section
        box = layout.box()
        box.label(text="Bake Animation", icon='RENDER_ANIMATION')
        row = box.row(align=True)
        row.prop(s, "bake_start_frame", text="Start")
        row.prop(s, "bake_end_frame",   text="End")
        box.operator("kimodo.bake_retargeting",
                     text="Bake & Remove Constraints", icon='NLA_PUSHDOWN')

        layout.separator()

        # Presets
        box = layout.box()
        box.label(text="Bone Map Presets", icon='PRESET')
        row = box.row(align=True)
        row.prop(s, "preset_name", text="")
        row.operator("kimodo.save_preset", text="", icon='FILE_TICK')
        row.operator("kimodo.load_preset", text="", icon='IMPORT').preset_name = s.preset_name

        # List saved presets
        try:
            prefs = context.preferences.addons[__package__].preferences
            from . import retarget as rt
            preset_names = rt.list_presets(prefs)
        except Exception:
            preset_names = []

        if preset_names:
            col = box.column(align=True)
            for name in preset_names:
                row2 = col.row(align=True)
                op_load = row2.operator("kimodo.load_preset",   text=name, icon='IMPORT')
                op_load.preset_name = name
                op_del  = row2.operator("kimodo.delete_preset", text="",   icon='TRASH')
                op_del.preset_name = name

        # File export / import
        row = box.row(align=True)
        row.operator("kimodo.export_preset_file", text="Export to File", icon='EXPORT')
        row.operator("kimodo.import_preset_file", text="Import from File", icon='IMPORT')


# ---------------------------------------------------------------------------
# Panel 4: Help / About
# ---------------------------------------------------------------------------

class KIMODO_PT_Help(KIMODO_PanelBase, Panel):
    bl_label   = "ℹ  Help"
    bl_idname  = "KIMODO_PT_Help"
    bl_order   = 90
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="Quick Start:", icon='QUESTION')
        col.separator()
        col.label(text="1. Click 'Install Kimodo (Auto)' in the Connection panel")
        col.label(text="   (or point Kimodo Python at your own venv)")
        col.label(text="2. Start Kimodo")
        col.label(text="   (model loads once, stays loaded)")
        col.separator()
        col.label(text="3. Enter prompt → Generate Motion")
        col.label(text="4. In Retarget tab, Pick source + target rigs")
        col.label(text="5. Auto-Match → Apply Constraints")
        col.label(text="6. If it does not match, manually add your control bones")
        col.label(text="7. Click Apply Constraints")
        col.label(text="8. Bake when satisfied")
        col.separator()
        col.label(text="Docs & Source:", icon='URL')
        col.label(text="github.com/nv-tlabs/kimodo")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_classes = [
    KIMODO_PT_Connection,
    KIMODO_PT_Generate,
    KIMODO_PT_LiveStream,
    KIMODO_PT_Constraints,
    KIMODO_PT_Retarget,
    KIMODO_PT_Help,
]


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
