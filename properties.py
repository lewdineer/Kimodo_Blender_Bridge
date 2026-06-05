"""
Kimodo Blender Bridge — Properties
All bpy.props definitions: addon preferences, scene-level settings, bone mapping.
"""

import bpy
from bpy.props import (
    StringProperty, FloatProperty, IntProperty, BoolProperty,
    EnumProperty, CollectionProperty, PointerProperty, FloatVectorProperty,
)
from bpy.types import PropertyGroup, AddonPreferences


# ---------------------------------------------------------------------------
# Motion segment (one prompt + time range bar in the timeline)
# ---------------------------------------------------------------------------

# Default colours cycling for new segments
_SEGMENT_COLORS = [
    (0.20, 0.55, 0.90, 0.85),  # blue
    (0.20, 0.75, 0.45, 0.85),  # green
    (0.90, 0.60, 0.15, 0.85),  # orange
    (0.75, 0.25, 0.75, 0.85),  # purple
    (0.90, 0.25, 0.25, 0.85),  # red
    (0.15, 0.80, 0.80, 0.85),  # cyan
    (0.90, 0.85, 0.20, 0.85),  # yellow
]


def _on_end_frame_update(self, context):
    """When a segment's end_frame changes, push the next segment's start_frame forward."""
    s = context.scene.kimodo
    segs = s.motion_segments
    for i, seg in enumerate(segs):
        if seg == self and i + 1 < len(segs):
            next_seg = segs[i + 1]
            duration = next_seg.end_frame - next_seg.start_frame
            next_seg.start_frame = self.end_frame + 1
            next_seg.end_frame = next_seg.start_frame + duration
            
            break
    
    # Update scene frame_end if any segment's end_frame is larger
    if segs:
        max_end_frame = max(seg.end_frame for seg in segs)
        if max_end_frame > context.scene.frame_end:
            context.scene.frame_end = max_end_frame


class KIMODO_MotionSegment(PropertyGroup):
    """One motion segment: a text prompt mapped to a frame range."""

    prompt: StringProperty(
        name="Prompt",
        description="Text description of the motion for this segment",
        default="a person walks forward",
    )
    start_frame: IntProperty(
        name="Start Frame",
        description="First frame of this motion segment",
        default=1,
        min=0,
    )
    end_frame: IntProperty(
        name="End Frame",
        description="Last frame of this motion segment",
        default=60,
        min=1,
        update=_on_end_frame_update,
    )
    model_type: EnumProperty(
        name="Model",
        items=[
            ("smpl",  "SOMA / SMPL",  "Standard human body skeleton"),
            #("smplx", "SMPL-X",       "Extended body with hands and face"),
        ],
        default="smpl",
    )
    seed: IntProperty(
        name="Seed",
        description="Random seed (-1 = random each time)",
        default=-1,
        min=-1,
    )
    color: FloatVectorProperty(
        name="Color",
        description="Bar colour in the timeline",
        subtype='COLOR_GAMMA',
        size=4,
        min=0.0, max=1.0,
        default=(0.20, 0.55, 0.90, 0.85),
    )
    enabled: BoolProperty(
        name="Enabled",
        description="Include this segment in generation",
        default=True,
    )
    # State tracking
    last_bvh_path: StringProperty(default="")
    generated: BoolProperty(default=False)


# ---------------------------------------------------------------------------
# Constraint item (one Kimodo motion constraint)
# ---------------------------------------------------------------------------

class KIMODO_ConstraintItem(PropertyGroup):
    """A single Kimodo constraint: a Blender object at a frame defines a spatial goal."""

    constraint_type: EnumProperty(
        name="Type",
        description="Kimodo constraint type",
        items=[
            ('root2d',      "Root Waypoint",    "2D ground-plane position (XZ). Place an Empty where you want the character's root to be.",  'EMPTY_ARROWS',   0),
            ('fullbody',    "Full-Body Pose",   "Full-body keyframe. Pose an armature exactly as you want the character at this frame.",       'ARMATURE_DATA',  1),
            ('left_hand',   "Left Hand",        "Left wrist/hand end-effector target. Place an Empty at the desired hand position.",          'VIEW_PAN',       2),
            ('right_hand',  "Right Hand",       "Right wrist/hand end-effector target.",                                                     'VIEW_PAN',       3),
            ('left_foot',   "Left Foot",        "Left foot/heel end-effector target. Place an Empty at the desired foot position.",           'SNAP_FACE',      4),
            ('right_foot',  "Right Foot",       "Right foot/heel end-effector target.",                                                      'SNAP_FACE',      5),
        ],
        default='root2d',
    )
    frame: IntProperty(
        name="Frame",
        description="Blender frame at which this constraint applies",
        default=1,
        min=0,
    )
    marker_object: PointerProperty(
        name="Object",
        description="Empty or Armature that defines the spatial constraint in the viewport",
        type=bpy.types.Object,
    )
    enabled: BoolProperty(
        name="Enabled",
        description="Include this constraint in generation",
        default=True,
    )
    # root2d extras
    include_heading: BoolProperty(
        name="Include Heading",
        description="Also constrain the facing direction at this waypoint",
        default=False,
    )
    heading_angle: FloatProperty(
        name="Heading (°)",
        description="Desired facing direction in degrees (0 = +Y forward in Blender / -Z in Kimodo)",
        default=0.0,
        subtype='ANGLE',
    )
    # display label
    label: StringProperty(
        name="Label",
        description="Optional human-readable label for this constraint",
        default="",
    )


# ---------------------------------------------------------------------------
# Generation history entry
# ---------------------------------------------------------------------------

class KIMODO_HistoryEntry(PropertyGroup):
    """One entry in the rolling generation history."""
    prompt: StringProperty(name="Prompt", default="")
    seed: IntProperty(name="Seed", default=0)
    duration: FloatProperty(name="Duration", default=5.0)
    bvh_path: StringProperty(name="BVH Path", default="")
    timestamp: StringProperty(name="Timestamp", default="")


# ---------------------------------------------------------------------------
# Bone mapping item (one row in the UIList)
# ---------------------------------------------------------------------------

class KIMODO_BoneMappingItem(PropertyGroup):
    """A single source → target bone pair for retargeting."""
    source_bone: StringProperty(
        name="Source Bone",
        description="Bone name in the Kimodo-generated armature",
        default="",
    )
    target_bone: StringProperty(
        name="Target Bone",
        description="Bone name in your target armature",
        default="",
    )
    enabled: BoolProperty(
        name="Enabled",
        description="Include this bone in retargeting",
        default=True,
    )
    retarget_mode: EnumProperty(
        name="Mode",
        description="How this bone pair is driven",
        items=[
            ("COPY_ROTATION",    "Copy Rotation",    "Copy only rotation; root bone also gets Copy Location"),
            ("COPY_TRANSFORMS",  "Copy Transforms",  "Copy location + rotation + scale together"),
            ("CHILD_OF",         "Child Of",         "Full parent-child relationship; preserves rest-pose offset"),
        ],
        default="CHILD_OF",
    )


# ---------------------------------------------------------------------------
# Scene-level settings
# ---------------------------------------------------------------------------

class KIMODO_SceneSettings(PropertyGroup):
    """Stored on bpy.context.scene.kimodo — all per-scene settings."""

    # --- Connection (subprocess bridge) ---
    python_executable: StringProperty(
        name="Python",
        description=(
            "Path to the Python executable (or venv/conda root) that has "
            "Kimodo installed. Leave blank to auto-detect from PATH."
        ),
        default="",
        subtype='FILE_PATH',
    )
    kimodo_model: EnumProperty(
        name="Model",
        description="Kimodo model to load into the bridge process",
        items=[
            ("Kimodo-SOMA-RP-v1",  "Kimodo SOMA",   "Standard human SOMA skeleton (recommended)"),
            ("Kimodo-SMPLX-RP-v1", "Kimodo SMPL-X (Unsupported atm)", "Extended body with hands and face"),
            ("Kimodo-G1-RP-v1",    "Kimodo G1 (Unsupported atm)",     "Unitree G1 robot skeleton"),
        ],
        default="Kimodo-SOMA-RP-v1",
    )
    connection_status: StringProperty(
        name="Status",
        default="Not started",
    )
    is_connected: BoolProperty(default=False)

    # --- Generation ---
    model_type: EnumProperty(
        name="Model",
        description="Which Kimodo skeleton/model to use",
        items=[
            ("smpl",  "SOMA / SMPL",  "Standard human body skeleton (SOMA). Best for most use cases."),
            #("smplx", "SMPL-X",       "Extended SMPL with hands and face. Requires Kimodo-SMPLX install."),
        ],
        default="smpl",
    )
    prompt: StringProperty(
        name="Prompt",
        description="Text description of the motion to generate",
        default="a person walks forward",
    )
    duration: FloatProperty(
        name="Duration (s)",
        description="Length of the generated motion in seconds",
        default=5.0,
        min=1.0,
        max=30.0,
        step=50,
    )
    seed: IntProperty(
        name="Seed",
        description="Random seed (-1 = random each time)",
        default=-1,
        min=-1,
    )
    output_format: EnumProperty(
        name="Format",
        description="File format Kimodo should export",
        items=[
            ("bvh", "BVH",  "Standard motion capture format. Blender imports natively."),
            ("npz", "NPZ",  "Kimodo native format (requires manual import)."),
        ],
        default="bvh",
    )
    bvh_standard_tpose: BoolProperty(
        name="Use Standard T-Pose",
        description="Export BVH with standard T-pose rest pose instead of BONES-SEED pose (SOMA models only)",
        default=True,
    )
    reuse_armature: PointerProperty(
        name="Reuse Armature",
        description=(
            "Apply generated motion to this armature instead of creating a new one. "
            "Preserves retargeting constraints already pointing at it. "
            "Leave empty to always create a new armature."
        ),
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'ARMATURE',
    )

    # Generation state (used by the modal operator)
    is_generating: BoolProperty(default=False)
    generation_progress: StringProperty(default="")
    last_bvh_path: StringProperty(
        name="Last BVH Path",
        description="Path of the most recently imported motion file",
        default="",
    )

    # --- Motion Segments (timeline bars) ---
    motion_segments: CollectionProperty(type=KIMODO_MotionSegment)
    segment_index: IntProperty(
        name="Active Segment",
        description="Currently selected motion segment",
        default=0,
    )
    # Which segment is currently being generated (for multi-generate progress)
    generating_segment_index: IntProperty(default=-1)

    # --- Retargeting ---
    source_armature: PointerProperty(
        name="Source Armature",
        description="The Kimodo-generated armature (imported from BVH)",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'ARMATURE',
    )
    target_armature: PointerProperty(
        name="Target Armature",
        description="Your character's armature to drive with the motion",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'ARMATURE',
    )
    bone_mappings: CollectionProperty(type=KIMODO_BoneMappingItem)
    bone_mapping_index: IntProperty(default=0)
    retarget_root_bone: StringProperty(
        name="Root Bone (Target)",
        description="Root / hip bone on the target armature (gets position + rotation)",
        default="",
    )
    bake_start_frame: IntProperty(name="Start Frame", default=1, min=0)
    bake_end_frame: IntProperty(name="End Frame", default=250, min=1)

    # --- Motion Constraints ---
    motion_constraints: CollectionProperty(type=KIMODO_ConstraintItem)
    constraint_index: IntProperty(default=0)
    kimodo_fps: FloatProperty(
        name="Kimodo FPS",
        description="Frames-per-second Kimodo generates at (default 30). "
                    "Used to convert Blender frame numbers to Kimodo frame indices.",
        default=30.0,
        min=1.0,
        max=120.0,
    )
    auto_canonicalize: BoolProperty(
        name="Auto-Canonicalize",
        description="Automatically offset all constraint positions so the earliest "
                    "waypoint lands at Kimodo's (0,0) origin.",
        default=False,
    )
    constraint_json_preview: StringProperty(
        name="Constraint JSON",
        description="Last-built constraints JSON (read-only preview)",
        default="",
    )

    # --- Multi-segment generation ---
    num_transition_frames: IntProperty(
        name="Transition Frames",
        description="Number of blended frames between segments in multi-prompt generation",
        default=5,
        min=1,
        max=30,
    )

    # --- Generation history ---
    generation_history: CollectionProperty(type=KIMODO_HistoryEntry)
    history_index: IntProperty(name="Active History Entry", default=0)
    history_expanded: BoolProperty(
        name="Show History",
        description="Expand / collapse the generation history list",
        default=False,
    )

    # --- Variations ---
    num_variations: IntProperty(
        name="Variations",
        description="Number of random-seed variations to generate",
        default=3,
        min=2,
        max=5,
    )

    # --- Curve path sampling ---
    path_curve: PointerProperty(
        name="Path Curve",
        description="Bezier or NURBS curve to sample as root XZ waypoints",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'CURVE',
    )
    path_waypoints: IntProperty(
        name="Waypoints",
        description="Number of evenly-spaced waypoints to sample from the curve",
        default=8,
        min=2,
        max=30,
    )
    path_start_frame: IntProperty(
        name="Start Frame",
        description="Timeline frame for the first waypoint",
        default=1,
        min=0,
    )
    path_end_frame: IntProperty(
        name="End Frame",
        description="Timeline frame for the last waypoint",
        default=90,
        min=1,
    )

    # Preset name for saving
    preset_name: StringProperty(
        name="Preset Name",
        description="Name to save / load bone mapping preset",
        default="my_rig",
    )


# ---------------------------------------------------------------------------
# Addon preferences
# ---------------------------------------------------------------------------

class KIMODO_AddonPreferences(AddonPreferences):
    bl_idname = __package__

    saved_presets: StringProperty(
        name="Saved Presets",
        description="JSON blob of all saved bone-mapping presets",
        default="{}",
    )

    install_python_path: StringProperty(
        name="System Python",
        description=(
            "Path to a Python 3.10+ executable used to create the managed venv. "
            "Leave blank to auto-detect."
        ),
        default="",
        subtype='FILE_PATH',
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_classes = [
    KIMODO_MotionSegment,
    KIMODO_ConstraintItem,
    KIMODO_BoneMappingItem,
    KIMODO_HistoryEntry,
    KIMODO_SceneSettings,
    KIMODO_AddonPreferences,
]


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.kimodo = PointerProperty(type=KIMODO_SceneSettings)


def unregister():
    del bpy.types.Scene.kimodo
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
