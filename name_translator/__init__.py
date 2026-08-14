# SPDX-License-Identifier: GPL-3.0-or-later
"""
Name Translator
===============

Translate Blender datablock names (objects, meshes, materials, bones, vertex
groups, shape keys, ...) from one language into another - built for cleaning up
Chinese / Japanese character models so they use readable English names that
follow Blender's own conventions.

Key ideas
---------
* Nothing is renamed until you press Apply. "Scan" builds a preview list where
  every single row can be edited or switched off.
* Left / right markers are recognised in the source language and converted to
  Blender's ".L" / ".R" suffix convention, then cross-checked against the actual
  geometry so a bone called "Hand.L" really is on the character's left.
* A latin marker is picked up wherever it stands on its own: at the end
  ("Hand_R"), at the start ("R_Hand") and in the middle ("Skn_R_WingC_ZF_01"),
  in which case it is moved to the end where Blender can act on it.
* Works fully offline through a built-in dictionary; online services are opt-in.
"""

# Legacy add-on metadata. Ignored when installed as an extension (which uses
# blender_manifest.toml instead) but kept so "Install legacy Add-on" works too.
bl_info = {
    "name": "Name Translator (CN/JP -> EN)",
    "author": "Generated for Blender 4.2 - 5.x",
    "version": (1, 0, 1),
    "blender": (4, 2, 0),
    "location": "3D Viewport > Sidebar (N) > Translate",
    "description": "Translate mesh / material / bone names between languages, with left-right consistency checking",
    "doc_url": "",
    "category": "Object",
}

import json
import os
import re

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import AddonPreferences, Operator, Panel, PropertyGroup, UIList
from mathutils import Vector

ADDON_ID = __package__ or __name__

# ---------------------------------------------------------------------------
# 1. Dictionaries
# ---------------------------------------------------------------------------
# Values are written in PascalCase. They are split back into words at render
# time, so the same entry can produce "UpperArm", "upper_arm" or "Upper Arm"
# depending on the naming style the user picked.

# Left / right markers. These are pulled out of the name and become a .L / .R
# suffix rather than a word.
SIDE_TABLE = {
    "\u5de6": "L",              # zuo   - left
    "\u53f3": "R",              # you   - right
    "\u5de6\u4fa7": "L",        # zuo ce
    "\u53f3\u4fa7": "R",
    "\u5de6\u8fb9": "L",
    "\u53f3\u8fb9": "R",
    "\u5de6\u9762": "L",
    "\u53f3\u9762": "R",
    "\u5de6\u90e8": "L",
    "\u53f3\u90e8": "R",
    "\u5de6\u53f3": "",         # "left-right" = both, no side
}

# Traditional / Japanese kanji -> simplified, applied only when looking a
# segment up. This makes the same dictionary work for traditional Chinese and
# for most Japanese kanji found in MMD models.
VARIANT_TO_SIMPLIFIED = {
    "\u982d": "\u5934", "\u9AEE": "\u53d1", "\u767c": "\u53d1", "\u8166": "\u8111",
    "\u81c9": "\u8138", "\u984f": "\u989c", "\u9846": "\u9897", "\u9F52": "\u9f7f",
    "\u8f49": "\u8f6c", "\u9AD4": "\u4f53", "\u5B9F": "\u5b9e", "\u7d75": "\u7ed8",
    "\u8173": "\u811a", "\u81C2": "\u81c2", "\u9AD8": "\u9ad8", "\u5074": "\u4fa7",
    "\u908A": "\u8fb9", "\u9577": "\u957f", "\u9ede": "\u70b9", "\u7dda": "\u7ebf",
    "\u7d30": "\u7ec6", "\u7d44": "\u7ec4", "\u7d10": "\u7ebd", "\u7d19": "\u7eb8",
    "\u7d05": "\u7ea2", "\u7DBF": "\u7ef5", "\u7DA0": "\u7eff", "\u7DE3": "\u7f18",
    "\u9EC3": "\u9ec4", "\u84DD": "\u84dd", "\u85CD": "\u84dd", "\u7070": "\u7070",
    "\u93A1": "\u955c", "\u93C8": "\u94fe", "\u9375": "\u952e", "\u9280": "\u94f6",
    "\u9435": "\u94c1", "\u92FC": "\u94a2", "\u9285": "\u94dc", "\u9322": "\u94b1",
    "\u978B": "\u978b", "\u8896": "\u8896", "\u88DD": "\u88c5", "\u98FE": "\u9970",
    "\u9762": "\u9762", "\u9AA8": "\u9aa8", "\u9EDE": "\u70b9", "\u820C": "\u820c",
    "\u773C": "\u773c", "\u776B": "\u776b", "\u7709": "\u7709", "\u9F3B": "\u9f3b",
    "\u8033": "\u8033", "\u624B": "\u624b", "\u8DB3": "\u8db3", "\u817F": "\u817f",
    "\u819D": "\u819d", "\u8098": "\u8098", "\u8155": "\u8155", "\u80A9": "\u80a9",
    "\u80F8": "\u80f8", "\u80CC": "\u80cc", "\u8170": "\u8170", "\u9838": "\u9888",
    "\u9838\u90E8": "\u9888\u90e8", "\u808C": "\u808c", "\u819A": "\u80a4",
    "\u670D": "\u670d", "\u88D9": "\u88d9", "\u8932": "\u88e4", "\u5E3D": "\u5e3d",
    "\u624B\u888B": "\u624b\u5957", "\u624B\u5957": "\u624b\u5957",
    "\u773C\u93E1": "\u773c\u955c", "\u9271": "\u77ff", "\u5E7E": "\u51e0",
    "\u756A": "\u756a", "\u898F": "\u89c4", "\u9928": "\u9986", "\u5716": "\u56fe",
    "\u5F71": "\u5f71", "\u8B77": "\u62a4", "\u885B": "\u536b", "\u7378": "\u517d",
    "\u9CF3": "\u51e4", "\u9F8D": "\u9f99", "\u7FFC": "\u7ffc", "\u7FBD": "\u7fbd",
    "\u5C3E": "\u5c3e", "\u89D2": "\u89d2", "\u722A": "\u722a", "\u9C7C": "\u9c7c",
    "\u9BAD": "\u9ccd", "\u9CD7": "\u9cde", "\u9EB5": "\u9762", "\u5E36": "\u5e26",
    "\u7E6B": "\u7cfb", "\u7D50": "\u7ed3", "\u7DAD": "\u7ef4", "\u7E4A": "\u7ea4",
    "\u5716\u5C64": "\u56fe\u5c42", "\u5C64": "\u5c42", "\u908A\u7DE3": "\u8fb9\u7f18",
    "\u9078": "\u9009", "\u984C": "\u9898", "\u985E": "\u7c7b", "\u9A57": "\u9a8c",
    "\u8207": "\u4e0e", "\u5011": "\u4eec", "\u5169": "\u4e24", "\u500B": "\u4e2a",
    "\u5F8C": "\u540e", "\u524D": "\u524d", "\u5167": "\u5185", "\u5916": "\u5916",
    "\u9593": "\u95f4", "\u958B": "\u5f00", "\u95DC": "\u5173", "\u7BC0": "\u8282",
    "\u6A21": "\u6a21", "\u578B": "\u578b", "\u8CEA": "\u8d28", "\u6750": "\u6750",
    "\u984F\u8272": "\u989c\u8272", "\u8272": "\u8272", "\u5149": "\u5149",
    "\u7A7A": "\u7a7a", "\u9ED1": "\u9ed1", "\u767D": "\u767d",
}

# The main dictionary. Roughly grouped so it is easy to extend by hand.
BASE_DICTIONARY = {
    # --- skeleton / core ------------------------------------------------
    "\u5168\u8eab": "Root",              # quan shen - whole body
    "\u4e2d\u5fc3": "Center",
    "\u6839": "Root",
    "\u6839\u9aa8": "RootBone",
    "\u8eab\u4f53": "Body",
    "\u8eaf\u5e72": "Torso",
    "\u4e0a\u534a\u8eab": "UpperBody",
    "\u4e0b\u534a\u8eab": "LowerBody",
    "\u8170": "Waist",
    "\u8170\u90e8": "Waist",
    "\u80ef": "Hip",
    "\u9aa8\u76c6": "Pelvis",
    "\u81c0": "Hip",
    "\u81c0\u90e8": "Hip",
    "\u5c41\u80a1": "Hip",
    "\u810a\u67f1": "Spine",
    "\u810a\u690e": "Spine",
    "\u80f8": "Chest",
    "\u80f8\u90e8": "Chest",
    "\u4e73": "Breast",
    "\u4e73\u623f": "Breast",
    "\u8179": "Belly",
    "\u8179\u90e8": "Abdomen",
    "\u809a\u5b50": "Belly",
    "\u80cc": "Back",
    "\u80cc\u90e8": "Back",
    "\u80a9": "Shoulder",
    "\u80a9\u8180": "Shoulder",
    "\u9501\u9aa8": "Clavicle",
    "\u8116\u5b50": "Neck",
    "\u9888": "Neck",
    "\u9888\u90e8": "Neck",
    "\u9996": "Head",
    "\u5934": "Head",
    "\u5934\u90e8": "Head",
    "\u9aa8": "Bone",
    "\u9aa8\u9abc": "Bone",
    "\u9aa8\u67b6": "Armature",
    "\u5173\u8282": "Joint",
    "\u808c\u8089": "Muscle",

    # --- face ------------------------------------------------------------
    "\u8138": "Face",
    "\u9762\u90e8": "Face",
    "\u8138\u988a": "Cheek",
    "\u988a": "Cheek",
    "\u4e0b\u5df4": "Chin",
    "\u4e0b\u989a": "Jaw",
    "\u989d\u5934": "Forehead",
    "\u592a\u9633\u7a74": "Temple",
    "\u8033": "Ear",
    "\u8033\u6735": "Ear",
    "\u517d\u8033": "AnimalEar",
    "\u732b\u8033": "CatEar",
    "\u72d0\u8033": "FoxEar",
    "\u5154\u8033": "RabbitEar",
    "\u773c": "Eye",
    "\u773c\u775b": "Eye",
    "\u76ee": "Eye",
    "\u4e24\u76ee": "Eyes",
    "\u773c\u7403": "Eyeball",
    "\u773c\u767d": "Sclera",
    "\u767d\u76ee": "Sclera",
    "\u9ed1\u76ee": "Pupil",
    "\u77b3": "Pupil",
    "\u77b3\u5b54": "Pupil",
    "\u8679\u819c": "Iris",
    "\u773c\u76ae": "Eyelid",
    "\u773c\u775d": "Eyelid",
    "\u776b\u6bdb": "Eyelash",
    "\u7709": "Eyebrow",
    "\u7709\u6bdb": "Eyebrow",
    "\u9f3b": "Nose",
    "\u9f3b\u5b50": "Nose",
    "\u5634": "Mouth",
    "\u5634\u5df4": "Mouth",
    "\u53e3": "Mouth",
    "\u5507": "Lip",
    "\u5634\u5507": "Lip",
    "\u7259": "Teeth",
    "\u7259\u9f7f": "Teeth",
    "\u9f7f": "Teeth",
    "\u820c": "Tongue",
    "\u820c\u5934": "Tongue",
    "\u5589\u5499": "Throat",
    "\u80e1\u5b50": "Beard",
    "\u80e1\u987b": "Whisker",
    "\u76ae\u80a4": "Skin",
    "\u808c\u80a4": "Skin",
    "\u8868\u60c5": "Expression",
    "\u53e3\u578b": "LipSync",

    # --- arms and hands ---------------------------------------------------
    "\u624b\u81c2": "Arm",
    "\u81c2": "Arm",
    "\u80f3\u818a": "Arm",
    "\u4e0a\u81c2": "UpperArm",
    "\u5927\u81c2": "UpperArm",
    "\u524d\u81c2": "Forearm",
    "\u5c0f\u81c2": "Forearm",
    "\u4e0b\u81c2": "Forearm",
    "\u624b\u8098": "Elbow",
    "\u8098": "Elbow",
    "\u624b\u8155": "Wrist",
    "\u8155": "Arm",
    "\u624b": "Hand",
    "\u624b\u638c": "Palm",
    "\u624b\u80cc": "HandBack",
    "\u62f3": "Fist",
    "\u62c7\u6307": "Thumb",
    "\u5927\u62c7\u6307": "Thumb",
    "\u4eb2\u6307": "Thumb",
    "\u98df\u6307": "IndexFinger",
    "\u4eba\u6307": "IndexFinger",
    "\u4eba\u5dee\u6307": "IndexFinger",
    "\u4e2d\u6307": "MiddleFinger",
    "\u65e0\u540d\u6307": "RingFinger",
    "\u836f\u6307": "RingFinger",
    "\u5c0f\u6307": "LittleFinger",
    "\u624b\u6307": "Finger",
    "\u6307": "Finger",
    "\u6307\u5c16": "FingerTip",
    "\u6307\u8282": "Knuckle",
    "\u6307\u7532": "Nail",

    # --- legs and feet ----------------------------------------------------
    "\u817f": "Leg",
    "\u5927\u817f": "Thigh",
    "\u5c0f\u817f": "Calf",
    "\u819d": "Knee",
    "\u819d\u76d6": "Knee",
    "\u811a": "Foot",
    "\u8db3": "Leg",
    "\u811a\u8e1d": "Ankle",
    "\u8db3\u9996": "Ankle",
    "\u8e1d": "Ankle",
    "\u811a\u5c16": "Toe",
    "\u811a\u8dbe": "Toe",
    "\u8dbe": "Toe",
    "\u5927\u811a\u8dbe": "BigToe",
    "\u811a\u8ddf": "Heel",
    "\u540e\u8ddf": "Heel",
    "\u811a\u5e95": "Sole",
    "\u811a\u80cc": "Instep",

    # --- extras -----------------------------------------------------------
    "\u5c3e": "Tail",
    "\u5c3e\u5df4": "Tail",
    "\u7fc5\u8180": "Wing",
    "\u7ffc": "Wing",
    "\u7fbd\u7ffc": "Wing",
    "\u7fbd\u6bdb": "Feather",
    "\u89d2": "Horn",
    "\u72c4\u89d2": "Horn",
    "\u722a": "Claw",
    "\u722a\u5b50": "Claw",
    "\u9ccd": "Fin",
    "\u9cde": "Scale",
    "\u5149\u73af": "Halo",
    "\u5929\u4f7f\u73af": "Halo",

    # --- hair -------------------------------------------------------------
    "\u5934\u53d1": "Hair",
    "\u53d1": "Hair",
    "\u6bdb\u53d1": "Hair",
    "\u524d\u53d1": "FrontHair",
    "\u5218\u6d77": "Bangs",
    "\u540e\u53d1": "BackHair",
    "\u4fa7\u53d1": "SideHair",
    "\u6a2a\u53d1": "SideHair",
    "\u9a6c\u5c3e": "Ponytail",
    "\u53cc\u9a6c\u5c3e": "Twintails",
    "\u8fab\u5b50": "Braid",
    "\u9ebb\u82b1\u8fab": "Braid",
    "\u4e38\u5b50\u5934": "HairBun",
    "\u53d1\u4e1d": "HairStrand",
    "\u53d1\u675f": "HairStrand",
    "\u5446\u6bdb": "Ahoge",
    "\u53d1\u9970": "HairAccessory",
    "\u53d1\u5361": "Hairpin",
    "\u53d1\u5939": "Hairclip",
    "\u53d1\u5e26": "Headband",
    "\u53d1\u5708": "HairTie",

    # --- clothing ---------------------------------------------------------
    "\u8863\u670d": "Clothes",
    "\u8863": "Clothes",
    "\u670d": "Clothes",
    "\u670d\u88c5": "Costume",
    "\u4e0a\u8863": "Top",
    "\u5916\u5957": "Jacket",
    "\u5939\u514b": "Jacket",
    "\u5927\u8863": "Coat",
    "\u886c\u886b": "Shirt",
    "\u886c\u8863": "Shirt",
    "\u6064": "Shirt",
    "\u6bdb\u8863": "Sweater",
    "\u80cc\u5fc3": "Vest",
    "\u9a6c\u7532": "Vest",
    "\u5236\u670d": "Uniform",
    "\u6821\u670d": "SchoolUniform",
    "\u548c\u670d": "Kimono",
    "\u6d74\u8863": "Yukata",
    "\u65d7\u888d": "Cheongsam",
    "\u8fde\u8863\u88d9": "Dress",
    "\u793c\u670d": "Gown",
    "\u88d9\u5b50": "Skirt",
    "\u88d9": "Skirt",
    "\u77ed\u88d9": "MiniSkirt",
    "\u957f\u88d9": "LongSkirt",
    "\u767e\u8936\u88d9": "PleatedSkirt",
    "\u88e4\u5b50": "Pants",
    "\u88e4": "Pants",
    "\u77ed\u88e4": "Shorts",
    "\u957f\u88e4": "Trousers",
    "\u725b\u4ed4\u88e4": "Jeans",
    "\u5185\u88e4": "Panties",
    "\u5185\u8863": "Underwear",
    "\u80f8\u7f69": "Bra",
    "\u6587\u80f8": "Bra",
    "\u6cf3\u88c5": "Swimsuit",
    "\u6cf3\u8863": "Swimsuit",
    "\u6bd4\u57fa\u5c3c": "Bikini",
    "\u889c\u5b50": "Socks",
    "\u889c": "Socks",
    "\u957f\u889c": "Stockings",
    "\u4e1d\u889c": "Stockings",
    "\u8fc7\u819d\u889c": "Thighhighs",
    "\u8fde\u88e4\u889c": "Pantyhose",
    "\u978b": "Shoes",
    "\u978b\u5b50": "Shoes",
    "\u9774": "Boots",
    "\u9774\u5b50": "Boots",
    "\u9ad8\u8ddf\u978b": "HighHeels",
    "\u51c9\u978b": "Sandals",
    "\u62d6\u978b": "Slippers",
    "\u5e3d": "Hat",
    "\u5e3d\u5b50": "Hat",
    "\u5934\u5dfe": "Bandana",
    "\u56f4\u5dfe": "Scarf",
    "\u56f4\u8116": "Scarf",
    "\u9886\u5e26": "Necktie",
    "\u9886\u7ed3": "Bowtie",
    "\u9886\u5b50": "Collar",
    "\u8863\u9886": "Collar",
    "\u9879\u5708": "Collar",
    "\u8896\u5b50": "Sleeve",
    "\u8896": "Sleeve",
    "\u8170\u5e26": "Belt",
    "\u76ae\u5e26": "Belt",
    "\u8170\u5c01": "Waistband",
    "\u624b\u5957": "Gloves",
    "\u62a4\u8155": "Wristband",
    "\u62ab\u98ce": "Cape",
    "\u6597\u7bf7": "Cloak",
    "\u515c\u5e3d": "Hood",
    "\u53e3\u888b": "Pocket",
    "\u7ebd\u6263": "Button",
    "\u6263\u5b50": "Button",
    "\u62c9\u94fe": "Zipper",
    "\u8774\u8776\u7ed3": "Bow",
    "\u4e1d\u5e26": "Ribbon",
    "\u7f0e\u5e26": "Ribbon",
    "\u7ef3": "Rope",
    "\u7ef3\u5b50": "Rope",
    "\u5e26": "Strap",
    "\u94fe": "Chain",
    "\u9501\u94fe": "Chain",
    "\u82b1\u8fb9": "Lace",
    "\u857e\u4e1d": "Lace",
    "\u8936\u76b1": "Fold",
    "\u8865\u4e01": "Patch",
    "\u56f4\u88d9": "Apron",
    "\u7761\u8863": "Pajamas",
    "\u6218\u6597\u670d": "BattleSuit",
    "\u76d4\u7532": "Armor",
    "\u62a4\u7532": "Armor",
    "\u5934\u76d4": "Helmet",
    "\u9762\u7f69": "Visor",
    "\u80a9\u7532": "Pauldron",
    "\u62a4\u819d": "KneePad",
    "\u62a4\u8098": "ElbowPad",

    # --- accessories ------------------------------------------------------
    "\u9970\u54c1": "Accessory",
    "\u914d\u4ef6": "Accessory",
    "\u88c5\u9970": "Decoration",
    "\u9996\u9970": "Jewelry",
    "\u9879\u94fe": "Necklace",
    "\u540a\u5760": "Pendant",
    "\u8033\u73af": "Earring",
    "\u8033\u9970": "Earring",
    "\u6212\u6307": "Ring",
    "\u624b\u956f": "Bracelet",
    "\u624b\u94fe": "Bracelet",
    "\u811a\u94fe": "Anklet",
    "\u773c\u955c": "Glasses",
    "\u58a8\u955c": "Sunglasses",
    "\u9762\u5177": "Mask",
    "\u53e3\u7f69": "FaceMask",
    "\u7687\u51a0": "Crown",
    "\u738b\u51a0": "Crown",
    "\u5934\u9970": "Headdress",
    "\u80cc\u5305": "Backpack",
    "\u5305": "Bag",
    "\u4e66\u5305": "Schoolbag",
    "\u94b1\u5305": "Wallet",
    "\u96e8\u4f1e": "Umbrella",
    "\u4f1e": "Umbrella",
    "\u6247\u5b50": "Fan",
    "\u82b1": "Flower",
    "\u82b1\u6735": "Flower",
    "\u53f6\u5b50": "Leaf",
    "\u661f\u661f": "Star",
    "\u5fc3": "Heart",
    "\u7231\u5fc3": "Heart",
    "\u94c3\u94db": "Bell",

    # --- props / weapons --------------------------------------------------
    "\u6b66\u5668": "Weapon",
    "\u5251": "Sword",
    "\u5200": "Blade",
    "\u67aa": "Gun",
    "\u5f13": "Bow",
    "\u7bad": "Arrow",
    "\u76fe": "Shield",
    "\u76fe\u724c": "Shield",
    "\u6cd5\u6756": "Staff",
    "\u9b54\u6756": "Wand",
    "\u9524": "Hammer",
    "\u65a7": "Axe",
    "\u9053\u5177": "Prop",
    "\u624b\u673a": "Phone",
    "\u4e66": "Book",
    "\u676f\u5b50": "Cup",

    # --- material / shading ----------------------------------------------
    "\u6750\u8d28": "Material",
    "\u6750\u6599": "Material",
    "\u8d34\u56fe": "Texture",
    "\u7eb9\u7406": "Texture",
    "\u56fe\u7247": "Image",
    "\u56fe\u50cf": "Image",
    "\u6cd5\u7ebf": "Normal",
    "\u9ad8\u5149": "Specular",
    "\u53cd\u5c04": "Reflection",
    "\u6298\u5c04": "Refraction",
    "\u9634\u5f71": "Shadow",
    "\u6295\u5f71": "Shadow",
    "\u900f\u660e": "Transparent",
    "\u534a\u900f\u660e": "Translucent",
    "\u4e0d\u900f\u660e": "Opaque",
    "\u63cf\u8fb9": "Outline",
    "\u8f6e\u5ed3": "Outline",
    "\u8fb9\u7f18\u5149": "RimLight",
    "\u53d1\u5149": "Emission",
    "\u81ea\u53d1\u5149": "Emission",
    "\u5149\u6cfd": "Gloss",
    "\u7c97\u7cd9": "Roughness",
    "\u91d1\u5c5e": "Metal",
    "\u91d1\u5c5e\u5ea6": "Metallic",
    "\u73bb\u7483": "Glass",
    "\u5851\u6599": "Plastic",
    "\u6a61\u80f6": "Rubber",
    "\u5e03": "Cloth",
    "\u5e03\u6599": "Fabric",
    "\u76ae\u9769": "Leather",
    "\u76ae": "Leather",
    "\u6bdb\u76ae": "Fur",
    "\u7ed2\u6bdb": "Fluff",
    "\u6728": "Wood",
    "\u6728\u5934": "Wood",
    "\u77f3": "Stone",
    "\u77f3\u5934": "Stone",
    "\u91d1": "Gold",
    "\u94f6": "Silver",
    "\u94dc": "Copper",
    "\u94c1": "Iron",
    "\u94a2": "Steel",
    "\u6c34": "Water",
    "\u51b0": "Ice",
    "\u706b": "Fire",
    "\u70df": "Smoke",
    "\u4e91": "Cloud",
    "\u5149": "Light",
    "\u5f71": "Shadow",
    "\u989c\u8272": "Color",
    "\u8272": "Color",
    "\u6e10\u53d8": "Gradient",
    "\u56fe\u5c42": "Layer",
    "\u906e\u7f69": "Mask",
    "\u8499\u7248": "Mask",
    "\u6df7\u5408": "Mix",
    "\u8282\u70b9": "Node",
    "\u7740\u8272\u5668": "Shader",
    "\u5361\u901a": "Toon",
    "\u5199\u5b9e": "Realistic",

    # --- colours ----------------------------------------------------------
    "\u767d": "White",
    "\u767d\u8272": "White",
    "\u9ed1": "Black",
    "\u9ed1\u8272": "Black",
    "\u7ea2": "Red",
    "\u7ea2\u8272": "Red",
    "\u84dd": "Blue",
    "\u84dd\u8272": "Blue",
    "\u7eff": "Green",
    "\u7eff\u8272": "Green",
    "\u9ec4": "Yellow",
    "\u9ec4\u8272": "Yellow",
    "\u7d2b": "Purple",
    "\u7d2b\u8272": "Purple",
    "\u7c89": "Pink",
    "\u7c89\u8272": "Pink",
    "\u6a59": "Orange",
    "\u6a59\u8272": "Orange",
    "\u7070": "Gray",
    "\u7070\u8272": "Gray",
    "\u68d5": "Brown",
    "\u8910\u8272": "Brown",
    "\u91d1\u8272": "Gold",
    "\u94f6\u8272": "Silver",

    # --- generic modifiers ------------------------------------------------
    "\u4e0a": "Upper",
    "\u4e0b": "Lower",
    "\u524d": "Front",
    "\u540e": "Back",
    "\u5185": "Inner",
    "\u5916": "Outer",
    "\u5185\u4fa7": "Inner",
    "\u5916\u4fa7": "Outer",
    "\u6b63\u9762": "Front",
    "\u80cc\u9762": "Back",
    "\u4e2d": "Middle",
    "\u4e2d\u95f4": "Middle",
    "\u4fa7": "Side",
    "\u5e95": "Bottom",
    "\u5e95\u90e8": "Bottom",
    "\u9876": "Top",
    "\u9876\u90e8": "Top",
    "\u8fb9": "Edge",
    "\u8fb9\u7f18": "Edge",
    "\u89d2\u843d": "Corner",
    "\u90e8\u5206": "Part",
    "\u90e8\u4ef6": "Part",
    "\u96f6\u4ef6": "Part",
    "\u6574\u4f53": "Whole",
    "\u526f": "Sub",
    "\u4e3b": "Main",
    "\u4e3b\u4f53": "MainBody",
    "\u5927": "Large",
    "\u5c0f": "Small",
    "\u4e2d\u53f7": "Medium",
    "\u957f": "Long",
    "\u77ed": "Short",
    "\u7c97": "Thick",
    "\u7ec6": "Thin",
    "\u539a": "Thick",
    "\u8584": "Thin",
    "\u9ad8": "High",
    "\u4f4e": "Low",
    "\u65b0": "New",
    "\u65e7": "Old",
    "\u5907\u4efd": "Backup",
    "\u526f\u672c": "Copy",
    "\u590d\u5236": "Copy",
    "\u6d4b\u8bd5": "Test",
    "\u4e34\u65f6": "Temp",
    "\u9ed8\u8ba4": "Default",
    "\u672a\u547d\u540d": "Unnamed",
    "\u65e0": "None",
    "\u7a7a": "Empty",
    "\u7ec4": "Group",
    "\u96c6\u5408": "Collection",
    "\u5c42": "Layer",
    "\u6bb5": "Segment",
    "\u8282": "Segment",
    "\u6839\u90e8": "Base",
    "\u5c16\u7aef": "Tip",
    "\u672b\u7aef": "End",
    "\u8d77\u70b9": "Start",
    "\u7ec8\u70b9": "End",
    "\u7236": "Parent",
    "\u7236\u7ea7": "Parent",
    "\u5b50": "Child",
    "\u63a7\u5236": "Control",
    "\u63a7\u5236\u5668": "Controller",
    "\u8f85\u52a9": "Helper",
    "\u76ee\u6807": "Target",
    "\u6781\u5411\u91cf": "PoleTarget",
    "\u53cd\u5411": "Reverse",

    # --- physics / simulation --------------------------------------------
    "\u7269\u7406": "Physics",
    "\u78b0\u649e": "Collision",
    "\u521a\u4f53": "RigidBody",
    "\u67d4\u4f53": "SoftBody",
    "\u5f39\u7c27": "Spring",
    "\u52a8\u529b\u5b66": "Dynamics",
    "\u6a21\u62df": "Simulation",
    "\u7c92\u5b50": "Particle",
    "\u70df\u96fe": "Smoke",
    "\u6d41\u4f53": "Fluid",

    # --- blender data -----------------------------------------------------
    "\u7269\u4f53": "Object",
    "\u5bf9\u8c61": "Object",
    "\u7f51\u683c": "Mesh",
    "\u6a21\u578b": "Model",
    "\u89d2\u8272": "Character",
    "\u4eba\u7269": "Character",
    "\u573a\u666f": "Scene",
    "\u6444\u50cf\u673a": "Camera",
    "\u76f8\u673a": "Camera",
    "\u706f\u5149": "Light",
    "\u706f": "Lamp",
    "\u592a\u9633": "Sun",
    "\u5929\u7a7a": "Sky",
    "\u5730\u9762": "Ground",
    "\u80cc\u666f": "Background",
    "\u52a8\u753b": "Animation",
    "\u52a8\u4f5c": "Action",
    "\u59ff\u52bf": "Pose",
    "\u5f62\u6001\u952e": "ShapeKey",
    "\u53d8\u5f62": "Morph",
    "\u9876\u70b9\u7ec4": "VertexGroup",
    "\u9876\u70b9": "Vertex",
    "\u6743\u91cd": "Weight",
    "\u4fee\u6539\u5668": "Modifier",
    "\u7ea6\u675f": "Constraint",
    "\u955c\u50cf": "Mirror",
    "\u7ec6\u5206": "Subdivision",
    "\u66f2\u7ebf": "Curve",
    "\u66f2\u9762": "Surface",
    "\u6587\u672c": "Text",
    "\u5c55\u5f00": "Unwrap",

    # --- katakana / MMD leftovers ----------------------------------------
    "\u30bb\u30f3\u30bf\u30fc": "Center",
    "\u30b0\u30eb\u30fc\u30d6": "Groove",
    "\u30b9\u30ab\u30fc\u30c8": "Skirt",
    "\u30ea\u30dc\u30f3": "Ribbon",
    "\u30cd\u30af\u30bf\u30a4": "Necktie",
    "\u30de\u30d5\u30e9\u30fc": "Muffler",
    "\u30dd\u30cb\u30fc": "Ponytail",
    "\u30a2\u30af\u30bb\u30b5\u30ea": "Accessory",
    "\u30de\u30f3\u30c8": "Mantle",
    "\u30d9\u30eb\u30c8": "Belt",
    "\u30d6\u30fc\u30c4": "Boots",
    "\u30cf\u30a4\u30d2\u30fc\u30eb": "HighHeel",
    "\u30ec\u30fc\u30b9": "Lace",
    "\u30d5\u30ea\u30eb": "Frill",
    "\u30d0\u30f3\u30c9": "Band",
    "\u30d4\u30a2\u30b9": "Earring",
    "\u30e1\u30ac\u30cd": "Glasses",
    "\u3072\u3058": "Elbow",
    "\u3072\u3056": "Knee",
    "\u3064\u307e\u5148": "Toe",
    "\u307e\u3086": "Eyebrow",
    "\u307e\u3064\u6bdb": "Eyelash",
    "\u4eb2": "Parent",

    # --- numbers ----------------------------------------------------------
    "\u96f6": "0", "\u4e00": "1", "\u4e8c": "2", "\u4e09": "3", "\u56db": "4",
    "\u4e94": "5", "\u516d": "6", "\u4e03": "7", "\u516b": "8", "\u4e5d": "9",
    "\u5341": "10", "\u5341\u4e00": "11", "\u5341\u4e8c": "12", "\u5341\u4e09": "13",
    "\u5341\u56db": "14", "\u5341\u4e94": "15", "\u5341\u516d": "16",
    "\u5341\u4e03": "17", "\u5341\u516b": "18", "\u5341\u4e5d": "19",
    "\u4e8c\u5341": "20",
}

# Filled in by rebuild_lookup(); the user dictionary is merged on top.
_LOOKUP = {}
_LOOKUP_MAX = 1
_USER_ENTRIES = {}
_USER_DICT_INFO = "no custom dictionary loaded"

CJK_PATTERN = re.compile(
    "["
    "\u3040-\u30ff"      # kana
    "\u3400-\u4dbf"      # CJK ext A
    "\u4e00-\u9fff"      # CJK unified
    "\uf900-\ufaff"      # compatibility
    "\uff66-\uff9f"      # half width kana
    "]"
)
SEPARATOR_CHARS = " _-.+/\\|,()[]{}<>:;#~*'\"\u3001\u3002\uff0c\uff08\uff09\u30fb\uff5c"
INDEX_SUFFIX_PATTERN = re.compile(r"\.\d{3}$")
# A side marker only counts when it stands on its own, i.e. it is fenced off by
# a separator (or by the start / end of the name). That keeps "Ctrl", "Roll" and
# "eyeR_bone" out of the way while still catching "Hand_R", "R_Hand" and the
# infix form "Skn_R_WingC_ZF_01" that MMD / game rigs like to use.
ASCII_SIDE_WORD = r"(L|R|Left|Right)"
ASCII_SIDE_SEP = r"[._\- ]"
ASCII_SIDE_TAIL = re.compile(ASCII_SIDE_SEP + ASCII_SIDE_WORD + r"$", re.IGNORECASE)
ASCII_SIDE_HEAD = re.compile(r"^" + ASCII_SIDE_WORD + ASCII_SIDE_SEP, re.IGNORECASE)
# Infix marker: separator, the marker, then another separator that is only
# looked at, never consumed, so "A_L_B" loses one separator rather than both.
ASCII_SIDE_MIDDLE = re.compile(
    ASCII_SIDE_SEP + ASCII_SIDE_WORD + r"(?=" + ASCII_SIDE_SEP + r")", re.IGNORECASE)
WORD_SPLIT = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+")

DEFAULT_NOUN = {
    "OBJECT": "Object",
    "DATA": "Data",
    "MATERIAL": "Material",
    "IMAGE": "Image",
    "BONE": "Bone",
    "BONECOLL": "BoneCollection",
    "VGROUP": "Group",
    "SHAPEKEY": "Key",
    "UVMAP": "UVMap",
    "COLATTR": "Color",
    "MODIFIER": "Modifier",
    "CONSTRAINT": "Constraint",
    "ACTION": "Action",
    "COLLECTION": "Collection",
    "NODEGROUP": "NodeGroup",
}


def rebuild_lookup():
    """Merge the built-in table with any user entries and cache the max key length."""
    global _LOOKUP, _LOOKUP_MAX
    merged = dict(BASE_DICTIONARY)
    merged.update(_USER_ENTRIES)
    _LOOKUP = merged
    lengths = [len(k) for k in merged] + [len(k) for k in SIDE_TABLE]
    _LOOKUP_MAX = max(lengths) if lengths else 1


rebuild_lookup()


def normalize_variants(text):
    """Map traditional / Japanese kanji onto their simplified equivalents."""
    return "".join(VARIANT_TO_SIMPLIFIED.get(ch, ch) for ch in text)


def has_source_script(text):
    return bool(CJK_PATTERN.search(text))


def split_index_suffix(name):
    """Split Blender's ".001" duplicate suffix off a name."""
    match = INDEX_SUFFIX_PATTERN.search(name)
    if match:
        return name[: match.start()], match.group(0)
    return name, ""


def find_ascii_side(name):
    """Locate an existing .L / _Right style marker in a latin name.

    Returns (body, side, position) where position is "TAIL", "HEAD", "MIDDLE"
    or "" when there is no marker. The end of the name is searched first,
    then the start, then the middle - so "L_Hand_R" reports the tail marker,
    exactly as it did before infix markers were understood.

    The marker is removed from body together with one of its neighbouring
    separators, so "Skn_R_WingC_ZF_01" comes back as "Skn_WingC_ZF_01" and not
    as "Skn__WingC_ZF_01".
    """
    match = ASCII_SIDE_TAIL.search(name)
    if match:
        return name[: match.start()], match.group(1)[0].upper(), "TAIL"
    match = ASCII_SIDE_HEAD.match(name)
    if match:
        return name[match.end():], match.group(1)[0].upper(), "HEAD"
    match = ASCII_SIDE_MIDDLE.search(name)
    if match:
        body = name[: match.start()] + name[match.end():]
        return body, match.group(1)[0].upper(), "MIDDLE"
    return name, None, ""


def strip_ascii_side(name):
    """Pull an existing .L / _Right style marker off a name."""
    body, side, _position = find_ascii_side(name)
    return body, side


def split_words(chunk):
    """Break a PascalCase / snake_case fragment into individual words."""
    words = WORD_SPLIT.findall(chunk)
    return words if words else ([chunk] if chunk else [])


def tokenize(text):
    """Greedy longest-match tokenizer.

    Returns a list of (kind, value) pairs where kind is one of:
        'word'    already-translated English fragment
        'ascii'   text that was already latin, passed through untouched
        'sep'     separator characters from the original name
        'unknown' source-language text with no dictionary entry
    plus a list of the sides that were found and the unknown fragments.
    """
    tokens = []
    sides = []
    unknown = []
    normalized = normalize_variants(text)
    i = 0
    total = len(text)
    pending_unknown = ""

    def flush_unknown():
        nonlocal pending_unknown
        if pending_unknown:
            tokens.append(("unknown", pending_unknown))
            unknown.append(pending_unknown)
            pending_unknown = ""

    while i < total:
        char = text[i]

        if char in SEPARATOR_CHARS:
            flush_unknown()
            tokens.append(("sep", char))
            i += 1
            continue

        if char.isascii() and (char.isalnum() or char == "%"):
            flush_unknown()
            start = i
            while i < total and text[i].isascii() and (text[i].isalnum() or text[i] == "%"):
                i += 1
            tokens.append(("ascii", text[start:i]))
            continue

        matched = False
        for length in range(min(_LOOKUP_MAX, total - i), 0, -1):
            segment = normalized[i:i + length]
            if segment in SIDE_TABLE:
                flush_unknown()
                side = SIDE_TABLE[segment]
                if side:
                    sides.append(side)
                i += length
                matched = True
                break
            if segment in _LOOKUP:
                flush_unknown()
                tokens.append(("word", _LOOKUP[segment]))
                i += length
                matched = True
                break
        if matched:
            continue

        pending_unknown += char
        i += 1

    flush_unknown()
    return tokens, sides, unknown


def assemble(tokens, style):
    """Render tokens back into a name using the requested style."""
    if style == "RAW":
        return "".join(value for _, value in tokens)

    words = []
    for kind, value in tokens:
        if kind == "sep":
            continue
        if kind == "unknown":
            words.append(value)
        else:
            words.extend(split_words(value))

    words = [w for w in words if w]
    if not words:
        return ""

    if style == "PASCAL":
        return "".join(w if w.isdigit() else w[:1].upper() + w[1:] for w in words)
    if style == "CAMEL":
        out = "".join(w if w.isdigit() else w[:1].upper() + w[1:] for w in words)
        return out[:1].lower() + out[1:]
    if style == "SNAKE":
        return "_".join(w.lower() for w in words)
    if style == "KEBAB":
        return "-".join(w.lower() for w in words)
    if style == "SPACED":
        return " ".join(w if w.isdigit() else w[:1].upper() + w[1:] for w in words)
    return "".join(words)


def format_side(side, side_style):
    if not side or side_style == "NONE":
        return ""
    if side_style == "DOT":
        return ".%s" % side
    if side_style == "UNDERSCORE":
        return "_%s" % side
    if side_style == "DOT_WORD":
        return ".Left" if side == "L" else ".Right"
    return ".%s" % side


def translate_name(name, style="PASCAL", side_style="DOT", kind="OBJECT",
                   online_lookup=None):
    """Translate a single datablock name.

    online_lookup: optional dict mapping original text -> pre-fetched translation.
    Returns a dict with the new name, the detected side and any unknown fragments.
    """
    base, index_suffix = split_index_suffix(name)
    body, ascii_side = strip_ascii_side(base)

    replaced = None
    if online_lookup is not None:
        replaced = online_lookup.get(body)

    if replaced:
        # Online output is English; recycle the same pipeline to catch
        # "left"/"right" words and to apply the naming style.
        body_source = replaced
    else:
        body_source = body

    tokens, sides, unknown = tokenize(body_source)

    # An English result may still spell the side out as a word.
    filtered = []
    for token_kind, value in tokens:
        if token_kind == "ascii" and value.lower() in ("left", "l"):
            sides.append("L")
            continue
        if token_kind == "ascii" and value.lower() in ("right", "r"):
            sides.append("R")
            continue
        filtered.append((token_kind, value))
    tokens = filtered

    side = None
    conflict = False
    unique_sides = set(sides)
    if len(unique_sides) > 1:
        conflict = True
        side = sides[0]
    elif unique_sides:
        side = sides.pop()

    if side is None and ascii_side:
        side = ascii_side
    elif side and ascii_side and side != ascii_side:
        conflict = True

    if side and side_style == "NONE":
        # The suffix is switched off, so keep the side as a normal word rather
        # than silently throwing the information away.
        tokens = [("word", "Left" if side == "L" else "Right")] + tokens

    new_body = assemble(tokens, style).strip("._- ")
    if not new_body:
        new_body = DEFAULT_NOUN.get(kind, "Name")

    return {
        "name": new_body + format_side(side, side_style) + index_suffix,
        "body": new_body,
        "side": side or "",
        "unknown": unknown,
        "conflict": conflict,
        "source_body": body,
        "suffix": index_suffix,
    }


def rebuild_with_side(result, side, side_style):
    """Rebuild a name after the side has been corrected."""
    return result["body"] + format_side(side, side_style) + result["suffix"]


# ---------------------------------------------------------------------------
# 2. Side-of-the-body maths
# ---------------------------------------------------------------------------

FRONT_VECTORS = {
    "-Y": Vector((0.0, -1.0, 0.0)),
    "+Y": Vector((0.0, 1.0, 0.0)),
    "-X": Vector((-1.0, 0.0, 0.0)),
    "+X": Vector((1.0, 0.0, 0.0)),
}


def left_vector(front_axis):
    """Character-left direction, given which way the character faces.

    up x forward = left, so with Blender's default (character facing -Y) the
    character's left hand sits at +X - which is exactly what Blender's
    Symmetrize and Flip Names expect from a ".L" bone.
    """
    front = FRONT_VECTORS.get(front_axis, FRONT_VECTORS["-Y"])
    return Vector((0.0, 0.0, 1.0)).cross(front)


def side_from_position(position, front_axis, tolerance):
    """Return 'L', 'R' or '' (too close to the mirror plane to tell)."""
    if position is None:
        return None
    offset = position.dot(left_vector(front_axis))
    if abs(offset) <= tolerance:
        return ""
    return "L" if offset > 0.0 else "R"


# ---------------------------------------------------------------------------
# 3. Optional online translation backends
# ---------------------------------------------------------------------------

_ONLINE_CACHE = {}


def _http_json(url, data=None, headers=None, timeout=10):
    import urllib.request

    request = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def google_translate(texts, source, target, timeout):
    """Unofficial public endpoint. Free, no key, but not guaranteed to keep working."""
    import urllib.parse

    out = {}
    for text in texts:
        query = urllib.parse.urlencode({
            "client": "gtx",
            "sl": source or "auto",
            "tl": target or "en",
            "dt": "t",
            "q": text,
        })
        url = "https://translate.googleapis.com/translate_a/single?" + query
        payload = _http_json(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        if payload and payload[0]:
            out[text] = "".join(part[0] for part in payload[0] if part and part[0])
    return out


def deepl_translate(texts, source, target, timeout, api_key):
    """Official DeepL API. Needs a free or pro key."""
    if not api_key:
        raise RuntimeError("No DeepL API key set in the add-on preferences")

    host = "https://api-free.deepl.com" if api_key.strip().endswith(":fx") else "https://api.deepl.com"
    out = {}
    batch_size = 45
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        body = {
            "text": batch,
            "target_lang": (target or "en").split("-")[0].upper(),
        }
        if source:
            body["source_lang"] = source.split("-")[0].upper()
        payload = _http_json(
            host + "/v2/translate",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": "DeepL-Auth-Key " + api_key.strip(),
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
        for original, entry in zip(batch, payload.get("translations", [])):
            out[original] = entry.get("text", "")
    return out


def fetch_online(texts, settings, prefs):
    """Translate a list of strings, using and filling the session cache."""
    todo = []
    for text in texts:
        cache_key = (settings.online_service, settings.source_lang, settings.target_lang, text)
        if cache_key not in _ONLINE_CACHE and text.strip():
            todo.append(text)
    todo = sorted(set(todo))

    if todo:
        timeout = getattr(prefs, "request_timeout", 10.0)
        if settings.online_service == "DEEPL":
            fetched = deepl_translate(
                todo, settings.source_lang, settings.target_lang, timeout,
                getattr(prefs, "deepl_api_key", ""),
            )
        else:
            fetched = google_translate(
                todo, settings.source_lang, settings.target_lang, timeout,
            )
        for original, translated in fetched.items():
            key = (settings.online_service, settings.source_lang, settings.target_lang, original)
            _ONLINE_CACHE[key] = translated

    result = {}
    for text in texts:
        key = (settings.online_service, settings.source_lang, settings.target_lang, text)
        if key in _ONLINE_CACHE:
            result[text] = _ONLINE_CACHE[key]
    return result


# ---------------------------------------------------------------------------
# 4. Geometry helpers
# ---------------------------------------------------------------------------

_VGROUP_CACHE = {}


def clear_caches():
    _VGROUP_CACHE.clear()


def object_center(obj, use_world):
    corners = [Vector(corner) for corner in obj.bound_box]
    if not corners:
        return obj.matrix_world.translation.copy() if use_world else Vector()
    center = sum(corners, Vector()) / len(corners)
    return (obj.matrix_world @ center) if use_world else center


def bone_center(obj, bone, use_world):
    try:
        middle = (Vector(bone.head_local) + Vector(bone.tail_local)) * 0.5
    except AttributeError:
        return None
    return (obj.matrix_world @ middle) if use_world else middle


def vertex_group_centers(obj, use_world):
    """Weighted centre of every vertex group on a mesh, computed in one pass."""
    cache_key = (obj.name, use_world)
    cached = _VGROUP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    centers = {}
    mesh = getattr(obj, "data", None)
    vertices = getattr(mesh, "vertices", None)
    if vertices:
        totals = {}
        for vertex in vertices:
            for group in vertex.groups:
                weight = group.weight
                if weight <= 0.0:
                    continue
                entry = totals.get(group.group)
                if entry is None:
                    entry = [Vector(), 0.0]
                    totals[group.group] = entry
                entry[0] += vertex.co * weight
                entry[1] += weight
        for index, (accumulated, weight_sum) in totals.items():
            if weight_sum > 0.0:
                point = accumulated / weight_sum
                centers[index] = (obj.matrix_world @ point) if use_world else point

    _VGROUP_CACHE[cache_key] = centers
    return centers


def shape_key_center(obj, key_block, use_world):
    """Centre of the region a shape key actually moves."""
    shape_keys = getattr(getattr(obj, "data", None), "shape_keys", None)
    if not shape_keys or not shape_keys.key_blocks:
        return None
    basis = shape_keys.key_blocks[0]
    if key_block == basis:
        return None
    count = len(key_block.data)
    if count == 0 or count != len(basis.data):
        return None

    try:
        import numpy as np

        moved = np.empty(count * 3, dtype=np.float32)
        rest = np.empty(count * 3, dtype=np.float32)
        key_block.data.foreach_get("co", moved)
        basis.data.foreach_get("co", rest)
        moved = moved.reshape(count, 3)
        rest = rest.reshape(count, 3)
        magnitude = np.linalg.norm(moved - rest, axis=1)
        peak = float(magnitude.max()) if magnitude.size else 0.0
        if peak < 1e-9:
            return None
        selection = magnitude > peak * 0.1
        weights = magnitude[selection]
        points = moved[selection]
        center = (points * weights[:, None]).sum(axis=0) / weights.sum()
        point = Vector((float(center[0]), float(center[1]), float(center[2])))
    except Exception:
        best = None
        best_distance = 0.0
        for index in range(count):
            delta = (key_block.data[index].co - basis.data[index].co).length
            if delta > best_distance:
                best_distance = delta
                best = key_block.data[index].co.copy()
        if best is None or best_distance < 1e-9:
            return None
        point = best

    return (obj.matrix_world @ point) if use_world else point


def material_center(obj, slot_index, use_world):
    """Centre of the faces that use one material slot."""
    mesh = getattr(obj, "data", None)
    polygons = getattr(mesh, "polygons", None)
    if not polygons:
        return None
    accumulated = Vector()
    count = 0
    for polygon in polygons:
        if polygon.material_index == slot_index:
            accumulated += polygon.center
            count += 1
    if not count:
        return None
    point = accumulated / count
    return (obj.matrix_world @ point) if use_world else point


# ---------------------------------------------------------------------------
# 5. Gathering everything that could be renamed
# ---------------------------------------------------------------------------

def target_objects(context, settings):
    if settings.scope == "ACTIVE":
        active = context.view_layer.objects.active
        return [active] if active else []
    if settings.scope == "ALL":
        return list(context.scene.objects)
    return [obj for obj in context.selected_objects]


def _walk_node_groups(node_tree, found, seen):
    if node_tree is None or node_tree.as_pointer() in seen:
        return
    seen.add(node_tree.as_pointer())
    for node in getattr(node_tree, "nodes", []):
        group = getattr(node, "node_tree", None)
        if group is not None:
            found.append(group)
            _walk_node_groups(group, found, seen)


def collect_entries(context, settings):
    """Build the raw list of rename candidates.

    Each entry carries a lazily-evaluated position callback so geometry is only
    touched for names that actually claim a side.
    """
    entries = []
    seen = set()
    use_world = settings.use_world_space

    # Datablocks that live in bpy.data and can be shared by several objects.
    # They must only appear once in the list even if many objects use them.
    shared_kinds = {"DATA", "MATERIAL", "IMAGE", "ACTION", "COLLECTION", "NODEGROUP"}

    def add(kind, name, owner="", sub_owner="", namespace="", position=None):
        if not name:
            return
        key = (kind, namespace, "" if kind in shared_kinds else owner, sub_owner, name)
        if key in seen:
            return
        seen.add(key)
        entries.append({
            "kind": kind,
            "name": name,
            "owner": owner,
            "sub_owner": sub_owner,
            "namespace": namespace or kind,
            "position": position,
        })

    objects = [obj for obj in target_objects(context, settings) if obj is not None]
    node_group_sources = []

    for obj in objects:
        if settings.do_objects:
            add("OBJECT", obj.name, owner=obj.name, namespace="OBJECT",
                position=(lambda o=obj: object_center(o, use_world)))

        data = getattr(obj, "data", None)
        if settings.do_data and data is not None:
            add("DATA", data.name, owner=obj.name, namespace="DATA:" + type(data).__name__,
                position=(lambda o=obj: object_center(o, use_world)))

        if settings.do_materials:
            for slot_index, slot in enumerate(obj.material_slots):
                material = slot.material
                if material is None:
                    continue
                add("MATERIAL", material.name, owner=obj.name, namespace="MATERIAL",
                    position=(lambda o=obj, i=slot_index: material_center(o, i, use_world)))
                node_group_sources.append(material)

        if settings.do_images:
            for slot in obj.material_slots:
                material = slot.material
                if material is None or not material.use_nodes:
                    continue
                for node in material.node_tree.nodes:
                    image = getattr(node, "image", None)
                    if image is not None:
                        add("IMAGE", image.name, namespace="IMAGE")

        if obj.type == "ARMATURE" and data is not None:
            if settings.do_bones:
                for bone in data.bones:
                    add("BONE", bone.name, owner=obj.name, namespace="BONE:" + obj.name,
                        position=(lambda o=obj, b=bone: bone_center(o, b, use_world)))
            if settings.do_bone_collections:
                for collection in getattr(data, "collections_all", []):
                    add("BONECOLL", collection.name, owner=obj.name,
                        namespace="BONECOLL:" + obj.name)
            if settings.do_constraints and obj.pose:
                for pose_bone in obj.pose.bones:
                    for constraint in pose_bone.constraints:
                        add("CONSTRAINT", constraint.name, owner=obj.name,
                            sub_owner=pose_bone.name,
                            namespace="CONSTRAINT:%s:%s" % (obj.name, pose_bone.name))

        if settings.do_vertex_groups and obj.vertex_groups:
            for group in obj.vertex_groups:
                def group_position(o=obj, index=group.index):
                    return vertex_group_centers(o, use_world).get(index)
                add("VGROUP", group.name, owner=obj.name, namespace="VGROUP:" + obj.name,
                    position=group_position)

        shape_keys = getattr(data, "shape_keys", None)
        if settings.do_shape_keys and shape_keys is not None:
            for key_block in shape_keys.key_blocks:
                add("SHAPEKEY", key_block.name, owner=obj.name,
                    namespace="SHAPEKEY:" + obj.name,
                    position=(lambda o=obj, k=key_block: shape_key_center(o, k, use_world)))

        if settings.do_uv_maps and hasattr(data, "uv_layers"):
            for uv_layer in data.uv_layers:
                add("UVMAP", uv_layer.name, owner=obj.name, namespace="UVMAP:" + obj.name)

        if settings.do_color_attributes and hasattr(data, "color_attributes"):
            for attribute in data.color_attributes:
                add("COLATTR", attribute.name, owner=obj.name,
                    namespace="COLATTR:" + obj.name)

        if settings.do_modifiers:
            for modifier in obj.modifiers:
                add("MODIFIER", modifier.name, owner=obj.name,
                    namespace="MODIFIER:" + obj.name)

        if settings.do_constraints:
            for constraint in obj.constraints:
                add("CONSTRAINT", constraint.name, owner=obj.name,
                    namespace="CONSTRAINT:" + obj.name)

        if settings.do_actions:
            for holder in (obj, data, shape_keys):
                animation_data = getattr(holder, "animation_data", None)
                if animation_data is None:
                    continue
                if animation_data.action is not None:
                    add("ACTION", animation_data.action.name, namespace="ACTION")
                for track in animation_data.nla_tracks:
                    for strip in track.strips:
                        if strip.action is not None:
                            add("ACTION", strip.action.name, namespace="ACTION")

        if settings.do_collections:
            for collection in obj.users_collection:
                if collection is not context.scene.collection:
                    add("COLLECTION", collection.name, namespace="COLLECTION")

    if settings.do_node_groups:
        found = []
        seen_trees = set()
        for material in node_group_sources:
            if material.use_nodes:
                _walk_node_groups(material.node_tree, found, seen_trees)
        for obj in objects:
            for modifier in obj.modifiers:
                if modifier.type == "NODES" and getattr(modifier, "node_group", None):
                    found.append(modifier.node_group)
                    _walk_node_groups(modifier.node_group, found, seen_trees)
        for group in found:
            add("NODEGROUP", group.name, namespace="NODEGROUP")

    return entries


# ---------------------------------------------------------------------------
# 6. Data stored on the scene
# ---------------------------------------------------------------------------

KIND_ICON = {
    "OBJECT": "OBJECT_DATA",
    "DATA": "MESH_DATA",
    "MATERIAL": "MATERIAL",
    "IMAGE": "IMAGE_DATA",
    "BONE": "BONE_DATA",
    "BONECOLL": "GROUP_BONE",
    "VGROUP": "GROUP_VERTEX",
    "SHAPEKEY": "SHAPEKEY_DATA",
    "UVMAP": "UV",
    "COLATTR": "COLOR",
    "MODIFIER": "MODIFIER",
    "CONSTRAINT": "CONSTRAINT",
    "ACTION": "ACTION",
    "COLLECTION": "OUTLINER_COLLECTION",
    "NODEGROUP": "NODETREE",
}

KIND_LABEL = {
    "OBJECT": "Object",
    "DATA": "Object data",
    "MATERIAL": "Material",
    "IMAGE": "Image",
    "BONE": "Bone",
    "BONECOLL": "Bone collection",
    "VGROUP": "Vertex group",
    "SHAPEKEY": "Shape key",
    "UVMAP": "UV map",
    "COLATTR": "Color attribute",
    "MODIFIER": "Modifier",
    "CONSTRAINT": "Constraint",
    "ACTION": "Action",
    "COLLECTION": "Collection",
    "NODEGROUP": "Node group",
}

STATUS_ICON = {
    "OK": "CHECKMARK",
    "SIDE_FIXED": "FILE_REFRESH",
    "SIDE_MISMATCH": "ERROR",
    "SIDE_AMBIGUOUS": "QUESTION",
    "SIDE_UNPAIRED": "UNLINKED",
    "SIDE_CONFLICT": "ERROR",
    "UNTRANSLATED": "INFO",
    "COLLISION": "DUPLICATE",
}


class NT_Item(PropertyGroup):
    use: BoolProperty(name="Apply", default=True,
                      description="Include this row when applying")
    kind: StringProperty()
    owner: StringProperty()
    sub_owner: StringProperty()
    old_name: StringProperty()
    new_name: StringProperty(name="New name",
                             description="Edit freely before applying")
    side: StringProperty()
    geometry_side: StringProperty()
    status: StringProperty(default="OK")
    note: StringProperty()

    @property
    def is_issue(self):
        return self.status not in ("OK", "SIDE_FIXED")


class NT_Settings(PropertyGroup):
    # --- translation source ---------------------------------------------
    backend: EnumProperty(
        name="Source",
        items=[
            ("DICT", "Dictionary only", "Offline. Uses the built-in table plus your own dictionary file"),
            ("HYBRID", "Dictionary + online", "Dictionary first, send only the leftovers to an online service"),
            ("ONLINE", "Online only", "Send every name to an online translation service"),
        ],
        default="DICT",
    )
    online_service: EnumProperty(
        name="Service",
        items=[
            ("GOOGLE", "Google (no key)", "Free public endpoint, no API key, may break without notice"),
            ("DEEPL", "DeepL (API key)", "Needs a free or pro DeepL key in the add-on preferences"),
        ],
        default="GOOGLE",
    )
    source_lang: StringProperty(name="From", default="zh-CN",
                                description="Source language code, e.g. zh-CN, ja, ko")
    target_lang: StringProperty(name="To", default="en",
                                description="Target language code, e.g. en, de, fr")

    scope: EnumProperty(
        name="Scope",
        items=[
            ("SELECTED", "Selected objects", "Every selected object"),
            ("ACTIVE", "Active object", "Only the active object"),
            ("ALL", "Whole scene", "Every object in the scene"),
        ],
        default="SELECTED",
    )

    # --- what to translate ----------------------------------------------
    do_objects: BoolProperty(name="Objects", default=True)
    do_data: BoolProperty(name="Object data (mesh, armature)", default=True)
    do_materials: BoolProperty(name="Materials", default=True)
    do_images: BoolProperty(name="Images / textures", default=False)
    do_bones: BoolProperty(name="Bones", default=True)
    do_bone_collections: BoolProperty(name="Bone collections", default=True)
    do_vertex_groups: BoolProperty(name="Vertex groups", default=True)
    do_shape_keys: BoolProperty(name="Shape keys", default=True)
    do_uv_maps: BoolProperty(name="UV maps", default=False)
    do_color_attributes: BoolProperty(name="Color attributes", default=False)
    do_modifiers: BoolProperty(name="Modifiers", default=False)
    do_constraints: BoolProperty(name="Constraints", default=False)
    do_actions: BoolProperty(name="Actions", default=False)
    do_collections: BoolProperty(name="Collections", default=False)
    do_node_groups: BoolProperty(name="Node groups", default=False)

    # --- naming ----------------------------------------------------------
    style: EnumProperty(
        name="Style",
        items=[
            ("PASCAL", "PascalCase", "UpperArm.L"),
            ("SNAKE", "snake_case", "upper_arm.L"),
            ("SPACED", "Spaced Words", "Upper Arm.L"),
            ("KEBAB", "kebab-case", "upper-arm.L"),
            ("CAMEL", "camelCase", "upperArm.L"),
            ("RAW", "Minimal edit", "Only swap the translated words, keep the original layout"),
        ],
        default="PASCAL",
    )
    side_style: EnumProperty(
        name="Side suffix",
        items=[
            ("DOT", ".L / .R", "Blender's convention - required for Symmetrize, Flip Names and mirrored weights"),
            ("UNDERSCORE", "_L / _R", "Also recognised by Blender, but less standard"),
            ("DOT_WORD", ".Left / .Right", "Also recognised by Blender"),
            ("NONE", "Keep as a word", "Do not convert side markers into a suffix"),
        ],
        default="DOT",
    )
    skip_non_source: BoolProperty(
        name="Skip names already in the target language",
        default=True,
        description="Leave names that contain no source-language characters alone "
                    "(they are still checked for left/right problems)",
    )

    # --- symmetry --------------------------------------------------------
    check_symmetry: BoolProperty(
        name="Cross-check left / right against geometry",
        default=True,
        description="Compare each side marker with where the item actually sits in space",
    )
    auto_fix_side: BoolProperty(
        name="Trust geometry and fix the label",
        default=False,
        description="When the name and the geometry disagree, rename to match the geometry "
                    "instead of only warning",
    )
    detect_missing_side: BoolProperty(
        name="Add a missing side marker",
        default=False,
        description="Give a side suffix to items that clearly sit on one side but have no marker. "
                    "Slower, because every item's position has to be measured",
    )
    front_axis: EnumProperty(
        name="Character faces",
        items=[
            ("-Y", "-Y (Blender default)", "Character looks towards -Y, so its left hand is at +X"),
            ("+Y", "+Y", "Character looks towards +Y, so its left hand is at -X"),
            ("-X", "-X", "Character looks towards -X"),
            ("+X", "+X", "Character looks towards +X"),
        ],
        default="-Y",
    )
    use_world_space: BoolProperty(
        name="Use world space",
        default=True,
        description="Measure positions in world space. Turn off if the character is parented "
                    "or offset and you want to measure inside its own object space",
    )
    tolerance: FloatProperty(
        name="Center tolerance",
        default=0.001, min=0.0, soft_max=1.0, precision=4,
        description="Items closer than this to the mirror plane are reported as ambiguous "
                    "rather than assigned a side",
    )

    # --- results ---------------------------------------------------------
    items: CollectionProperty(type=NT_Item)
    active_index: IntProperty(default=0)
    show_only_issues: BoolProperty(
        name="Only show rows that need attention", default=False)
    history: StringProperty(default="", options={"HIDDEN"})
    last_summary: StringProperty(default="")


class NT_Preferences(AddonPreferences):
    bl_idname = ADDON_ID

    dictionary_path: StringProperty(
        name="Custom dictionary",
        subtype="FILE_PATH",
        description="CSV (source,target per line) or JSON object. Entries here win over the built-in table",
    )
    deepl_api_key: StringProperty(
        name="DeepL API key", subtype="PASSWORD", default="")
    request_timeout: FloatProperty(
        name="Request timeout (s)", default=10.0, min=1.0, max=60.0)

    def draw(self, context):
        layout = self.layout
        column = layout.column()
        column.prop(self, "dictionary_path")
        row = column.row()
        row.operator("name_translator.reload_dictionary", icon="FILE_REFRESH")
        row.label(text=_USER_DICT_INFO)
        column.separator()
        column.prop(self, "deepl_api_key")
        column.prop(self, "request_timeout")
        column.separator()
        box = column.box()
        box.label(text="Built-in dictionary: %d entries" % len(_LOOKUP), icon="INFO")
        box.label(text="Online translation only runs if you pick it and Blender's "
                       "'Allow Online Access' preference is enabled.")


def get_prefs():
    try:
        return bpy.context.preferences.addons[ADDON_ID].preferences
    except (KeyError, AttributeError):
        return None


def load_user_dictionary(path):
    """Load a CSV or JSON dictionary. Returns (entry_count, message)."""
    global _USER_ENTRIES, _USER_DICT_INFO

    if not path:
        _USER_ENTRIES = {}
        _USER_DICT_INFO = "no custom dictionary loaded"
        rebuild_lookup()
        return 0, _USER_DICT_INFO

    resolved = bpy.path.abspath(path)
    if not os.path.isfile(resolved):
        _USER_DICT_INFO = "file not found"
        return 0, "Dictionary file not found: %s" % resolved

    entries = {}
    try:
        with open(resolved, "r", encoding="utf-8-sig") as handle:
            if resolved.lower().endswith(".json"):
                payload = json.load(handle)
                entries = {str(k): str(v) for k, v in payload.items() if k and v}
            else:
                import csv

                for row in csv.reader(handle):
                    if len(row) < 2:
                        continue
                    source = row[0].strip()
                    target = row[1].strip()
                    if source and target and not source.startswith("#"):
                        entries[source] = target
    except Exception as error:
        _USER_DICT_INFO = "failed to read"
        return 0, "Could not read dictionary: %s" % error

    _USER_ENTRIES = entries
    _USER_DICT_INFO = "%d custom entries" % len(entries)
    rebuild_lookup()
    return len(entries), _USER_DICT_INFO


# ---------------------------------------------------------------------------
# 7. Operators
# ---------------------------------------------------------------------------

STATUS_RANK = {
    "OK": 0,
    "SIDE_FIXED": 1,
    "UNTRANSLATED": 2,
    "SIDE_UNPAIRED": 3,
    "SIDE_AMBIGUOUS": 4,
    "COLLISION": 5,
    "SIDE_CONFLICT": 6,
    "SIDE_MISMATCH": 7,
}

APPLY_ORDER = [
    "BONE", "BONECOLL", "VGROUP", "SHAPEKEY", "UVMAP", "COLATTR",
    "MODIFIER", "CONSTRAINT", "MATERIAL", "IMAGE", "ACTION",
    "NODEGROUP", "COLLECTION", "DATA", "OBJECT",
]


def worst(current, candidate):
    return candidate if STATUS_RANK.get(candidate, 0) > STATUS_RANK.get(current, 0) else current


def _lookup(collection, name):
    if collection is None:
        return None
    try:
        found = collection.get(name)
        if found is not None:
            return found
    except (AttributeError, TypeError):
        pass
    for element in collection:
        if getattr(element, "name", None) == name:
            return element
    return None


def resolve_target(kind, owner, sub_owner, name, bone_renames):
    """Find the struct whose .name should be changed, or None if it is gone."""
    if kind == "OBJECT":
        return bpy.data.objects.get(name)
    if kind == "MATERIAL":
        return bpy.data.materials.get(name)
    if kind == "IMAGE":
        return bpy.data.images.get(name)
    if kind == "ACTION":
        return bpy.data.actions.get(name)
    if kind == "COLLECTION":
        return bpy.data.collections.get(name)
    if kind == "NODEGROUP":
        return bpy.data.node_groups.get(name)

    obj = bpy.data.objects.get(owner)
    if obj is None:
        return None
    data = getattr(obj, "data", None)

    if kind == "DATA":
        return data if data is not None and data.name == name else None
    if kind == "BONE":
        return _lookup(getattr(data, "bones", None), name)
    if kind == "BONECOLL":
        return _lookup(getattr(data, "collections_all", None), name)
    if kind == "VGROUP":
        return _lookup(obj.vertex_groups, name)
    if kind == "SHAPEKEY":
        shape_keys = getattr(data, "shape_keys", None)
        return _lookup(getattr(shape_keys, "key_blocks", None), name)
    if kind == "UVMAP":
        return _lookup(getattr(data, "uv_layers", None), name)
    if kind == "COLATTR":
        return _lookup(getattr(data, "color_attributes", None), name)
    if kind == "MODIFIER":
        return _lookup(obj.modifiers, name)
    if kind == "CONSTRAINT":
        if sub_owner:
            if obj.pose is None:
                return None
            current_bone = bone_renames.get((owner, sub_owner), sub_owner)
            pose_bone = obj.pose.bones.get(current_bone)
            if pose_bone is None:
                return None
            return _lookup(pose_bone.constraints, name)
        return _lookup(obj.constraints, name)
    return None


class NT_OT_scan(Operator):
    bl_idname = "name_translator.scan"
    bl_label = "Scan Names"
    bl_description = "Build a preview of the renames. Nothing is changed yet"
    bl_options = {"REGISTER"}

    def execute(self, context):
        if context.mode.startswith("EDIT"):
            self.report({"ERROR"}, "Leave Edit Mode first - names cannot be read "
                                   "reliably from an object that is being edited")
            return {"CANCELLED"}

        settings = context.scene.name_translator
        prefs = get_prefs()
        clear_caches()
        settings.items.clear()

        entries = collect_entries(context, settings)
        if not entries:
            self.report({"WARNING"}, "Nothing found. Check the scope and the category toggles")
            return {"CANCELLED"}

        results = [
            translate_name(entry["name"], settings.style, settings.side_style, entry["kind"])
            for entry in entries
        ]

        if settings.backend in ("HYBRID", "ONLINE"):
            if not getattr(bpy.app, "online_access", True):
                self.report({"ERROR"},
                            "Online access is off. Enable Preferences > System > Network > "
                            "Allow Online Access, or use the dictionary backend")
                return {"CANCELLED"}

            wanted = set()
            for result in results:
                body = result["source_body"].strip()
                if not body:
                    continue
                if settings.backend == "ONLINE":
                    if has_source_script(body):
                        wanted.add(body)
                elif result["unknown"]:
                    wanted.add(body)

            if wanted:
                window_manager = context.window_manager
                window_manager.progress_begin(0, len(wanted))
                try:
                    online_lookup = fetch_online(sorted(wanted), settings, prefs)
                except Exception as error:
                    window_manager.progress_end()
                    self.report({"ERROR"}, "Online translation failed: %s" % error)
                    return {"CANCELLED"}
                window_manager.progress_end()

                for index, entry in enumerate(entries):
                    if results[index]["source_body"] in online_lookup:
                        results[index] = translate_name(
                            entry["name"], settings.style, settings.side_style,
                            entry["kind"], online_lookup=online_lookup,
                        )

        rows = []
        for entry, result in zip(entries, results):
            old_name = entry["name"]
            base, suffix = split_index_suffix(old_name)
            is_source = has_source_script(base)
            side = result["side"]
            status = "OK"
            note = ""

            geometry_side = None
            if settings.check_symmetry and (side or settings.detect_missing_side):
                position = None
                if entry["position"] is not None:
                    try:
                        position = entry["position"]()
                    except Exception:
                        position = None
                geometry_side = side_from_position(position, settings.front_axis, settings.tolerance)

            final_side = side
            if settings.check_symmetry:
                if side and geometry_side in ("L", "R") and geometry_side != side:
                    if settings.auto_fix_side:
                        final_side = geometry_side
                        status = "SIDE_FIXED"
                        note = "geometry says %s, label corrected" % geometry_side
                    else:
                        status = "SIDE_MISMATCH"
                        note = "name says %s but it sits on the %s" % (side, geometry_side)
                elif side and geometry_side == "":
                    status = "SIDE_AMBIGUOUS"
                    note = "too close to the mirror plane to confirm"
                elif not side and geometry_side in ("L", "R") and settings.detect_missing_side:
                    final_side = geometry_side
                    status = "SIDE_FIXED"
                    note = "side taken from geometry"

            if result["conflict"]:
                status = worst(status, "SIDE_CONFLICT")
                note = note or "the original name mixes left and right markers"

            if settings.skip_non_source and not is_source:
                original_body, original_side, side_position = find_ascii_side(base)
                # A marker sitting in the middle ("Skn_R_WingC_ZF_01") is invisible
                # to Blender's Symmetrize / Flip Names, so it is moved to the end
                # even when it already says the right thing.
                moved = side_position == "MIDDLE"
                if settings.side_style == "NONE":
                    # No suffix convention to apply - leave the name as the user
                    # wrote it rather than deleting the marker it already has.
                    new_name = old_name
                elif final_side and (final_side != original_side or moved):
                    new_name = original_body + format_side(final_side, settings.side_style) + suffix
                    if moved and not note:
                        note = "side marker moved out of the middle of the name"
                else:
                    new_name = old_name
            else:
                new_name = result["body"] + format_side(final_side, settings.side_style) + suffix
                if result["unknown"]:
                    status = worst(status, "UNTRANSLATED")
                    if not note:
                        note = "no entry for: " + " ".join(result["unknown"])

            rows.append({
                "entry": entry,
                "old": old_name,
                "new": new_name,
                "side": final_side or "",
                "geometry": geometry_side if geometry_side in ("L", "R") else "",
                "status": status,
                "note": note,
            })

        # Pairing: a .L bone should have a matching .R bone in the same namespace.
        if settings.side_style != "NONE":
            pairs = {}
            for row in rows:
                if not row["side"]:
                    continue
                body = strip_ascii_side(split_index_suffix(row["new"])[0])[0].lower()
                pairs.setdefault((row["entry"]["namespace"], body), set()).add(row["side"])
            for row in rows:
                if not row["side"] or row["status"] != "OK":
                    continue
                body = strip_ascii_side(split_index_suffix(row["new"])[0])[0].lower()
                found = pairs.get((row["entry"]["namespace"], body), set())
                if len(found) < 2:
                    row["status"] = "SIDE_UNPAIRED"
                    row["note"] = "no matching %s counterpart" % ("R" if row["side"] == "L" else "L")

        # Collisions inside the same namespace.
        counts = {}
        for row in rows:
            key = (row["entry"]["namespace"], row["new"].lower())
            counts[key] = counts.get(key, 0) + 1
        for row in rows:
            key = (row["entry"]["namespace"], row["new"].lower())
            if counts[key] > 1:
                row["status"] = worst(row["status"], "COLLISION")
                row["note"] = "several items would share this name (Blender will add .001)"

        issues = 0
        changes = 0
        for row in rows:
            changed = row["new"] != row["old"]
            is_issue = STATUS_RANK.get(row["status"], 0) >= STATUS_RANK["UNTRANSLATED"]
            if not changed and not is_issue:
                continue
            item = settings.items.add()
            item.use = changed
            item.kind = row["entry"]["kind"]
            item.owner = row["entry"]["owner"]
            item.sub_owner = row["entry"]["sub_owner"]
            item.old_name = row["old"]
            item.new_name = row["new"]
            item.side = row["side"]
            item.geometry_side = row["geometry"]
            item.status = row["status"]
            item.note = row["note"]
            changes += 1 if changed else 0
            issues += 1 if is_issue else 0

        settings.active_index = 0
        clear_caches()
        summary = "%d rename%s ready, %d need%s a look (%d items scanned)" % (
            changes, "" if changes == 1 else "s",
            issues, "s" if issues == 1 else "", len(entries),
        )
        settings.last_summary = summary
        self.report({"INFO"}, summary)
        return {"FINISHED"}


class NT_OT_apply(Operator):
    bl_idname = "name_translator.apply"
    bl_label = "Apply Renames"
    bl_description = "Rename everything that is still ticked in the list"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene.name_translator.items)

    def execute(self, context):
        if context.mode.startswith("EDIT"):
            self.report({"ERROR"}, "Leave Edit Mode first - renaming from Edit Mode "
                                   "would not stick")
            return {"CANCELLED"}

        settings = context.scene.name_translator
        pending = [
            item for item in settings.items
            if item.use and item.new_name and item.new_name != item.old_name
        ]
        if not pending:
            self.report({"WARNING"}, "No rows are ticked")
            return {"CANCELLED"}

        pending.sort(key=lambda item: APPLY_ORDER.index(item.kind)
                     if item.kind in APPLY_ORDER else len(APPLY_ORDER))

        bone_renames = {}
        history = []
        applied = 0
        skipped = 0

        for item in pending:
            target = resolve_target(item.kind, item.owner, item.sub_owner,
                                    item.old_name, bone_renames)
            if target is None:
                skipped += 1
                continue
            try:
                target.name = item.new_name
            except Exception:
                skipped += 1
                continue
            actual = target.name
            if item.kind == "BONE":
                bone_renames[(item.owner, item.old_name)] = actual
            history.append([item.kind, item.owner, item.sub_owner, item.old_name, actual])
            applied += 1

        settings.history = json.dumps(history)
        settings.items.clear()
        message = "Renamed %d item%s" % (applied, "" if applied == 1 else "s")
        if skipped:
            message += ", skipped %d that could not be found "
            message += "(vertex groups are renamed automatically with their bone)"
            message = message % skipped
        settings.last_summary = message
        self.report({"INFO"}, message)
        return {"FINISHED"}


class NT_OT_restore(Operator):
    bl_idname = "name_translator.restore"
    bl_label = "Undo Last Apply"
    bl_description = "Put the names back the way they were before the last Apply"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene.name_translator.history)

    def execute(self, context):
        settings = context.scene.name_translator
        try:
            history = json.loads(settings.history)
        except ValueError:
            self.report({"ERROR"}, "The stored history could not be read")
            return {"CANCELLED"}

        bone_renames = {}
        for kind, owner, sub_owner, old_name, new_name in history:
            if kind == "BONE":
                bone_renames[(owner, old_name)] = new_name

        restored = 0
        for kind, owner, sub_owner, old_name, new_name in reversed(history):
            target = resolve_target(kind, owner, sub_owner, new_name, bone_renames)
            if target is None:
                continue
            try:
                target.name = old_name
                restored += 1
            except Exception:
                continue

        settings.history = ""
        message = "Restored %d name%s" % (restored, "" if restored == 1 else "s")
        settings.last_summary = message
        self.report({"INFO"}, message)
        return {"FINISHED"}


class NT_OT_set_all(Operator):
    bl_idname = "name_translator.set_all"
    bl_label = "Toggle Rows"
    bl_options = {"REGISTER", "INTERNAL"}

    action: EnumProperty(items=[
        ("ENABLE", "All", "Tick every row"),
        ("DISABLE", "None", "Untick every row"),
        ("SAFE", "Safe only", "Tick only the rows with no warning"),
        ("INVERT", "Invert", "Invert the ticks"),
    ])

    def execute(self, context):
        settings = context.scene.name_translator
        for item in settings.items:
            changed = item.new_name != item.old_name
            if self.action == "ENABLE":
                item.use = changed
            elif self.action == "DISABLE":
                item.use = False
            elif self.action == "INVERT":
                item.use = (not item.use) and changed
            else:
                item.use = changed and not item.is_issue
        return {"FINISHED"}


class NT_OT_flip_side(Operator):
    bl_idname = "name_translator.flip_side"
    bl_label = "Flip Side"
    bl_description = "Swap this row between .L and .R"
    bl_options = {"REGISTER", "INTERNAL"}

    index: IntProperty(default=-1)

    def execute(self, context):
        settings = context.scene.name_translator
        index = self.index if self.index >= 0 else settings.active_index
        if not (0 <= index < len(settings.items)):
            return {"CANCELLED"}
        item = settings.items[index]
        base, suffix = split_index_suffix(item.new_name)
        body, side = strip_ascii_side(base)
        if not side:
            self.report({"WARNING"}, "This name has no side marker to flip")
            return {"CANCELLED"}
        flipped = "R" if side == "L" else "L"
        item.new_name = body + format_side(flipped, settings.side_style) + suffix
        item.side = flipped
        item.use = True
        if item.status == "SIDE_MISMATCH":
            item.status = "SIDE_FIXED"
            item.note = "flipped by hand"
        return {"FINISHED"}


class NT_OT_clear(Operator):
    bl_idname = "name_translator.clear"
    bl_label = "Clear List"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        settings = context.scene.name_translator
        settings.items.clear()
        settings.last_summary = ""
        return {"FINISHED"}


class NT_OT_reload_dictionary(Operator):
    bl_idname = "name_translator.reload_dictionary"
    bl_label = "Reload Dictionary"
    bl_description = "Re-read the custom dictionary file"
    bl_options = {"REGISTER"}

    def execute(self, context):
        prefs = get_prefs()
        count, message = load_user_dictionary(getattr(prefs, "dictionary_path", ""))
        self.report({"INFO"}, "%s (%d entries in total)" % (message, len(_LOOKUP)))
        return {"FINISHED"}


class NT_OT_export_untranslated(Operator):
    bl_idname = "name_translator.export_untranslated"
    bl_label = "Export Untranslated Terms"
    bl_description = ("Write every fragment the dictionary could not translate to a CSV, "
                      "ready to fill in and load back as a custom dictionary")
    bl_options = {"REGISTER"}

    filepath: StringProperty(subtype="FILE_PATH", default="untranslated.csv")
    filter_glob: StringProperty(default="*.csv", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return bool(context.scene.name_translator.items)

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        settings = context.scene.name_translator
        fragments = {}
        for item in settings.items:
            base, _ = split_index_suffix(item.old_name)
            body, _ = strip_ascii_side(base)
            _, _, unknown = tokenize(body)
            for fragment in unknown:
                fragments.setdefault(fragment, set()).add(item.old_name)

        if not fragments:
            self.report({"INFO"}, "Everything in the list was translated")
            return {"CANCELLED"}

        try:
            import csv

            with open(bpy.path.abspath(self.filepath), "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["# source", "english", "seen in"])
                for fragment in sorted(fragments):
                    examples = sorted(fragments[fragment])[:3]
                    writer.writerow([fragment, "", " | ".join(examples)])
        except Exception as error:
            self.report({"ERROR"}, "Could not write the file: %s" % error)
            return {"CANCELLED"}

        self.report({"INFO"}, "Wrote %d untranslated fragment(s) to %s"
                    % (len(fragments), self.filepath))
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# 8. Interface
# ---------------------------------------------------------------------------

class NT_UL_items(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "use", text="")

        split = row.split(factor=0.44, align=True)
        left = split.row(align=True)
        left.label(text=item.old_name, icon=KIND_ICON.get(item.kind, "DOT"))

        right = split.row(align=True)
        right.prop(item, "new_name", text="", emboss=True)
        if item.is_issue:
            right.label(text="", icon=STATUS_ICON.get(item.status, "ERROR"))
        elif item.status == "SIDE_FIXED":
            right.label(text="", icon=STATUS_ICON["SIDE_FIXED"])

    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        settings = context.scene.name_translator
        flags = []
        if settings.show_only_issues:
            flags = [self.bitflag_filter_item if item.is_issue else 0 for item in items]
        return flags, []


class NT_PT_main(Panel):
    bl_label = "Name Translator"
    bl_idname = "NT_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Translate"
    bl_order = 0

    def draw(self, context):
        settings = context.scene.name_translator
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        column = layout.column(align=True)
        column.prop(settings, "scope")
        column.prop(settings, "backend")

        if settings.backend != "DICT":
            box = layout.box()
            box.use_property_split = True
            box.prop(settings, "online_service")
            row = box.row(align=True)
            row.prop(settings, "source_lang")
            row.prop(settings, "target_lang")
            if not getattr(bpy.app, "online_access", True):
                warning = box.column()
                warning.alert = True
                warning.label(text="Allow Online Access is off", icon="ERROR")
                warning.label(text="Preferences > System > Network")


class NT_PT_targets(Panel):
    bl_label = "What to Translate"
    bl_parent_id = "NT_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Translate"

    def draw(self, context):
        settings = context.scene.name_translator
        layout = self.layout

        grid = layout.grid_flow(row_major=True, columns=2, even_columns=True)
        for name in ("do_objects", "do_data", "do_materials", "do_bones",
                     "do_vertex_groups", "do_shape_keys", "do_bone_collections",
                     "do_uv_maps", "do_color_attributes", "do_modifiers",
                     "do_constraints", "do_actions", "do_collections",
                     "do_node_groups", "do_images"):
            grid.prop(settings, name)

        layout.separator()
        layout.prop(settings, "skip_non_source")


class NT_PT_naming(Panel):
    bl_label = "Naming Style"
    bl_parent_id = "NT_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Translate"

    def draw(self, context):
        settings = context.scene.name_translator
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(settings, "style")
        layout.prop(settings, "side_style")

        preview = translate_name(
            "\u5de6\u5c0f\u81c2", settings.style, settings.side_style, "BONE")["name"]
        row = layout.row()
        row.label(text="\u5de6\u5c0f\u81c2  \u2192  %s" % preview, icon="SORTALPHA")


class NT_PT_symmetry(Panel):
    bl_label = "Left / Right Check"
    bl_parent_id = "NT_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Translate"

    def draw_header(self, context):
        self.layout.prop(context.scene.name_translator, "check_symmetry", text="")

    def draw(self, context):
        settings = context.scene.name_translator
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.active = settings.check_symmetry

        layout.prop(settings, "front_axis")
        layout.prop(settings, "auto_fix_side")
        layout.prop(settings, "detect_missing_side")

        advanced = layout.column(align=True)
        advanced.prop(settings, "use_world_space")
        advanced.prop(settings, "tolerance")

        info = layout.column(align=True)
        info.scale_y = 0.8
        info.label(text="With the default -Y facing, .L belongs at +X,")
        info.label(text="which is what Symmetrize and Flip Names expect.")


class NT_PT_results(Panel):
    bl_label = "Preview and Apply"
    bl_idname = "NT_PT_results"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Translate"
    bl_order = 1

    def draw(self, context):
        settings = context.scene.name_translator
        layout = self.layout

        row = layout.row(align=True)
        row.scale_y = 1.4
        row.operator("name_translator.scan", icon="VIEWZOOM")
        row.operator("name_translator.clear", text="", icon="X")

        if settings.last_summary:
            info = layout.row()
            info.scale_y = 0.8
            info.label(text=settings.last_summary, icon="INFO")

        if settings.items:
            selector = layout.row(align=True)
            selector.label(text="Tick:")
            selector.operator("name_translator.set_all", text="All").action = "ENABLE"
            selector.operator("name_translator.set_all", text="None").action = "DISABLE"
            selector.operator("name_translator.set_all", text="Safe").action = "SAFE"
            selector.operator("name_translator.set_all", text="Invert").action = "INVERT"

            layout.prop(settings, "show_only_issues")
            layout.template_list("NT_UL_items", "", settings, "items",
                                 settings, "active_index", rows=10)

            if 0 <= settings.active_index < len(settings.items):
                item = settings.items[settings.active_index]
                box = layout.box()
                header = box.row(align=True)
                header.label(text=KIND_LABEL.get(item.kind, item.kind),
                             icon=KIND_ICON.get(item.kind, "DOT"))
                if item.owner:
                    header.label(text="on %s" % item.owner)
                if item.note:
                    note = box.row()
                    note.alert = item.is_issue
                    note.label(text=item.note,
                               icon=STATUS_ICON.get(item.status, "INFO"))
                if item.side:
                    detail = box.row(align=True)
                    detail.label(text="Label: %s   Geometry: %s"
                                 % (item.side, item.geometry_side or "unknown"))
                    detail.operator("name_translator.flip_side",
                                    text="Flip", icon="ARROW_LEFTRIGHT").index = settings.active_index

            apply_row = layout.row()
            apply_row.scale_y = 1.5
            apply_row.operator("name_translator.apply", icon="CHECKMARK")

        footer = layout.column(align=True)
        footer.operator("name_translator.restore", icon="LOOP_BACK")
        footer.operator("name_translator.export_untranslated", icon="EXPORT")


# ---------------------------------------------------------------------------
# 9. Registration
# ---------------------------------------------------------------------------

CLASSES = (
    NT_Item,
    NT_Settings,
    NT_Preferences,
    NT_OT_scan,
    NT_OT_apply,
    NT_OT_restore,
    NT_OT_set_all,
    NT_OT_flip_side,
    NT_OT_clear,
    NT_OT_reload_dictionary,
    NT_OT_export_untranslated,
    NT_UL_items,
    NT_PT_main,
    NT_PT_targets,
    NT_PT_naming,
    NT_PT_symmetry,
    NT_PT_results,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.name_translator = bpy.props.PointerProperty(type=NT_Settings)

    prefs = get_prefs()
    if prefs is not None and prefs.dictionary_path:
        try:
            load_user_dictionary(prefs.dictionary_path)
        except Exception:
            pass


def unregister():
    if hasattr(bpy.types.Scene, "name_translator"):
        del bpy.types.Scene.name_translator
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    clear_caches()
    _ONLINE_CACHE.clear()


if __name__ == "__main__":
    register()
