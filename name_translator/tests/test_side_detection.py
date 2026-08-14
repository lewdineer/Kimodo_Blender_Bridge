# SPDX-License-Identifier: GPL-3.0-or-later
"""Left / right detection tests for the Name Translator add-on.

Runs outside Blender against the stub modules in tests/stubs.

    python3 name_translator/tests/test_side_detection.py
"""

import importlib.util
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON = os.path.join(os.path.dirname(HERE), "__init__.py")
sys.path.insert(0, os.path.join(HERE, "stubs"))

_spec = importlib.util.spec_from_file_location("name_translator_under_test", ADDON)
nt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nt)


def scan_name(name, side_style="DOT", style="PASCAL", skip_non_source=True):
    """Reproduce the naming half of NT_OT_scan.execute for one name.

    The geometry cross-check is left out on purpose - these tests are about
    what the *name* says, not about where the bone sits.
    """
    result = nt.translate_name(name, style, side_style)
    base, suffix = nt.split_index_suffix(name)
    final_side = result["side"]

    if skip_non_source and not nt.has_source_script(base):
        body, original_side, position = nt.find_ascii_side(base)
        if side_style == "NONE":
            return name
        if final_side and (final_side != original_side or position == "MIDDLE"):
            return body + nt.format_side(final_side, side_style) + suffix
        return name
    return result["body"] + nt.format_side(final_side, side_style) + suffix


class FindAsciiSide(unittest.TestCase):
    def test_marker_at_the_end(self):
        self.assertEqual(nt.find_ascii_side("Hand.L"), ("Hand", "L", "TAIL"))
        self.assertEqual(nt.find_ascii_side("Skn_Wing_R"), ("Skn_Wing", "R", "TAIL"))
        self.assertEqual(nt.find_ascii_side("Arm-right"), ("Arm", "R", "TAIL"))
        self.assertEqual(nt.find_ascii_side("Arm Left"), ("Arm", "L", "TAIL"))

    def test_marker_at_the_start(self):
        self.assertEqual(nt.find_ascii_side("L_Hand"), ("Hand", "L", "HEAD"))
        self.assertEqual(nt.find_ascii_side("Right.Arm"), ("Arm", "R", "HEAD"))

    def test_marker_in_the_middle(self):
        # The case this release is about.
        self.assertEqual(nt.find_ascii_side("Skn_R_WingC_ZF_01"),
                         ("Skn_WingC_ZF_01", "R", "MIDDLE"))
        self.assertEqual(nt.find_ascii_side("Skn_L_WingC_ZF_01"),
                         ("Skn_WingC_ZF_01", "L", "MIDDLE"))
        self.assertEqual(nt.find_ascii_side("Skn_Right_Wing_01"),
                         ("Skn_Wing_01", "R", "MIDDLE"))

    def test_middle_marker_leaves_a_single_separator(self):
        for name, expected in (
            ("Skn_R_Wing", "Skn_Wing"),
            ("Skn-R-Wing", "Skn-Wing"),
            ("Skn.R.Wing", "Skn.Wing"),
            ("Skn R Wing", "Skn Wing"),
        ):
            self.assertEqual(nt.find_ascii_side(name)[0], expected, name)

    def test_case_is_ignored(self):
        self.assertEqual(nt.find_ascii_side("skn_r_wingc_zf_01"),
                         ("skn_wingc_zf_01", "R", "MIDDLE"))
        self.assertEqual(nt.find_ascii_side("bone_l"), ("bone", "L", "TAIL"))

    def test_the_end_is_searched_first(self):
        # An infix marker must not steal the row from a trailing one.
        self.assertEqual(nt.find_ascii_side("L_Hand_R"), ("L_Hand", "R", "TAIL"))
        self.assertEqual(nt.find_ascii_side("Skn_R_Wing_L"), ("Skn_R_Wing", "L", "TAIL"))

    def test_only_the_first_infix_marker_is_removed(self):
        self.assertEqual(nt.find_ascii_side("Skn_R_Wing_L_01"),
                         ("Skn_Wing_L_01", "R", "MIDDLE"))

    def test_letters_glued_to_a_word_are_not_markers(self):
        for name in ("SknRWing", "eyeR_ctrl", "Ctrl_Roll", "Bone_LR_01",
                     "Skn_Rig_Wing_01", "Lamp_Reflector"):
            self.assertEqual(nt.find_ascii_side(name), (name, None, ""), name)

    def test_a_bare_marker_keeps_its_body(self):
        # "R" on its own would leave nothing to name the item after, so it is
        # not treated as a strippable marker here.
        self.assertEqual(nt.find_ascii_side("R"), ("R", None, ""))

    def test_strip_ascii_side_keeps_its_two_value_signature(self):
        self.assertEqual(nt.strip_ascii_side("Skn_R_Wing"), ("Skn_Wing", "R"))
        self.assertEqual(nt.strip_ascii_side("Skn_Wing"), ("Skn_Wing", None))


class TranslateName(unittest.TestCase):
    def test_infix_marker_becomes_a_suffix(self):
        self.assertEqual(nt.translate_name("Skn_R_WingC_ZF_01")["name"], "SknWingCZF01.R")
        self.assertEqual(nt.translate_name("Skn_L_WingC_ZF_01")["name"], "SknWingCZF01.L")

    def test_side_is_reported(self):
        self.assertEqual(nt.translate_name("Skn_R_WingC_ZF_01")["side"], "R")
        self.assertEqual(nt.translate_name("Skn_L_WingC_ZF_01")["side"], "L")
        self.assertEqual(nt.translate_name("Skn_Wing_01")["side"], "")

    def test_raw_style_does_not_leave_a_double_separator(self):
        self.assertEqual(nt.translate_name("Skn_R_WingC_ZF_01", style="RAW")["name"],
                         "Skn_WingC_ZF_01.R")

    def test_mixed_markers_are_flagged(self):
        self.assertTrue(nt.translate_name("Skn_R_Wing_L_01")["conflict"])
        self.assertFalse(nt.translate_name("Skn_R_Wing_01")["conflict"])

    def test_chinese_markers_still_work(self):
        self.assertEqual(nt.translate_name("左手")["name"], "Hand.L")
        self.assertEqual(nt.translate_name("右手")["name"], "Hand.R")


class ScanRow(unittest.TestCase):
    """End-to-end behaviour of an already-english name during a scan."""

    def test_infix_marker_is_moved_to_the_end(self):
        self.assertEqual(scan_name("Skn_R_WingC_ZF_01"), "Skn_WingC_ZF_01.R")
        self.assertEqual(scan_name("Skn_L_WingC_ZF_01"), "Skn_WingC_ZF_01.L")

    def test_infix_marker_follows_the_chosen_suffix_style(self):
        self.assertEqual(scan_name("Skn_R_WingC_ZF_01", side_style="UNDERSCORE"),
                         "Skn_WingC_ZF_01_R")
        self.assertEqual(scan_name("Skn_R_WingC_ZF_01", side_style="DOT_WORD"),
                         "Skn_WingC_ZF_01.Right")

    def test_duplicate_suffix_is_not_appended(self):
        # The 1.0.0 bug: the infix marker was detected but never removed, so the
        # row read "Skn_R_WingC_ZF_01.R".
        self.assertNotIn("_R_", scan_name("Skn_R_WingC_ZF_01"))
        self.assertEqual(scan_name("bone_l"), "bone_l")
        self.assertEqual(scan_name("bone_r"), "bone_r")

    def test_the_duplicate_suffix_survives_a_numbered_name(self):
        self.assertEqual(scan_name("Skn_R_Wing.001"), "Skn_Wing.R.001")

    def test_names_blender_already_understands_are_left_alone(self):
        for name in ("Hand.L", "Skn_Wing_R", "L_Hand", "Hand_L.001",
                     "eyeR_ctrl", "Skn_Rig_Wing"):
            self.assertEqual(scan_name(name), name, name)

    def test_side_as_a_word_is_kept_when_suffixes_are_off(self):
        self.assertEqual(scan_name("Skn_R_WingC_ZF_01", side_style="NONE"),
                         "Skn_R_WingC_ZF_01")

    def test_a_pair_lines_up(self):
        left = scan_name("Skn_L_WingC_ZF_01")
        right = scan_name("Skn_R_WingC_ZF_01")
        self.assertEqual(nt.strip_ascii_side(left)[0], nt.strip_ascii_side(right)[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
