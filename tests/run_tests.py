#!/usr/bin/env python3
"""Run every test module in this folder.

    python3 tests/run_tests.py

Each module is a plain script with a run() -> bool. Modules that need an
optional dependency (numpy/scipy for the BVH writer, torch for the bridge)
raise Skip at import time and are reported as skipped rather than failing, so
the suite is still useful on a machine without the model stack installed.

Nothing here needs Blender: bpy, mathutils and addon_utils are mocked in
tests/mocks, and the ARDY model is mocked too.
"""

import importlib.util
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TESTS_DIR)

from _harness import Skip   # noqa: E402

MODULES = [
    "test_ardy_bvh",
    "test_ardy_bridge",
    "test_addon_registration",
    "test_ardy_setup",
    "test_armature_reuse",
    "test_panels",
]


def main() -> int:
    passed, failed, skipped = [], [], []

    for name in MODULES:
        path = os.path.join(TESTS_DIR, name + ".py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Skip as exc:
            print(f"\n=== {name} ===\n  SKIP  {exc}")
            skipped.append(name)
            continue
        except Exception as exc:                      # a crash is a failure
            import traceback
            traceback.print_exc()
            print(f"  ERROR {name}: {exc}")
            failed.append(name)
            continue
        (passed if module.run() else failed).append(name)

    print("\n" + "=" * 60)
    print(f"passed {len(passed)}   failed {len(failed)}   skipped {len(skipped)}")
    if skipped:
        print("skipped: " + ", ".join(skipped))
    if failed:
        print("FAILED:  " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
