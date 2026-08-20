"""Shared test scaffolding.

The add-on cannot be imported normally: it is a Blender add-on package whose
directory name is whatever the user cloned it as, and it imports ``bpy``. These
helpers load it under a stable name with the mocks in ``tests/mocks`` on the
path, so the pure-Python parts can be tested without Blender.
"""

import importlib.util
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TESTS_DIR)
MOCKS_DIR = os.path.join(TESTS_DIR, "mocks")

# The name the add-on package is imported as. Preferences and panels look
# themselves up with __package__, so tests must use this same key.
PACKAGE = "kimodo_blender_bridge"


def use_mocks():
    """Put the mock bpy / mathutils / addon_utils ahead of anything real."""
    if MOCKS_DIR not in sys.path:
        sys.path.insert(0, MOCKS_DIR)


def load_addon():
    """Import the add-on package from the repo, under a stable name."""
    use_mocks()
    if PACKAGE in sys.modules:
        return sys.modules[PACKAGE]
    spec = importlib.util.spec_from_file_location(
        PACKAGE,
        os.path.join(REPO_DIR, "__init__.py"),
        submodule_search_locations=[REPO_DIR],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = module
    spec.loader.exec_module(module)
    return module


def load_standalone(name):
    """Import a bridge-side module (ardy_bvh, …) that never imports bpy.

    These run under the model venv, not Blender, so they are loaded directly
    rather than as part of the add-on package.
    """
    path = os.path.join(REPO_DIR, name + ".py")
    spec = importlib.util.spec_from_file_location(f"_standalone_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Checker:
    """Tiny assertion recorder — keeps output readable without pytest."""

    def __init__(self, title):
        self.title = title
        self.failures = []
        self.count = 0
        print(f"\n=== {title} ===")

    def __call__(self, cond, msg):
        self.count += 1
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            self.failures.append(msg)
        return bool(cond)

    def done(self):
        if self.failures:
            print(f"  -> {len(self.failures)}/{self.count} FAILED")
            return False
        print(f"  -> {self.count} checks passed")
        return True


class Skip(Exception):
    """Raised by a test that cannot run here (missing optional dependency)."""
