"""Guard the ARDY text-encoder install against shipping a half-built encoder.

The McGill LLM2Vec repos hold only a LoRA adapter; the base Llama-3 weights come
from a separate gated repo. An install that fetched the adapters and stopped
looked like a success and then failed at bridge startup with an OSError from
inside transformers — the encoder directory existed, had a config and tokenizer,
and no weights. These tests pin down the check that now stands between that
state and the install sentinel.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import Checker, PACKAGE, load_addon

load_addon()
A = sys.modules[PACKAGE + ".ardy_setup"]

check = Checker("ardy_setup: encoder weight verification")


def _dir_with(*names):
    """A throwaway directory containing empty files with the given names."""
    path = tempfile.mkdtemp(prefix="ardy_enc_")
    for name in names:
        open(os.path.join(path, name), "w").close()
    return path


def _rejects(path):
    """True when _require_weights refuses this directory."""
    try:
        A._require_weights(path)
        return False
    except RuntimeError:
        return True


# The exact state the broken install left on disk.
adapter_only = _dir_with(
    "adapter_config.json", "adapter_model.safetensors",
    "config.json", "tokenizer.json", "tokenizer_config.json",
)
check(_rejects(adapter_only), "adapter-only directory is rejected")

# adapter_model.safetensors must not be mistaken for a checkpoint just because
# it ends in .safetensors — that near-miss is what makes this failure subtle.
message = ""
try:
    A._require_weights(adapter_only)
except RuntimeError as exc:
    message = str(exc)
check("adapter_model.safetensors" in message,
      "  rejection names what it did find, so the log is diagnosable")
check(A.BASE_MODEL_REPO in message,
      "  rejection names the repo whose download is missing")

# The layouts that should pass.
check(not _rejects(_dir_with("adapter_model.safetensors",
                             "model-00001-of-00004.safetensors",
                             "model-00002-of-00004.safetensors",
                             "model.safetensors.index.json")),
      "sharded checkpoint beside the adapter is accepted")
check(not _rejects(_dir_with("adapter_model.safetensors", "model.safetensors")),
      "single-file checkpoint is accepted")
check(not _rejects(_dir_with("pytorch_model.bin.index.json")),
      "legacy .bin checkpoint is accepted")

check(_rejects(os.path.join(adapter_only, "does-not-exist")),
      "unreadable directory is rejected rather than passing silently")

# The base-weight patterns must not reach into subfolders: Meta ships a second
# full copy of the weights under original/, and a loose "*.safetensors" would
# match it and double an already 16 GB download.
check(all(not p.startswith("*") for p in A.BASE_MODEL_PATTERNS),
      "base-weight patterns are anchored, not leading-wildcard")
check(any("index.json" in p for p in A.BASE_MODEL_PATTERNS),
      "base-weight patterns include the shard index")


def run():
    return check.done()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
