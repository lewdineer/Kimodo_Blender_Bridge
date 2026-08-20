#!/usr/bin/env python
"""
ARDY Blender Bridge Server

The ARDY counterpart to bridge_server.py: a persistent process that runs under
the managed ARDY venv, loads the model once, and answers JSON generation
requests from stdin.  It speaks **exactly the same protocol** as the Kimodo
bridge so subprocess_client.py does not care which backend is running.

stdin  -> one JSON line per request:   {"cmd": "generate"|"generate_multi"|"ping"|"quit", ...}
stdout -> one JSON line per message:   {"status": "loading"|"ready"|"progress"|"done"|"error", ...}

stderr is left alone (ARDY / PyTorch logging goes there).

Differences from Kimodo that this file absorbs so the add-on does not have to:

* ARDY has no BVH exporter, so output goes through ardy_bvh.save_ardy_bvh().
* ARDY's model call is three steps (sample -> motion_rep.inverse -> optional
  post-processing) rather than one, and constraints become
  observed_motion / motion_mask tensors instead of a constraint_lst kwarg.
* ARDY has no multi-prompt call.  generate_multi chains segments through
  init_history_sequence, accumulating the normalized motion tensor exactly the
  way scripts/interactive_demo/generation.py does.
* ARDY models are 20 FPS (Core) or 25 FPS (G1), never 30.  The real value is
  reported in the "ready" message and the add-on reads it from there.
"""

import json
import os
import sys
import tempfile
import traceback

# Match bridge_server.py: a non-UTF-8 Windows console must not be able to kill
# the process before a real error can be reported over the pipe.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ardy_bvh.py sits next to this file; when launched by absolute path the
# directory is already sys.path[0], but be explicit so an odd launcher works.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _out(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Model context
# ---------------------------------------------------------------------------

class _Ctx:
    """Everything the request handlers need, resolved once at startup."""

    def __init__(self, model, device: str, postprocess: bool):
        self.model = model
        self.device = device
        self.postprocess = postprocess

        self.fps = float(model.motion_rep.fps)
        self.patch = int(model.num_frames_per_token)
        self.gen_horizon_len = int(model.gen_horizon_len)
        self.num_base_steps = int(model.diffusion.num_base_steps)

        # ARDY is trained on windows of at most 10 s. Without cropping, the
        # attention window grows with the output and long generations degrade
        # into jitter, so every rollout keeps history within this budget.
        # Same arithmetic as scripts/generate.py::_default_history_frames.
        self.max_window_len = (int(10 * self.fps) // self.patch) * self.patch
        self.history_budget = max(
            self.patch,
            ((self.max_window_len - self.gen_horizon_len) // self.patch) * self.patch,
        )
        # Longest new stretch that still fits alongside a full history.
        self.chunk_max = max(
            self.gen_horizon_len, self.max_window_len - self.history_budget
        )

    def clamp_steps(self, requested) -> int:
        """The diffusion schedule can only be subsampled.

        Asking for more steps than num_base_steps indexes past the timestep map
        and trips a device-side assert, so clamp instead of crashing.
        """
        if requested is None:
            return self.num_base_steps
        return max(1, min(int(requested), self.num_base_steps))

    def align_history(self, requested) -> int:
        """Round a requested history length to a usable multiple of the patch."""
        if not requested:
            return self.history_budget
        aligned = (int(requested) // self.patch) * self.patch
        return max(self.patch, min(aligned, self.history_budget))


def _cfg_weight(req: dict):
    """(text_weight, constraint_weight) from a request, defaulting to ARDY's 2.0/2.0."""
    text_w = float(req.get("cfg_text_weight", 2.0))
    con_w = req.get("cfg_constraint_weight")
    return (text_w, float(con_w)) if con_w is not None else (text_w, text_w)


def _normalise_prompt(text: str) -> str:
    text = (text or "").strip()
    if text and not text.endswith("."):
        text += "."
    return text


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

def _load_constraints(constraints_json, skeleton):
    """Parse the add-on's constraint JSON into ARDY constraint objects.

    The JSON the add-on emits (constraints.py::build_constraints_json) is
    already in ARDY's format: the same type strings, and from_dict() accepts
    'smooth_root_2d' as an alias for 'root_2d'.  Nothing is translated here.
    """
    if not constraints_json:
        return []
    from ardy.constraints import load_constraints_lst

    try:
        data = json.loads(constraints_json)
    except (TypeError, ValueError) as exc:
        _out({"status": "progress",
              "message": f"Warning: constraints skipped (bad JSON: {exc})"})
        return []
    try:
        return load_constraints_lst(data, skeleton)
    except Exception as exc:
        _out({"status": "progress",
              "message": f"Warning: constraints skipped ({exc})"})
        return []


def _build_conditions(ctx, constraint_lst, total_frames: int):
    """Full-length (observed_motion, motion_mask), or (None, None)."""
    if not constraint_lst:
        return None, None
    import torch

    lengths = torch.tensor([total_frames], device=ctx.device)
    return ctx.model.motion_rep.create_conditions_from_constraints_batched(
        constraint_lst, lengths, to_normalize=True, device=ctx.device,
    )


def _slice_conditions(observed, mask, start: int, length: int, history_len: int):
    """Window the conditions onto one rollout call.

    The call sees ``history_len`` frames of already-generated motion followed by
    the new stretch, so its frame 0 is absolute frame ``start``.  History frames
    must not carry constraints — they are already fixed — which is the same
    zeroing the interactive demo does.
    """
    if observed is None or mask is None:
        return None, None
    import torch

    total = observed.shape[1]
    if start >= total:
        return None, None

    obs = observed[:, start:start + length]
    msk = mask[:, start:start + length]
    if obs.shape[1] < length:                      # pad the tail past the end
        pad = length - obs.shape[1]
        obs = torch.cat([obs, torch.zeros(obs.shape[0], pad, obs.shape[2],
                                          device=obs.device, dtype=obs.dtype)], dim=1)
        msk = torch.cat([msk, torch.zeros(msk.shape[0], pad, msk.shape[2],
                                          device=msk.device, dtype=msk.dtype)], dim=1)
    if history_len:
        obs, msk = obs.clone(), msk.clone()
        obs[:, :history_len] = 0.0
        msk[:, :history_len] = 0.0
    if not bool(msk.any()):
        return None, None
    return obs, msk


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def _rollout(ctx, segments, diffusion_steps, cfg_weight, history_frames,
             observed=None, mask=None):
    """Generate one continuous motion from a list of (text, num_frames) pairs.

    The first segment is a plain full-length call — identical to
    scripts/generate.py.  Every segment after it is rolled out in window-sized
    chunks that carry the accumulated motion in as init_history_sequence, which
    is how the interactive demo keeps an autoregressive stream continuous.

    Returns the decoded output dict, with every array covering the whole motion.
    """
    import torch
    from ardy.motion_rep.tools import length_to_mask

    total_frames = sum(n for _, n in segments)
    motion_tensor = None            # accumulated *normalized* motion rep
    decoded_parts: "list[dict]" = []
    produced = 0

    def _call(text, num_frames, init_history):
        """One model call. Returns (normalized_samples, decoded_dict)."""
        history_len = 0 if init_history is None else int(init_history.shape[1])
        call_len = num_frames + history_len

        lengths = torch.tensor([call_len], device=ctx.device)
        pad_mask = length_to_mask(lengths)
        obs, msk = _slice_conditions(
            observed, mask, produced - history_len, call_len, history_len,
        )

        kwargs = dict(
            num_denoising_steps=diffusion_steps,
            pad_mask=pad_mask,
            motion_mask=msk,
            observed_motion=obs,
            cfg_weight=cfg_weight,
        )
        if init_history is None:
            # facing +Z, and let the model crop its own history as it rolls out
            kwargs["first_heading_angle"] = torch.zeros(1, device=ctx.device)
            kwargs["crop_history_length"] = history_frames
        else:
            # __call__ rejects both at once, so the caller keeps chunks short
            # enough that the trained window is never exceeded.
            kwargs["first_heading_angle"] = None
            kwargs["init_history_sequence"] = init_history

        with torch.no_grad():
            samples = ctx.model([text], call_len, **kwargs)
            decoded = ctx.model.motion_rep.inverse(samples, is_normalized=True)
        return samples, decoded, history_len

    for seg_idx, (text, num_frames) in enumerate(segments):
        remaining = num_frames
        while remaining > 0:
            if motion_tensor is None:
                take = remaining                      # first call: whole segment
                init_history = None
            else:
                take = min(remaining, ctx.chunk_max)
                hist = min(history_frames, motion_tensor.shape[1])
                hist = max(ctx.patch, (hist // ctx.patch) * ctx.patch)
                init_history = motion_tensor[:, -hist:]

            _out({"status": "progress",
                  "message": (f"Segment {seg_idx + 1}/{len(segments)} — "
                              f"{produced}/{total_frames} frames "
                              f"({diffusion_steps} steps)…")})

            samples, decoded, history_len = _call(text, take, init_history)

            # Drop the history prefix (motion we already have) and trim the
            # tail: a rollout is rounded up to a whole generation horizon, so
            # the model returns >= the frames asked for. Timeline mode maps
            # segments onto exact frame ranges, so the extra must not survive.
            stop = history_len + take
            new_samples = samples[:, history_len:stop]
            decoded_parts.append(
                {k: (v[:, history_len:stop] if hasattr(v, "shape") and v.ndim >= 2 else v)
                 for k, v in decoded.items()}
            )
            motion_tensor = (
                new_samples if motion_tensor is None
                else torch.cat([motion_tensor, new_samples], dim=1)
            )
            got = int(new_samples.shape[1])
            if got == 0:
                raise RuntimeError(
                    f"model returned no new frames for segment {seg_idx + 1} "
                    f"(asked for {take}) — aborting rather than looping forever"
                )
            produced += got
            remaining -= got

    # Stitch the decoded pieces back into one motion.
    output: "dict" = {}
    for key in decoded_parts[0]:
        vals = [p[key] for p in decoded_parts if p.get(key) is not None]
        if not vals:
            continue
        first = vals[0]
        if hasattr(first, "shape") and getattr(first, "ndim", 0) >= 2:
            output[key] = torch.cat(vals, dim=1)
        else:
            output[key] = first
    return output


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _postprocess(ctx, output, constraint_lst):
    """Run ARDY's foot-skate correction in place, if it is available."""
    if not ctx.postprocess:
        return output
    try:
        from ardy.postprocess import post_process_motion
    except Exception as exc:
        _out({"status": "progress",
              "message": f"Post-processing unavailable ({exc}) — continuing without."})
        return output
    try:
        corrected = post_process_motion(
            output["local_rot_mats"],
            output["root_positions"],
            output["foot_contacts"],
            ctx.model.skeleton,
            constraint_lst=constraint_lst or None,
        )
        output.update(corrected)
    except Exception as exc:
        # Motion correction is a quality pass, never a hard requirement — a
        # failure here must not lose an otherwise good generation.
        _out({"status": "progress",
              "message": f"Warning: post-processing failed ({exc}) — using raw motion."})
    return output


def _save_output(ctx, output, output_format: str, standard_tpose: bool,
                 prefix: str = "ardy_") -> str:
    """Write the motion to a temp file and return its path."""
    import numpy as np
    from ardy.skeleton import SOMASkeleton30
    from ardy.tools import to_numpy

    import ardy_bvh

    skeleton = ctx.model.skeleton
    # SOMA models are emitted on the 77-joint rig, matching Kimodo's BVH. No
    # SOMA checkpoint is released yet, so today this branch never runs.
    if isinstance(skeleton, SOMASkeleton30):
        output = skeleton.output_to_SOMASkeleton77(output)
        skeleton = skeleton.somaskel77.to(ctx.device)

    if standard_tpose:
        if ardy_bvh.supports_standard_tpose(skeleton):
            local, _ = skeleton.to_standard_tpose(output["local_rot_mats"])
            output = dict(output)
            output["local_rot_mats"] = local
        else:
            _out({"status": "progress",
                  "message": (f"Standard T-pose is not available for "
                              f"'{getattr(skeleton, 'name', '?')}' — "
                              f"exporting in the model's own rest pose.")})

    output = to_numpy(output)

    if output_format == "npz":
        fd, path = tempfile.mkstemp(suffix=".npz", prefix=prefix)
        os.close(fd)
        arrays = {k: np.asarray(v) for k, v in output.items()
                  if hasattr(v, "shape")}
        arrays["fps"] = np.asarray(ctx.fps)
        np.savez(path, **arrays)
        return path

    fd, path = tempfile.mkstemp(suffix=".bvh", prefix=prefix)
    os.close(fd)
    local = output["local_rot_mats"]
    root = output["root_positions"]
    if getattr(local, "ndim", 0) == 5:      # [B, T, J, 3, 3] -> first sample
        local, root = local[0], root[0]
    return ardy_bvh.save_ardy_bvh(path, local, root, skeleton, ctx.fps)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _generate(req: dict, ctx: _Ctx) -> None:
    from ardy.tools import seed_everything

    text = _normalise_prompt(req.get("prompt", "A person walks forward."))
    duration = float(req.get("duration", 5.0))
    num_frames = max(ctx.patch, int(round(duration * ctx.fps)))
    seed = req.get("seed")
    if seed is not None:
        seed_everything(int(seed))

    steps = ctx.clamp_steps(req.get("diffusion_steps"))
    history = ctx.align_history(req.get("history_frames"))
    constraint_lst = _load_constraints(req.get("constraints_json"),
                                       ctx.model.skeleton)
    observed, mask = _build_conditions(ctx, constraint_lst, num_frames)

    output = _rollout(ctx, [(text, num_frames)], steps, _cfg_weight(req),
                      history, observed, mask)
    output = _postprocess(ctx, output, constraint_lst)

    _out({"status": "progress", "message": "Saving output file…"})
    path = _save_output(ctx, output, req.get("output_format", "bvh"),
                        bool(req.get("bvh_standard_tpose", False)))
    _out({"status": "done", "path": path})


def _generate_multi(req: dict, ctx: _Ctx) -> None:
    from ardy.tools import seed_everything

    prompts = req.get("prompts") or []
    durations = req.get("durations") or []
    if not prompts or len(prompts) != len(durations):
        _out({"status": "error",
              "message": ("generate_multi requires non-empty 'prompts' and "
                          "'durations' lists of equal length.")})
        return

    seed = req.get("seed")
    if seed is not None:
        seed_everything(int(seed))

    segments = [
        (_normalise_prompt(p), max(ctx.patch, int(round(float(d) * ctx.fps))))
        for p, d in zip(prompts, durations)
    ]
    total_frames = sum(n for _, n in segments)

    steps = ctx.clamp_steps(req.get("diffusion_steps"))
    history = ctx.align_history(req.get("history_frames"))
    constraint_lst = _load_constraints(req.get("constraints_json"),
                                       ctx.model.skeleton)
    observed, mask = _build_conditions(ctx, constraint_lst, total_frames)

    output = _rollout(ctx, segments, steps, _cfg_weight(req), history,
                      observed, mask)
    output = _postprocess(ctx, output, constraint_lst)

    _out({"status": "progress", "message": "Saving combined output file…"})
    path = _save_output(ctx, output, req.get("output_format", "bvh"),
                        bool(req.get("bvh_standard_tpose", False)),
                        prefix="ardy_multi_")
    _out({"status": "done", "path": path})


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ARDY Blender Bridge Server")
    parser.add_argument("--model", default="core",
                        help="ARDY model nickname (core/core8/g1/…) or full name")
    parser.add_argument("--device", default=None,
                        help="Compute device override (e.g. cuda:0, cpu)")
    parser.add_argument("--text-encoder-device", default=None,
                        help="Run the LLM2Vec encoder here instead (cpu frees ~14 GB VRAM)")
    parser.add_argument("--checkpoints-dir", default=None,
                        help="Local folder holding released ARDY model folders")
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Skip foot-skate correction (needed when the "
                             "motion_correction extension was not built)")
    args = parser.parse_args()

    # Honoured by ardy/model/llm2vec/llm2vec_wrapper.py — no source patching
    # needed, unlike Kimodo's wrapper.
    if args.text_encoder_device:
        os.environ["TEXT_ENCODER_DEVICE"] = args.text_encoder_device

    _out({"status": "loading", "message": "Importing ARDY…"})
    try:
        import torch
        from ardy.model import load_model
        from ardy.model.registry import resolve_model_name
    except ImportError as exc:
        _out({"status": "error",
              "message": (f"ARDY not found in this Python environment: {exc}\n"
                          "Point the add-on at the ARDY venv's Python, or run "
                          "the ARDY installer from the Connection panel.")})
        sys.exit(1)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        resolved = resolve_model_name(args.model,
                                      checkpoints_dir=args.checkpoints_dir)
    except ValueError as exc:
        _out({"status": "error", "message": str(exc)})
        sys.exit(1)

    _out({"status": "loading", "message": f"Loading {resolved} on {device}…"})
    try:
        # text_encoder_mode="local" keeps everything in-process: "auto" would
        # first probe for a Gradio text-encoder service on localhost, which the
        # add-on never runs, and wait out the connection attempt every launch.
        model = load_model(
            resolved,
            device=device,
            text_encoder_mode="local",
            checkpoints_dir=args.checkpoints_dir,
        )
    except Exception as exc:
        _out({"status": "error",
              "message": f"Model load failed: {exc}\n{traceback.format_exc()}"})
        sys.exit(1)

    ctx = _Ctx(model, device, postprocess=not args.no_postprocess)

    import ardy_bvh
    _out({
        "status": "ready",
        "model": resolved,
        "device": device,
        "fps": ctx.fps,
        "skeleton": getattr(model.skeleton, "name", "?"),
        "supports_tpose": ardy_bvh.supports_standard_tpose(model.skeleton),
        "max_diffusion_steps": ctx.num_base_steps,
        "postprocess": ctx.postprocess,
    })

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError as exc:
            _out({"status": "error", "message": f"Bad JSON: {exc}"})
            continue

        cmd = req.get("cmd", "")
        if cmd == "ping":
            _out({"status": "pong"})
        elif cmd == "quit":
            _out({"status": "bye"})
            break
        elif cmd in ("generate", "generate_multi"):
            handler = _generate if cmd == "generate" else _generate_multi
            try:
                handler(req, ctx)
            except Exception as exc:
                _out({"status": "error",
                      "message": str(exc),
                      "traceback": traceback.format_exc()})
        else:
            _out({"status": "error", "message": f"Unknown cmd: {cmd!r}"})


if __name__ == "__main__":
    main()
