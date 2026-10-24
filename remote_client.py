# SPDX-License-Identifier: GPL-2.0-or-later
"""
Kimodo Blender Bridge — remote MMCP client.

Talks to a motionmcp-kimodo server (GET /capabilities, POST /generate)
using only the standard library, so it runs inside Blender's Python.

The GLB unpacker follows motionmcp.client.glb (Apache-2.0): a server may
answer with model/gltf-binary, and we turn that into the same glTF JSON
document the JSON path returns.
"""

import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


# Generation can sit on a cold model load plus diffusion.
_GENERATE_TIMEOUT = 600.0
_CONNECT_TIMEOUT = 30.0
_POLL_TIMEOUT = 30.0


class RemoteError(RuntimeError):
    def __init__(self, message, code="transport"):
        super().__init__(message)
        self.code = code


_lock = threading.Lock()
_session = None          # {"url", "capabilities", "model_id", "fps"}
_busy = False
_cancel = threading.Event()


def request_cancel() -> None:
    """Stop waiting for the in-flight job. The server keeps computing."""
    _cancel.set()


def is_busy() -> bool:
    return _busy


def is_ready() -> bool:
    return _session is not None


def session():
    """The cached connect result, or None."""
    return _session


def capabilities():
    info = _session
    return None if info is None else info.get("capabilities")


def disconnect() -> None:
    """Drop the cached session. Does not ask the server to unload the model."""
    global _session
    request_cancel()
    with _lock:
        _session = None


def normalize_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if not url:
        raise RemoteError("Enter a server URL.")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def connect(url: str, progress_callback=None):
    """
    GET /capabilities and cache the model, fps, and canonical skeleton.
    Returns (success, status_message). Safe to call from a background thread.
    """
    global _session
    _cancel.clear()
    try:
        base = normalize_url(url)
    except RemoteError as exc:
        return False, str(exc)

    if progress_callback:
        progress_callback(f"Connecting to {base}…")

    try:
        caps = _get_json(base + "/capabilities", timeout=_CONNECT_TIMEOUT)
    except RemoteError as exc:
        return False, str(exc)

    try:
        model = _pick_model(caps)
    except RemoteError as exc:
        return False, str(exc)

    with _lock:
        _session = {
            "url": base,
            "capabilities": caps,
            "model_id": model["id"],
            "fps": float(model["fps"]),
        }
    message = (
        f"Ready — {model['id']} at {base} ({model['fps']} fps)"
    )
    return True, message


def build_generate_request(
    caps: dict,
    prompts: list,
    durations: list,
    seeds=None,
    constraints=None,
    transition_frames: int = 5,
    diffusion_steps: int = 100,
) -> dict:
    """
    Build a POST /generate body. ``caps`` is the /capabilities document.
    The skeleton is copied verbatim — the server rejects anything else.
    Per-segment seeds are attached only when the model advertises them.
    """
    if not prompts or not durations or len(prompts) != len(durations):
        raise RemoteError("Need one duration for each prompt.")

    model = _pick_model(caps)
    fps = float(model["fps"])
    supports_segment_seed = bool(model.get("supports_segment_seed"))
    seed_list = list(seeds) if seeds else []

    if (
        seed_list
        and len(set(int(s) for s in seed_list)) > 1
        and not supports_segment_seed
    ):
        print(
            "[Kimodo] Server does not support per-segment seeds; "
            "using the first seed for the whole clip.",
            flush=True,
        )

    segments = []
    for i, (prompt, duration) in enumerate(zip(prompts, durations)):
        text = (prompt or "").strip()
        if text and not text.endswith("."):
            text += "."
        if not text:
            text = "A person walks forward."
        seg = {
            "type": "text",
            "prompt": text,
            "duration_frames": max(1, int(round(float(duration) * fps))),
        }
        if (
            supports_segment_seed
            and i < len(seed_list)
            and seed_list[i] is not None
            and int(seed_list[i]) >= 0
        ):
            seg["seed"] = int(seed_list[i])
        segments.append(seg)

    options = {
        "diffusion_steps": int(diffusion_steps),
        "num_samples": 1,
        "post_processing": True,
        "transition_frames": max(0, min(60, int(transition_frames))),
    }
    if seed_list and seed_list[0] is not None and int(seed_list[0]) >= 0:
        options["seed"] = int(seed_list[0])

    return {
        "protocol_version": caps.get("protocol_version") or "1.4",
        "model": model["id"],
        "skeleton": model["canonical_skeleton"],
        "segments": segments,
        "constraints": list(constraints or []),
        "options": options,
    }


def generate_to_file(body: dict, progress_callback=None, timeout: float = _GENERATE_TIMEOUT):
    """
    POST /generate and write the glTF document to a temp file.
    Returns (success, file_path_or_error). Blocks; call from a background thread.
    """
    global _busy
    info = _session
    if info is None:
        return False, "Not connected — set the server URL and click Connect."

    with _lock:
        if _busy:
            return False, (
                "Kimodo is still finishing the previous request — "
                "wait for it to complete and try again."
            )
        _busy = True
        _cancel.clear()

    path = None
    try:
        if progress_callback:
            progress_callback("Sending request…")
        doc = _generate(info["url"], body, timeout, progress_callback)
        if _cancel.is_set():
            return False, "Cancelled by user"
        fd, path = tempfile.mkstemp(suffix=".gltf", prefix="kimodo_")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(doc, handle)
        if _cancel.is_set():
            _unlink(path)
            return False, "Cancelled by user"
        return True, path
    except RemoteError as exc:
        _unlink(path)
        return False, str(exc)
    except Exception as exc:
        _unlink(path)
        return False, f"Remote generation failed: {exc}"
    finally:
        with _lock:
            _busy = False


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _pick_model(caps: dict) -> dict:
    if not isinstance(caps, dict):
        raise RemoteError(
            "Not an MMCP Kimodo server (capabilities was not a JSON object).",
            code="bad_response",
        )
    models = caps.get("models")
    if not isinstance(models, list) or not models or not isinstance(models[0], dict):
        raise RemoteError(
            "Not an MMCP Kimodo server (no models in /capabilities).",
            code="bad_response",
        )
    model = models[0]
    skeleton = model.get("canonical_skeleton")
    joints = skeleton.get("joints") if isinstance(skeleton, dict) else None
    if not model.get("id") or not model.get("fps") or not joints:
        raise RemoteError(
            "Not an MMCP Kimodo server (model is missing id, fps, or skeleton).",
            code="bad_response",
        )
    return model


def _get_json(url: str, timeout: float) -> dict:
    status, headers, body = _open(url, method="GET", timeout=timeout)
    return _parse_json_body(status, headers, body, url)


def _generate(server_url: str, body: dict, timeout: float, progress_callback):
    url = server_url.rstrip("/") + "/generate"
    payload = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "model/gltf+json, application/json, model/gltf-binary;q=0.9",
    }
    deadline = time.monotonic() + timeout
    status, resp_headers, raw = _open(
        url, method="POST", data=payload, headers=headers, timeout=timeout,
    )
    if _cancel.is_set():
        raise RemoteError("Cancelled by user", code="cancelled")
    if status == 202:
        location = resp_headers.get("Location") or ""
        try:
            retry_after = float(resp_headers.get("Retry-After") or "2")
        except ValueError:
            retry_after = 2.0
        return _poll(server_url, location, retry_after, deadline, progress_callback)
    if status != 200:
        raise RemoteError(
            f"Unexpected status {status} from /generate",
            code="bad_response",
        )
    return _decode_generate_body(status, resp_headers, raw)


def _poll(server_url, location, retry_after, deadline, progress_callback):
    if not location:
        raise RemoteError(
            "Server returned 202 Accepted with no Location header.",
            code="bad_response",
        )
    url = _job_url(server_url, location)
    started = time.monotonic()
    while time.monotonic() < deadline:
        if _cancel.is_set():
            raise RemoteError("Cancelled by user", code="cancelled")
        time.sleep(max(retry_after, 0.5))
        if progress_callback:
            progress_callback(f"Generating… {time.monotonic() - started:.0f}s")
        remaining = max(1.0, deadline - time.monotonic())
        status, headers, body = _open(
            url, method="GET", timeout=min(_POLL_TIMEOUT, remaining),
        )
        if status == 200:
            return _decode_generate_body(status, headers, body)
        if status == 202:
            try:
                retry_after = float(headers.get("Retry-After") or retry_after)
            except ValueError:
                pass
            continue
        raise RemoteError(f"Poll returned HTTP {status}", code="http_error")
    raise RemoteError(
        f"Remote generation did not finish within {int(_GENERATE_TIMEOUT)}s",
        code="timeout",
    )


def _job_url(server_url: str, location: str) -> str:
    if location.startswith(("http://", "https://")):
        parsed = urllib.parse.urlparse(location)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        location = path
    if not location.startswith("/"):
        location = "/" + location
    return server_url.rstrip("/") + location


def _open(url, method="GET", data=None, headers=None, timeout=30.0):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read()
        except Exception:
            raw = b""
        raise RemoteError(_http_error_message(exc, raw), code="http_error") from exc
    except urllib.error.URLError as exc:
        raise RemoteError(
            f"Could not reach server: {exc.reason}",
            code="connection_failed",
        ) from exc
    except TimeoutError as exc:
        raise RemoteError(f"Timed out talking to {url}", code="timeout") from exc


def _http_error_message(exc, raw: bytes) -> str:
    fallback = f"HTTP {exc.code} from the server"
    if not raw:
        return f"{fallback}: {exc.reason}"
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception:
        preview = raw[:200].decode("utf-8", errors="replace")
        return f"{fallback}: {preview}"
    if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
        err = parsed["error"]
        return err.get("message") or fallback
    return fallback


def _parse_json_body(status, headers, body, url) -> dict:
    if not body:
        raise RemoteError(
            f"Empty response from {url} (HTTP {status}).",
            code="bad_response",
        )
    ctype = headers.get("Content-Type", "") if headers is not None else ""
    if "json" not in ctype and not body.lstrip().startswith((b"{", b"[")):
        preview = body[:200].decode("utf-8", errors="replace")
        raise RemoteError(
            f"Not an MMCP Kimodo server (HTTP {status}, {ctype or 'no content type'}: {preview})",
            code="bad_response",
        )
    try:
        data = json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise RemoteError(
            f"Capabilities response was not valid JSON: {exc}",
            code="bad_response",
        ) from exc
    if not isinstance(data, dict):
        raise RemoteError(
            "Not an MMCP Kimodo server (capabilities was not a JSON object).",
            code="bad_response",
        )
    return data


def _decode_generate_body(status, headers, body) -> dict:
    if not body:
        raise RemoteError(
            f"Server returned an empty HTTP {status} from /generate.",
            code="bad_response",
        )
    ctype = headers.get("Content-Type", "") if headers is not None else ""
    if "model/gltf-binary" in ctype or body[:4] == b"glTF":
        try:
            return glb_to_gltf(body)
        except ValueError as exc:
            raise RemoteError(
                f"Generate response was not a valid GLB: {exc}",
                code="bad_response",
            ) from exc
    try:
        data = json.loads(body.decode("utf-8"))
    except Exception as exc:
        preview = body[:200].decode("utf-8", errors="replace")
        raise RemoteError(
            f"Generate response was not valid JSON: {exc} ({preview})",
            code="bad_response",
        ) from exc
    if not isinstance(data, dict):
        raise RemoteError(
            "Generate response was not a glTF document.",
            code="bad_response",
        )
    return data


def _unlink(path) -> None:
    if path:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# GLB → glTF JSON
# Adapted from motionmcp.client.glb (Apache-2.0).
# ---------------------------------------------------------------------------

_GLB_MAGIC = b"glTF"
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942


def glb_to_gltf(data: bytes) -> dict:
    """Unpack a GLB container into a JSON-serialisable glTF document."""
    import base64
    import struct

    if len(data) < 12:
        raise ValueError(f"GLB shorter than its 12-byte header ({len(data)} bytes)")
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != _GLB_MAGIC:
        raise ValueError(f"not a GLB: magic {magic!r}")
    if version != 2:
        raise ValueError(f"unsupported GLB version {version}")
    if length != len(data):
        raise ValueError(f"GLB length field {length} != actual size {len(data)}")

    chunks = []
    offset = 12
    while offset < length:
        if offset + 8 > length:
            raise ValueError("truncated GLB chunk header")
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        start = offset + 8
        end = start + chunk_len
        if end > length:
            raise ValueError("GLB chunk runs past the end of the file")
        chunks.append((chunk_type, data[start:end]))
        offset = end

    if not chunks or chunks[0][0] != _CHUNK_JSON:
        raise ValueError("GLB does not start with a JSON chunk")
    doc = json.loads(chunks[0][1].decode("utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("GLB JSON chunk is not an object")

    bin_chunk = None
    if len(chunks) > 1 and chunks[1][0] == _CHUNK_BIN:
        bin_chunk = chunks[1][1]
    buffers = doc.get("buffers")
    if bin_chunk is not None and isinstance(buffers, list) and buffers:
        first = buffers[0]
        if isinstance(first, dict) and "uri" not in first:
            byte_length = first.get("byteLength")
            if isinstance(byte_length, int) and 0 <= byte_length <= len(bin_chunk):
                bin_chunk = bin_chunk[:byte_length]
            first["uri"] = (
                "data:application/octet-stream;base64,"
                + base64.b64encode(bin_chunk).decode("ascii")
            )
    return doc
