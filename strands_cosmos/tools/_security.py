# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Security primitives for agent-reachable Cosmos tools.

Why this module exists
----------------------
Tool arguments come from model output and are treated as untrusted input. To
keep them safe, the tools never interpolate those values into shell or ``just``
recipe templates; instead they go through the helpers below, which validate and
confine every path, URL, and identifier before it is used.

This module provides the building blocks:

  * ``safe_run``            -- execute a binary via an argv list (shell=False,
                              no string interpolation, never via ``just``).
  * ``resolve_in_workspace`` -- confine an *input* path to an allow-listed root
                              (resolves symlinks, rejects ".." / escape).
  * ``resolve_output_path``  -- confine an *output* path the same way.
  * ``validate_url``        -- SSRF guard for any agent-influenced URL
                              (scheme + host allow-list, blocks private/
                              link-local/loopback/metadata ranges).
  * ``validate_nats_subject`` -- enforce a NATS subject namespace prefix.

Configuration (environment)
---------------------------
  COSMOS_WORKSPACE        Path-separated allow-listed roots for file I/O.
                          Default: current working directory + system tempdir.
  COSMOS_ALLOW_TEMP       "1"/"true" (default) to include the system tempdir.
  COSMOS_URL_ALLOWLIST    Comma-separated host[:port] entries the tools may
                          POST to (in addition to localhost).
  COSMOS_ALLOW_REMOTE_URLS "1" to relax the SSRF host allow-list to any public
                          host (private/link-local still blocked). Default off.
  COSMOS_NATS_NAMESPACE   Comma-separated allowed NATS subject prefixes.
                          Default: "cosmos,agent,perception".
"""
from __future__ import annotations

import ipaddress
import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse


class SecurityError(ValueError):
    """Raised when an agent-supplied argument violates a security policy."""


# -- Workspace path containment -------------------------------------------
def _allowed_roots() -> list[Path]:
    """Return the list of allow-listed filesystem roots (resolved)."""
    roots: list[Path] = []
    env = os.getenv("COSMOS_WORKSPACE", "").strip()
    if env:
        for part in env.split(os.pathsep):
            part = part.strip()
            if part:
                try:
                    roots.append(Path(part).expanduser().resolve())
                except Exception:
                    continue
    else:
        roots.append(Path.cwd().resolve())

    if os.getenv("COSMOS_ALLOW_TEMP", "1").lower() in ("1", "true", "yes"):
        try:
            roots.append(Path(tempfile.gettempdir()).resolve())
        except Exception:
            pass

    seen: set[str] = set()
    uniq: list[Path] = []
    for r in roots:
        s = str(r)
        if s not in seen:
            seen.add(s)
            uniq.append(r)
    return uniq


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _contain(path: str, *, must_exist: bool) -> Path:
    """Resolve ``path`` and verify it stays within an allow-listed root."""
    if path is None or str(path).strip() == "":
        raise SecurityError("empty path")

    raw = str(path)
    p = Path(raw).expanduser()

    if ".." in Path(raw).parts:
        raise SecurityError(f"path traversal ('..') not allowed: {raw}")

    try:
        resolved = p.resolve()
    except Exception as e:  # pragma: no cover
        raise SecurityError(f"cannot resolve path: {raw} ({e})")

    roots = _allowed_roots()
    if not any(_is_within(resolved, root) or resolved == root for root in roots):
        raise SecurityError(
            f"path escapes the allowed workspace: {resolved} "
            f"(allowed roots: {', '.join(str(r) for r in roots)}; "
            f"set COSMOS_WORKSPACE to widen)"
        )

    if must_exist and not resolved.exists():
        raise SecurityError(f"path not found in workspace: {resolved}")

    return resolved


def resolve_in_workspace(path: str, *, must_exist: bool = True) -> Path:
    """Validate and resolve an *input* path. Raises ``SecurityError`` on escape."""
    return _contain(path, must_exist=must_exist)


def resolve_output_path(path: str) -> Path:
    """Validate and resolve an *output* path (parent dir created on demand)."""
    resolved = _contain(path, must_exist=False)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


# -- SSRF / URL allow-listing ---------------------------------------------
_BLOCKED_NETS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _host_allowlist() -> set[str]:
    allow = {"localhost", "127.0.0.1", "::1"}
    extra = os.getenv("COSMOS_URL_ALLOWLIST", "").strip()
    if extra:
        for part in extra.split(","):
            part = part.strip().lower()
            if part:
                allow.add(part)
    return allow


def _is_blocked_ip(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return True
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr.split("%")[0])
        except ValueError:
            return True
        if any(ip in net for net in _BLOCKED_NETS):
            return True
        if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            return True
    return False


def validate_url(url: str, *, allow_public: bool = False) -> str:
    """Validate an agent-influenced URL against the SSRF policy.

    Policy (control-plane endpoints, e.g. inference ``server_url``):
      * scheme must be http/https
      * host must be in the allow-list (localhost + COSMOS_URL_ALLOWLIST), OR
        COSMOS_ALLOW_REMOTE_URLS=1 and the host does not resolve to a
        private/link-local/loopback/metadata address.

    When ``allow_public=True`` (used for remote *media* references that the
    backend fetches, e.g. ``<video>https://...</video>``): any public host is
    permitted, but private/link-local/loopback/metadata targets are still
    blocked -- this preserves remote-media support while killing SSRF to
    internal/metadata endpoints (e.g. 169.254.169.254).

    Returns the URL unchanged if allowed; raises ``SecurityError`` otherwise.
    """
    if not url or not str(url).strip():
        raise SecurityError("empty URL")
    parsed = urlparse(str(url))
    if parsed.scheme not in ("http", "https"):
        raise SecurityError(f"URL scheme not allowed: {parsed.scheme!r} (use http/https)")
    host = (parsed.hostname or "").lower()
    if not host:
        raise SecurityError(f"URL has no host: {url}")

    allow = _host_allowlist()
    host_port = f"{host}:{parsed.port}" if parsed.port else host

    if host in allow or host_port in allow:
        return str(url)

    remote_ok = allow_public or os.getenv("COSMOS_ALLOW_REMOTE_URLS", "").lower() in (
        "1", "true", "yes"
    )
    if remote_ok:
        if _is_blocked_ip(host):
            raise SecurityError(
                f"URL host resolves to a private/link-local/metadata address: {host}"
            )
        return str(url)

    raise SecurityError(
        f"URL host not in allow-list: {host}. "
        f"Add it to COSMOS_URL_ALLOWLIST or set COSMOS_ALLOW_REMOTE_URLS=1."
    )


# -- Identifier / enum validation -----------------------------------------
import re as _re

# Model names / HF repo ids / dtypes / recipe names: a conservative charset that
# cannot carry shell or `just`-template metacharacters. Covers "org/repo",
# "Cosmos3-Nano", "fp8", "vision_sft_nano", versioned tags, etc.
_IDENT_RE = _re.compile(r"^[A-Za-z0-9~./][A-Za-z0-9 ._/@:+=-]{0,1024}$")


def validate_identifier(value: str, *, what: str = "value", allow_empty: bool = False) -> str:
    """Validate a model/dataset name, dtype, or recipe id contains only safe characters.

    These values may be passed positionally into ``just`` recipes, so they must
    not contain quotes, whitespace, or shell/template metacharacters. Raises
    ``SecurityError`` if the value is unsafe.
    """
    if value is None or str(value) == "":
        if allow_empty:
            return ""
        raise SecurityError(f"empty {what}")
    s = str(value)
    if not _IDENT_RE.match(s):
        raise SecurityError(
            f"invalid {what} {s[:60]!r}: only letters, digits and ._/@:+- allowed"
        )
    return s


# -- NATS subject namespace -----------------------------------------------
def _nats_namespaces() -> list[str]:
    env = os.getenv("COSMOS_NATS_NAMESPACE", "cosmos,agent,perception").strip()
    return [p.strip() for p in env.split(",") if p.strip()]


def validate_nats_subject(subject: str) -> str:
    """Enforce a NATS subject namespace + reject control chars/wildcards."""
    if not subject or not str(subject).strip():
        raise SecurityError("empty NATS subject")
    s = str(subject)
    if any(c.isspace() for c in s):
        raise SecurityError("NATS subject must not contain whitespace")
    if "*" in s or ">" in s:
        raise SecurityError("NATS publish subject must not contain wildcards (*, >)")
    for tok in s.split("."):
        if tok == "":
            raise SecurityError(f"invalid NATS subject (empty token): {s!r}")
    namespaces = _nats_namespaces()
    root = s.split(".", 1)[0]
    if root not in namespaces:
        raise SecurityError(
            f"NATS subject {s!r} outside allowed namespaces {namespaces}. "
            f"Set COSMOS_NATS_NAMESPACE to change."
        )
    return s


# -- Safe subprocess execution (argv list, shell=False, never via just) ---
def safe_run(
    argv: Sequence[str],
    *,
    timeout_s: int = 3600,
    cwd: str | None = None,
    env: dict | None = None,
    extra_env: dict | None = None,
    input_bytes: bytes | None = None,
) -> dict:
    """Run a command from an argv list with shell=False. Never raises.

    This is the trusted execution primitive: arguments are passed as discrete
    argv elements, so there is no shell/template layer to break out of.

    Returns a normalized proc dict compatible with ``proc_result``:
        {"ok", "returncode", "stdout", "stderr", "cmd", "cwd"}
    """
    argv = [str(a) for a in argv]
    if not argv:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": "empty argv", "cmd": ""}

    binary = argv[0]
    if shutil.which(binary) is None and not os.path.isabs(binary):
        return {
            "ok": False,
            "returncode": 127,
            "stdout": "",
            "stderr": f"`{binary}` not found on PATH",
            "cmd": " ".join(argv),
            "cwd": cwd,
        }

    run_env = os.environ.copy() if env is None else dict(env)
    if extra_env:
        run_env.update(extra_env)

    try:
        p = subprocess.run(
            argv,
            capture_output=True,
            text=input_bytes is None,
            timeout=timeout_s,
            cwd=cwd,
            env=run_env,
            input=input_bytes if input_bytes is not None else None,
            shell=False,
        )
        out = p.stdout if isinstance(p.stdout, str) else (p.stdout or b"").decode("utf-8", "replace")
        errtxt = p.stderr if isinstance(p.stderr, str) else (p.stderr or b"").decode("utf-8", "replace")
        return {
            "ok": p.returncode == 0,
            "returncode": p.returncode,
            "stdout": out[-8000:],
            "stderr": errtxt[-4000:],
            "cmd": " ".join(argv),
            "cwd": cwd,
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False, "returncode": -1, "stdout": "",
            "stderr": f"timeout after {timeout_s}s", "cmd": " ".join(argv), "cwd": cwd,
        }
    except Exception as e:
        return {
            "ok": False, "returncode": -1, "stdout": "",
            "stderr": str(e), "cmd": " ".join(argv), "cwd": cwd,
        }
