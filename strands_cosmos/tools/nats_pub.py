# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish a JSON payload to a NATS subject.

Sends a message to a NATS messaging server via the ``nats`` CLI. The subject is
confined to a configurable namespace (``COSMOS_NATS_NAMESPACE``) and the payload
is delivered on stdin, so it integrates cleanly with downstream Cosmos services.
"""
from __future__ import annotations

import json
import os

from strands import tool

from ._common import err, ok
from ._security import SecurityError, safe_run, validate_nats_subject


@tool
def nats_publish(
    subject: str,
    payload: str,
    servers: str = "",
) -> dict:
    """Publish a JSON payload to a NATS subject.

    The subject must fall within an allowed namespace (default: cosmos.*,
    agent.*, perception.*; configurable via COSMOS_NATS_NAMESPACE).

    Args:
        subject: NATS subject (e.g. "perception.vlm").
        payload: JSON string payload.
        servers: NATS URL(s). Default: NATS_URL env or nats://127.0.0.1:4222.

    Returns:
        A Strands tool-result dict ``{"status", "content"}``. On success the
        content carries confirmation that the payload was published to the subject; on error ``status`` is ``"error"`` with a message.
    """
    # Validate subject namespace (rejects system topics, wildcards, injection).
    try:
        subject = validate_nats_subject(subject)
    except SecurityError as e:
        return err(str(e))

    # Validate JSON payload.
    try:
        json.loads(payload)
    except json.JSONDecodeError as e:
        return err(f"payload is not valid JSON: {e}")

    nats_url = servers or os.getenv("NATS_URL", "nats://127.0.0.1:4222")

    # Direct argv invocation: `nats pub <subject> --server <url>` with the
    # payload streamed on stdin. No shell, no interpolation.
    proc = safe_run(
        ["nats", "pub", subject, "--server", nats_url],
        timeout_s=10,
        input_bytes=payload.encode("utf-8"),
    )
    if not proc.get("ok"):
        return err(
            f"NATS publish failed: {proc.get('stderr', '')[:200]}",
            data={"subject": subject, "cmd": proc.get("cmd")},
        )
    return ok(
        f"\U0001F4E1 published to {subject} ({len(payload)}B)",
        data={"subject": subject, "bytes_sent": len(payload), "servers": nats_url},
    )
