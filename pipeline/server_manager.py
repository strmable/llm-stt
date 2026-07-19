"""llama-server lifecycle management (design.md SS6.3 Launch Mode: External/Managed).

External (config.json local_api.launch_mode == "external", the default): the
user already has llama-server running; we only verify /health and use it,
never touching its lifecycle.

Managed (launch_mode == "managed"): if nothing answers on the configured
port, launch llama-server ourselves from config.json's local_api section
(subprocess), poll /health until ready (or config's managed.startup_timeout_sec
elapses), and terminate it again once the `with ensure_llama_server(...)`
block exits. Per design.md SS6.3's explicit rule, a server that was ALREADY
up when we checked is reused and left running either way, regardless of
launch_mode -- we only ever stop a process we ourselves started.
"""

import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import requests


def is_server_up(server_url: str, timeout: float = 3.0) -> bool:
    try:
        resp = requests.get(f"{server_url.rstrip('/')}/health", timeout=timeout)
        return resp.ok
    except requests.exceptions.RequestException:
        return False


def build_launch_command(local_api: dict) -> list[str]:
    binary = local_api.get("server_binary", "")
    if not binary:
        raise RuntimeError("config.json local_api.server_binary is empty -- required for managed launch_mode")

    port = local_api.get("managed", {}).get("port", 8080)
    cmd = [binary, "--host", "127.0.0.1", "--port", str(port)]

    model_path = local_api.get("model_path", "")
    if model_path:
        cmd += ["--model", model_path]
        mmproj_path = local_api.get("mmproj_path", "")
        if mmproj_path:
            cmd += ["--mmproj", mmproj_path]
    else:
        hf_repo = local_api.get("hf_repo", "")
        if not hf_repo:
            raise RuntimeError("config.json local_api.model_path and local_api.hf_repo are both empty -- "
                                "need one of them to know which model to load")
        cmd += ["-hf", hf_repo]

    extra_args = local_api.get("managed", {}).get("extra_args", "")
    if extra_args:
        cmd += shlex.split(extra_args)
    return cmd


def wait_for_health(server_url: str, timeout_sec: float, poll_interval: float = 2.0) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if is_server_up(server_url, timeout=poll_interval):
            return
        time.sleep(poll_interval)
    raise TimeoutError(f"llama-server did not become healthy at {server_url} within {timeout_sec:.0f}s")


def adapt_text_correction_server_config(tc_server: dict) -> dict:
    """`ensure_llama_server` expects config.json's top-level `local_api` shape
    (managed.port/extra_args/startup_timeout_sec nested). Phase C's server
    config (config.json text_enhancement.text_correction.server,
    postprocessing.md SS12.2) stores those flat instead -- adapt rather than
    duplicate the launch/health-check logic for a second server shape."""
    return {
        "local_api": {
            "launch_mode": tc_server.get("launch_mode", "external"),
            "server_binary": tc_server.get("server_binary", ""),
            "model_path": tc_server.get("model_path", ""),
            "mmproj_path": "",
            "hf_repo": "",
            "managed": {
                "port": tc_server.get("port", 8081),
                "extra_args": tc_server.get("extra_args", ""),
                "startup_timeout_sec": tc_server.get("startup_timeout_sec", 120),
            },
        },
    }


@contextmanager
def ensure_llama_server(server_url: str, config: dict, log_path: Path | None = None):
    """Yields once `server_url` is confirmed reachable. Terminates the
    process on exit ONLY if this call is the one that started it -- an
    already-running server (external mode, or a managed-mode port someone
    else's already using) is never touched.
    """
    local_api = config.get("local_api", {})
    launch_mode = local_api.get("launch_mode", "external")

    if is_server_up(server_url):
        print(f"[server] {server_url} already responding -- reusing it (not managing its lifecycle)")
        yield None
        return

    if launch_mode != "managed":
        sys.exit(f"llama-server not reachable at {server_url} and local_api.launch_mode is "
                 f"{launch_mode!r} (not \"managed\") -- start it first (SETUP.MD SS2/TESTING.md SS1), "
                 f"or set local_api.launch_mode to \"managed\" in config.json.")

    cmd = build_launch_command(local_api)
    print(f"[managed] starting: {' '.join(cmd)}")
    log_file = open(log_path, "w", encoding="utf-8") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    if log_path:
        print(f"[managed] llama-server output -> {log_path}")

    timeout = local_api.get("managed", {}).get("startup_timeout_sec", 120)
    print(f"[managed] waiting up to {timeout:.0f}s for /health ...")
    t0 = time.time()
    try:
        wait_for_health(server_url, timeout)
    except TimeoutError:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise
    print(f"[managed] llama-server ready ({time.time() - t0:.1f}s, PID {proc.pid})")

    try:
        yield proc
    finally:
        if proc.poll() is None:
            print(f"[managed] stopping llama-server (PID {proc.pid})")
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                print("[managed] did not exit in time, killing")
                proc.kill()
                proc.wait()
        if log_path:
            log_file.close()
