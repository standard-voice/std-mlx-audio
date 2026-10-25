# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Downloads honor the endpoint and proxy variables of the environment.

Standard ASR neither sets nor changes ``HF_ENDPOINT``, ``HTTP_PROXY``,
``HTTPS_PROXY``, or ``NO_PROXY``; the native library reads them. This test runs the real download
path, the plugin's call to ``huggingface_hub.snapshot_download``, in a
subprocess, because the library reads ``HF_ENDPOINT`` once per process, on
first use. A local proxy records each request and answers 404, so no HTTP
request leaves the machine.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

#: Variables the subprocess must not inherit, in upper or lower case: they
#: would change where the download goes or whether it runs at all.
_INHERITED = frozenset(
    {
        "HF_ENDPOINT",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "STANDARD_ASR_ALLOW_DOWNLOAD",
    }
)

_ACQUIRE = """
import sys
from std_mlx_audio import WhisperTiny
try:
    WhisperTiny(download_root=sys.argv[1]).acquire_artifacts()
except Exception as error:
    # The chain of causes, outermost first, shows why acquisition failed.
    link, names = error, []
    while link is not None and len(names) < 12:
        names.append(type(link).__name__)
        link = link.__cause__ or link.__context__
    print(" > ".join(names))
"""

#: The Hub repository the ``whisper-tiny`` preset downloads.
_REPO = "openai/whisper-tiny"


@pytest.fixture
def proxy() -> Iterator[tuple[str, list[str]]]:
    """Run a proxy on 127.0.0.1 that records request lines and answers 404."""
    seen: list[str] = []

    class Recorder(BaseHTTPRequestHandler):
        def _answer(self) -> None:
            seen.append(self.requestline)
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_GET = do_HEAD = do_CONNECT = _answer

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", seen
    finally:
        server.shutdown()
        server.server_close()


def _acquire(tmp_path: Path, settings: dict[str, str]) -> str:
    """Run an acquisition in a subprocess; return the class names in its failure's cause chain."""
    env = {key: value for key, value in os.environ.items() if key.upper() not in _INHERITED}
    env.update(settings)
    env["STANDARD_ASR_ALLOW_DOWNLOAD"] = "1"
    env["HF_HOME"] = str(tmp_path / "hf-home")
    result = subprocess.run(
        [sys.executable, "-c", _ACQUIRE, str(tmp_path / "models")],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    # Every case ends in a failed acquisition, because no host answers with
    # files; the recorded request lines say where each request went.
    assert result.stdout.startswith("ArtifactAcquisitionError"), result
    return result.stdout


def test_download_goes_where_the_environment_points(
    proxy: tuple[str, list[str]], tmp_path: Path
) -> None:
    url, seen = proxy

    # HF_ENDPOINT names the host, and the request for it goes to the proxy in HTTP_PROXY.
    _acquire(tmp_path, {"HF_ENDPOINT": "http://mirror.invalid", "HTTP_PROXY": url})
    assert any(f"http://mirror.invalid/api/models/{_REPO}" in line for line in seen), seen

    # Without HF_ENDPOINT, the request for the default endpoint goes to the proxy in HTTPS_PROXY.
    seen.clear()
    _acquire(tmp_path, {"HTTPS_PROXY": url})
    assert any(line.startswith("CONNECT huggingface.co:443 ") for line in seen), seen

    # NO_PROXY sends the request for the mirror around the proxy: it goes
    # straight to mirror.invalid, whose name never resolves.
    seen.clear()
    chain = _acquire(
        tmp_path,
        {"HF_ENDPOINT": "http://mirror.invalid", "HTTP_PROXY": url, "NO_PROXY": "mirror.invalid"},
    )
    assert seen == []
    assert "gaierror" in chain, chain
