"""Full local Chromium smoke test for independent Gaussian/HFS review controls.

Only newly generated analytical hyperfine profiles are used. No private data.
"""
# ruff: noqa: E402
from __future__ import annotations

import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path
from urllib.parse import urlparse

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("bokeh")
playwright = pytest.importorskip("playwright.sync_api")

from sc_teff_method.hfs_synthetic import make_hfs_synthetic_spectrum


def test_local_dual_method_review_browser_and_independent_approvals(tmp_path):
    with playwright.sync_playwright() as engine:
        if not Path(engine.chromium.executable_path).exists():
            pytest.skip("Local Chromium is optional outside CI")
    toy = make_hfs_synthetic_spectrum("clean")
    source = tmp_path / "entirely-mathematical.csv"
    np.savetxt(source, np.column_stack((toy.wavelength_air_angstrom, toy.flux)),
               fmt="%.17g", delimiter=",", header="wavelength,flux", comments="")
    original_bytes = source.read_bytes()
    g_review, h_review = tmp_path / "g-review.json", tmp_path / "h-review.json"
    argv = [
        sys.executable, "-m", "sc_teff_method.cli", "inspect", str(source),
        "--with-hfs", "--delimiter", ",", "--header",
        "--wavelength-column", "wavelength", "--flux-column", "flux",
        "--wavelength-unit", "angstrom", "--wavelength-frame", "air",
        "--continuum-normalised", "--feh", "-0.2", "--no-browser",
        "--save-review", str(g_review), "--save-hfs-review", str(h_review),
        "--output-dir", str(tmp_path / "gaussian-output"),
    ]
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, env={**os.environ, "BOKEH_RESOURCES": "cdn"})
    messages, lines = queue.Queue(), []
    def collect():
        for line in process.stdout:
            lines.append(line)
            messages.put(line)
    reader = threading.Thread(target=collect, daemon=True)
    reader.start()
    url = None
    try:
        for _ in range(200):
            if process.poll() is not None:
                pytest.fail("Review server exited: " + "".join(lines))
            try:
                msg = messages.get(timeout=0.2)
            except queue.Empty:
                continue
            match = re.search(r"http://127\.0\.0\.1:\d+/", msg)
            if match:
                url = match.group()
                break
        assert url is not None, "Missing loopback URL: " + "".join(lines)
        with playwright.sync_playwright() as engine:
            browser = engine.chromium.launch(headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"])
            page = browser.new_page(viewport={"width": 1600, "height": 1250})
            external, errors = [], []
            def route(request):
                if urlparse(request.request.url).hostname == "127.0.0.1":
                    request.continue_()
                else:
                    external.append(request.request.url)
                    request.abort()
            page.route("**/*", route)
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            try:
                page.goto(url, wait_until="networkidle", timeout=45000)
                page.locator(".profile_method select").select_option("HFS")
                playwright.expect(
                    page.get_by_text("Sc I HFS line-fitting review")
                ).to_be_visible(timeout=15000)
                page.locator(".accept_line button:visible").click()
                playwright.expect(
                    page.locator(".reviewed_summary:visible")
                ).to_contain_text("Accepted lines (1)", timeout=15000)
                page.locator(".save_review button:visible").click()
                playwright.expect(page.locator(".review_message:visible")).to_contain_text(
                    "saved", timeout=15000
                )
                assert h_review.exists()
                payload = json.loads(h_review.read_text())
                assert payload["measurement_source"] == "hfs"
                assert payload["lines"]["line1"]["decision"]["state"] == "accepted"

                page.locator(".profile_method select").select_option("Gaussian")
                playwright.expect(
                    page.get_by_text("Sc I Gaussian line-fitting review")
                ).to_be_visible(timeout=15000)
                page.locator(".accept_line button:visible").click()
                page.locator(".save_review button:visible").click()
                playwright.expect(page.locator(".review_message:visible")).to_contain_text(
                    "saved", timeout=15000
                )
                assert g_review.exists()
                g_data = json.loads(g_review.read_text())
                assert g_data["lines"]["line1"]["decision"]["state"] == "accepted"
                page.locator(".profile_method select").select_option("HFS")
                playwright.expect(
                    page.get_by_text("Sc I HFS line-fitting review")
                ).to_be_visible(timeout=15000)
                assert h_review.exists()
                assert source.read_bytes() == original_bytes
                assert not errors, errors
                assert not external, external
            except Exception:
                print("Local Bokeh server log:", "".join(lines[-80:]))
                print("Visible page:", page.content()[:4000])
                print("Browser errors:", errors)
                raise
            finally:
                browser.close()
        cmd = [sys.executable, "-m", "sc_teff_method.cli", "replay-hfs", str(source),
               "--review-file", str(h_review), "--output-dir", str(tmp_path / "hfs-replay")]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads((tmp_path / "hfs-replay" / "summary.json").read_text())[
            "review"]["decisions"]["line1"]["state"] == "accepted"
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        reader.join(timeout=2)
        if url is not None:
            with socket.socket() as sock:
                sock.settimeout(1)
                assert sock.connect_ex(("127.0.0.1", urlparse(url).port)) != 0
