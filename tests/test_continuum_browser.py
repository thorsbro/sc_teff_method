"""Real mouse/numerical continuum editing, with analytical synthetic input only."""
# ruff: noqa: E402

import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("matplotlib")
pytest.importorskip("bokeh")
playwright = pytest.importorskip("playwright.sync_api")

from sc_teff_method import get_calibration
from sc_teff_method.synthetic import make_continuum_demo


def model_text(page, name):
    return page.evaluate("name => Bokeh.documents[0].get_model_by_name(name).text", name)


def wait_text(page, name, fragment):
    page.wait_for_function(
        "([name, text]) => "
        "window.Bokeh?.documents?.[0]?.get_model_by_name(name)?.text.includes(text)",
        arg=[name, fragment],
        timeout=20000,
    )


def click(page, name):
    page.locator(f".{name} button").click(timeout=15000)


def fill(page, name, text):
    field = page.locator(f".{name} input")
    field.fill(str(text))
    field.press("Tab")


def select_line(page, line):
    page.locator(".line_selector select").select_option(line)
    wait_text(page, "fit_details", f"{line}:")


@pytest.fixture
def continuum_application(tmp_path):
    with playwright.sync_playwright() as engine:
        if not Path(engine.chromium.executable_path).exists():
            pytest.skip(
                "Install the optional Chromium test browser: python -m playwright install chromium"
            )
    synthetic = make_continuum_demo(noise_sigma=0.001)
    source = tmp_path / "analytical.csv"
    np.savetxt(
        source,
        np.column_stack(
            (
                synthetic.wavelength_air_angstrom,
                synthetic.flux,
                synthetic.uncertainty,
                synthetic.mask.astype(int),
            )
        ),
        delimiter=",",
        header="wave,flux,error,exclude",
        comments="",
        fmt="%.17g",
    )
    review, output = tmp_path / "review.json", tmp_path / "products"
    command = [
        sys.executable,
        "-m",
        "sc_teff_method.cli",
        "inspect",
        str(source),
        "--delimiter",
        ",",
        "--header",
        "--wavelength-column",
        "wave",
        "--flux-column",
        "flux",
        "--uncertainty-column",
        "error",
        "--mask-column",
        "exclude",
        "--wavelength-unit",
        "angstrom",
        "--wavelength-frame",
        "air",
        "--continuum-normalised",
        "--feh",
        "-0.2",
        "--save-review",
        str(review),
        "--output-dir",
        str(output),
        "--no-browser",
        "--label",
        "Analytical manual continuum browser test",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "BOKEH_RESOURCES": "cdn"},
    )
    messages, lines = queue.Queue(), []

    def collect():
        for line in process.stdout:
            lines.append(line)
            messages.put(line)

    reader = threading.Thread(target=collect, daemon=True)
    reader.start()
    url = None
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail("Local review server exited: " + "".join(lines))
            try:
                message = messages.get(timeout=0.2)
            except queue.Empty:
                continue
            match = re.search(r"http://127\.0\.0\.1:\d+/", message)
            if match:
                url = match.group()
                break
        assert url is not None, "Server did not report a loopback URL: " + "".join(lines)
        yield {"url": url, "source": source, "review": review, "output": output, "log": lines}
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
            with socket.socket() as connection:
                connection.settimeout(1)
                assert connection.connect_ex(("127.0.0.1", urlparse(url).port)) != 0


def test_browser_manual_continuum_anchors_review_save_restore_and_headless_replay(
    continuum_application,
):
    app = continuum_application
    original_source = app["source"].read_bytes()
    with playwright.sync_playwright() as engine:
        browser = engine.chromium.launch(
            headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(viewport={"width": 1600, "height": 1450})
        external, errors = [], []

        def route(request):
            if urlparse(request.request.url).hostname == "127.0.0.1":
                request.continue_()
            else:
                external.append(request.request.url)
                request.abort()

        page.route("**/*", route)
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(app["url"], wait_until="networkidle", timeout=30000)
            wait_text(page, "reviewed_summary", "Missing")
            select_line(page, "line3")
            click(page, "accept_line")
            wait_text(page, "reviewed_summary", "Accepted lines (1)")
            before = float(
                re.search(r"Area ([\d.eE+-]+)", model_text(page, "fit_details")).group(1)
            )
            # Actual mouse taps add/move both anchors; no Python callback injection.
            for action, fraction in (("anchor1", 0.22), ("anchor2", 0.80)):
                page.locator(".continuum_mouse select").select_option(
                    index=2 if action == "anchor1" else 3
                )
                canvas = page.locator(".selected_profile canvas").first
                canvas.scroll_into_view_if_needed()
                rectangle = canvas.bounding_box()
                assert rectangle is not None
                page.mouse.click(rectangle["x"] + fraction*rectangle["width"],
                                 rectangle["y"] + 0.18*rectangle["height"])
                wait_text(page, "review_message", "refitted")
                page.wait_for_function(
                    "Bokeh.documents[0].get_model_by_name('continuum_enabled').active.length===1",
                    timeout=15000,
                )
            wait_text(page, "reviewed_summary", "Missing")
            assert page.locator(".continuum_mode select").input_value() == "linear"
            x1 = float(page.locator(".continuum_anchor1_x input").input_value())
            x2 = float(page.locator(".continuum_anchor2_x input").input_value())
            assert x2 > x1
            # Precision editing supplies the known invented analytical continuum.
            rest = get_calibration().lines["line3"].wavelength_air_angstrom
            fill(page, "continuum_anchor1_y", 1.012+0.006*(x1-rest))
            fill(page, "continuum_anchor2_y", 1.012+0.006*(x2-rest))
            wait_text(page, "fit_details", "Before correction:")
            after = float(re.search(r"Area ([\d.eE+-]+)", model_text(page, "fit_details")).group(1))
            assert abs(before-after) > 0.01
            click(page, "accept_line")
            wait_text(page, "reviewed_summary", "Accepted lines (1)")
            fill(page, "continuum_anchor2_y", "0")
            wait_text(page, "fit_details", "Unapplied draft")
            playwright.expect(page.locator(".accept_line button")).to_be_disabled()
            # Disabling remains possible even when an anchor draft is invalid.
            page.locator(".continuum_enabled input").uncheck()
            wait_text(page, "review_message", "refitted")
            playwright.expect(page.locator(".accept_line button")).to_be_enabled()
            fill(page, "continuum_anchor2_y", 1.012+0.006*(x2-rest))
            page.locator(".continuum_enabled input").check()
            wait_text(page, "review_message", "refitted")
            click(page, "accept_line")
            click(page, "save_review")
            wait_text(page, "review_message", "saved")
            saved = json.loads(app["review"].read_text())
            correction = saved["lines"]["line3"]["controls"]["continuum"]
            assert saved["schema_version"] == 2
            assert correction["enabled"] and correction["mode"] == "linear"
            click(page, "reset_continuum")
            wait_text(page, "review_message", "reset to unity")
            wait_text(page, "reviewed_summary", "Missing")
            click(page, "load_review")
            wait_text(page, "review_message", "restored")
            wait_text(page, "reviewed_summary", "Accepted lines (1)")
            click(page, "export_results")
            wait_text(page, "review_message", "exported")
            summary = json.loads((app["output"] / "summary.json").read_text())
            replay_dir = app["output"].parent / "headless-results"
            replay = subprocess.run(
                [sys.executable, "-m", "sc_teff_method.cli", "replay", str(app["source"]),
                 "--review-file", str(app["review"]), "--output-dir", str(replay_dir)],
                capture_output=True, text=True, timeout=60,
            )
            assert replay.returncode == 0, replay.stdout+replay.stderr
            replayed = json.loads((replay_dir / "summary.json").read_text())
            assert replayed["temperature"] == summary["temperature"]
            assert replayed["included_line_ids"] == ["line3"]
            assert app["source"].read_bytes() == original_source
            assert not external
            assert not errors
        except Exception:
            print("Browser message:", model_text(page, "review_message"))
            print("Browser details:", model_text(page, "fit_details"))
            print("Continuum controls:", page.evaluate(
                "(() => { const d=Bokeh.documents[0]; return "
                "Object.fromEntries(['continuum_anchor1_x','continuum_anchor1_y',"
                "'continuum_anchor2_x','continuum_anchor2_y'].map(n => "
                "[n,[d.get_model_by_name(n).value,d.get_model_by_name(n).value_input]])); })()"
            ))
            print("Server log:", "".join(app["log"][-20:]))
            raise
        finally:
            browser.close()
