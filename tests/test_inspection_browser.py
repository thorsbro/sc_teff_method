"""Real offline browser checks, using only analytical Gaussian demonstration data."""
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

from sc_teff_method.synthetic import make_review_demo


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
def local_application(tmp_path):
    with playwright.sync_playwright() as engine:
        if not Path(engine.chromium.executable_path).exists():
            pytest.skip(
                "Install the optional Chromium test browser: python -m playwright install chromium"
            )
    synthetic = make_review_demo()
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
        "Analytical synthetic browser test",
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


def test_browser_review_masks_approval_save_restore_export_and_offline_assets(local_application):
    app = local_application
    original_source = app["source"].read_bytes()
    with playwright.sync_playwright() as engine:
        browser = engine.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(viewport={"width": 1600, "height": 1250})
        external, errors, sockets = [], [], []

        def route(request):
            if urlparse(request.request.url).hostname == "127.0.0.1":
                request.continue_()
            else:
                external.append(request.request.url)
                request.abort()

        page.route("**/*", route)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("websocket", lambda ws: sockets.append(ws.url))
        try:
            page.goto(app["url"], wait_until="networkidle", timeout=30000)
            wait_text(page, "reviewed_summary", "Missing")
            assert page.locator("canvas").count() >= 20
            click(page, "accept_line")
            wait_text(page, "reviewed_summary", "Accepted lines (1)")
            lower = float(page.locator(".window_lower input").input_value())
            fill(page, "window_lower", repr(lower + 0.2))
            wait_text(page, "review_message", "refitted")
            wait_text(page, "reviewed_summary", "Missing")
            click(page, "accept_line")
            wait_text(page, "reviewed_summary", "Accepted lines (1)")
            fill(page, "window_lower", "not a wavelength")
            wait_text(page, "fit_details", "Unapplied draft")
            playwright.expect(page.locator(".accept_line button")).to_be_disabled()
            click(page, "save_review")
            wait_text(page, "review_message", "draft fitting changes")
            assert not app["review"].exists()
            click(page, "reset_line")
            wait_text(page, "review_message", "reset to default")
            click(page, "accept_line")
            wait_text(page, "reviewed_summary", "Accepted lines (1)")

            select_line(page, "line4")
            before = float(
                re.search(r"Area ([\d.eE+-]+)", model_text(page, "fit_details")).group(1)
            )
            playwright.expect(page.locator(".accept_line button")).to_have_text(
                "Accept flagged fit"
            )
            click(page, "accept_line")
            wait_text(page, "reviewed_summary", "Accepted lines (2)")
            fill(page, "mask_lower", "22024.318")
            fill(page, "mask_upper", "22025.768")
            click(page, "mask_add")
            wait_text(page, "review_message", "refitted")
            wait_text(page, "reviewed_summary", "Accepted lines (1)")
            after = float(re.search(r"Area ([\d.eE+-]+)", model_text(page, "fit_details")).group(1))
            assert before - after > 0.1
            fill(page, "review_note", "Inspected the analytically invented neighbour and new mask.")
            click(page, "accept_line")
            wait_text(page, "reviewed_summary", "Accepted lines (2)")

            select_line(page, "line5")
            canvas = page.locator(".selected_profile canvas").first
            rectangle = canvas.bounding_box()
            assert rectangle is not None
            page.mouse.move(
                rectangle["x"] + 0.65 * rectangle["width"],
                rectangle["y"] + 0.5 * rectangle["height"],
            )
            page.mouse.down()
            page.mouse.move(
                rectangle["x"] + 0.8 * rectangle["width"],
                rectangle["y"] + 0.5 * rectangle["height"],
                steps=10,
            )
            page.mouse.up()
            page.wait_for_function(
                "Bokeh.documents[0].get_model_by_name('mask_selector').options"
                ".filter(o => (Array.isArray(o) ? o[0] : o) !== '').length > 0",
                timeout=15000,
            )
            # Bokeh may encode tuple-valued options in the native DOM; select by position.
            page.locator(".mask_selector select").select_option(index=1)
            playwright.expect(page.locator(".mask_lower input")).not_to_have_value("")
            click(page, "mask_remove")
            page.wait_for_function(
                "Bokeh.documents[0].get_model_by_name('mask_selector').options"
                ".filter(o => (Array.isArray(o) ? o[0] : o) !== '').length === 0",
                timeout=15000,
            )
            select_line(page, "line4")
            wait_text(page, "fit_details", "ACCEPTED")
            assert float(page.locator(".mask_lower input").input_value()) == pytest.approx(
                22024.318
            )
            click(page, "save_review")
            wait_text(page, "review_message", "saved")
            saved = json.loads(app["review"].read_text())
            assert saved["lines"]["line4"]["decision"]["state"] == "accepted"
            assert saved["lines"]["line4"]["fit"]["flags"]
            click(page, "reject_line")
            wait_text(page, "reviewed_summary", "Accepted lines (1)")
            click(page, "load_review")
            wait_text(page, "review_message", "restored")
            wait_text(page, "reviewed_summary", "Accepted lines (2)")
            click(page, "export_results")
            wait_text(page, "review_message", "exported")
            summary = json.loads((app["output"] / "summary.json").read_text())
            assert summary["included_line_ids"] == ["line1", "line4"]
            assert summary["review"]["decisions"]["line4"]["state"] == "accepted"
            assert "post-hoc" in summary["review"]["selection_note"]
            assert app["source"].read_bytes() == original_source
            assert not external
            assert sockets and all(urlparse(url).hostname == "127.0.0.1" for url in sockets)
            assert not errors
        except Exception:
            print("Browser message:", model_text(page, "review_message"))
            print(
                "Browser mask state:",
                page.evaluate(
                    "(() => { const d=Bokeh.documents[0]; return {"
                    "options:d.get_model_by_name('mask_selector').options,"
                    "selected:d.get_model_by_name('mask_selector').value,"
                    "lower:d.get_model_by_name('mask_lower').value,"
                    "upper:d.get_model_by_name('mask_upper').value}; })()"
                ),
            )
            print("Local server log:", "".join(app["log"][-20:]))
            raise
        finally:
            browser.close()
