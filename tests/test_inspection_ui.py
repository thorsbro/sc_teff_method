"""Local interface checks use analytical synthetic spectra only."""
# ruff: noqa: E402

import hashlib
import queue
import threading
import urllib.request

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("bokeh")

from bokeh.client import pull_session
from bokeh.document import Document
from bokeh.events import ButtonClick, SelectionGeometry

from sc_teff_method.inspection import build_review_document, create_inspection_server
from sc_teff_method.review import ReviewSession
from sc_teff_method.spectrum import prepare_spectrum
from sc_teff_method.synthetic import make_synthetic_spectrum


def session_for(case="blends", *, configuration="nandakumar_feh", feh=-0.2):
    synthetic = make_synthetic_spectrum(case)
    spectrum = prepare_spectrum(
        synthetic.wavelength_air_angstrom, synthetic.flux,
        wavelength_unit="angstrom", wavelength_frame="air", continuum_normalised=True,
        uncertainty=synthetic.uncertainty, mask=synthetic.mask, rv_kms=synthetic.rv_kms,
    )
    identity = {
        "filename": "analytical.csv",
        "sha256": hashlib.sha256(b"invented analytical input identity").hexdigest(),
        "loader_settings": {},
    }
    return ReviewSession.create(spectrum, input_identity=identity,
                                configuration=configuration, feh=feh)


@pytest.fixture
def controller(tmp_path):
    return build_review_document(
        Document(), session_for(), review_path=tmp_path / "review.json",
        output_dir=tmp_path / "review-results", spectrum_label="Analytical synthetic UI test",
    )


def click(widget):
    widget.document.callbacks.trigger_event(ButtonClick(model=widget))


def test_nine_line_overview_and_independent_scientific_state(controller):
    assert len(controller.overview) == 9
    assert len(controller.doc.roots) == 1
    assert "Missing" in controller.summary.text
    assert controller.session.reviewed.temperature is None
    assert all(decision.state == "pending" for decision in controller.session.decisions.values())
    click(controller.widgets["accept_line"])
    assert controller.session.decisions["line1"].state == "accepted"
    assert controller.session.reviewed.included_line_ids == ("line1",)
    assert "Reviewed Teff" in controller.summary.text
    assert "Empirical exact-subset RMSE" in controller.summary.text
    assert "Line-to-line sample scatter" in controller.summary.text


def test_window_change_refits_only_selected_line_and_invalidates_acceptance(
    controller, monkeypatch,
):
    from sc_teff_method import fitting

    old = dict(controller.session.fits)
    original = fitting.least_squares
    calls = []

    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    controller.decide_selected("accepted")
    monkeypatch.setattr(fitting, "least_squares", record)
    controller.widgets["window_lower"].value = str(old["line1"].window[0] + 0.2)
    assert len(calls) == 1
    assert controller.session.decisions["line1"].state == "pending"
    assert controller.session.reviewed.temperature is None
    assert all(controller.session.fits[line] is fit for line, fit in old.items() if line != "line1")


def test_flagged_measurement_requires_explicit_accept_button(controller):
    click(controller.widgets["inspect_line4"])
    assert controller.selected_line == "line4"
    assert controller.session.fits["line4"].flags
    assert controller.widgets["accept_line"].label == "Accept flagged fit"
    assert controller.session.reviewed.temperature is None
    controller.widgets["review_note"].value = "Inspected invented neighbour; deliberate example."
    click(controller.widgets["accept_line"])
    assert controller.session.decisions["line4"].state == "accepted"
    assert controller.session.fits["line4"].flags
    assert controller.session.reviewed.included_line_ids == ("line4",)
    click(controller.widgets["reject_line"])
    assert controller.session.decisions["line4"].state == "rejected"
    assert controller.session.reviewed.temperature is None


def test_invalid_visible_controls_clear_acceptance_and_block_save(controller, tmp_path):
    controller.decide_selected("accepted")
    original = controller.session.fits["line1"]
    controller.widgets["window_lower"].value = "not a wavelength"
    assert controller.session.fits["line1"] is original
    assert controller.session.decisions["line1"].state == "pending"
    assert controller.session.reviewed.temperature is None
    assert controller.widgets["accept_line"].disabled
    click(controller.widgets["save_review"])
    assert not (tmp_path / "review.json").exists()
    assert "draft fitting changes" in controller.message.text
    controller.widgets["line_selector"].value = "line2"
    controller.widgets["line_selector"].value = "line1"
    assert controller.widgets["window_lower"].value == "not a wavelength"
    assert controller.widgets["window_upper"].value == repr(original.window[1])
    click(controller.widgets["reset_line"])
    assert not controller.widgets["accept_line"].disabled
    assert controller.session.decisions["line1"].state == "pending"
    assert controller.session.controls["line1"].window is None


def test_uncommitted_draft_does_not_blank_other_fields_on_navigation(controller):
    expected = {name: controller.widgets[name].value for name in (
        "window_upper", "initial_amplitude", "initial_centre", "initial_sigma",
    )}
    controller._field_draft_changed("window_lower", "bad draft")
    controller.widgets["line_selector"].value = "line2"
    controller.widgets["line_selector"].value = "line1"
    assert controller.widgets["window_lower"].value == "bad draft"
    assert all(controller.widgets[name].value == value for name, value in expected.items())


def test_mouse_mask_numeric_edit_and_remove_reuse_original_pixels(controller):
    controller.widgets["line_selector"].value = "line4"
    original = controller.session.fits["line4"]
    wavelength = controller.session.spectrum.wavelength_air_angstrom.copy()
    flux = controller.session.spectrum.flux.copy()
    rest = original.rest_wavelength_air_angstrom
    geometry = {"type": "rect", "x0": rest+0.4, "x1": rest+1.1, "y0": 0, "y1": 1}
    controller._mask_geometry(SelectionGeometry(model=controller.detail_plot.profile,
                                               geometry=geometry, final=False))
    assert not controller.session.controls["line4"].excluded_regions
    controller._mask_geometry(SelectionGeometry(model=controller.detail_plot.profile,
                                               geometry=geometry, final=True))
    assert controller.session.controls["line4"].excluded_regions == ((rest+0.4, rest+1.1),)
    assert controller.session.fits["line4"].n_excluded > 0
    assert controller.session.fits["line4"].area_angstrom != pytest.approx(original.area_angstrom)
    assert controller.session.fits["line4"].window_provisional
    controller.widgets["mask_lower"].value = str(rest+0.45)
    assert controller.session.controls["line4"].excluded_regions[0][0] == rest+0.45
    click(controller.widgets["mask_remove"])
    assert not controller.session.controls["line4"].excluded_regions
    assert controller.session.fits["line4"].area_angstrom == pytest.approx(
        original.area_angstrom, abs=1e-12,
    )
    assert (controller.session.spectrum.wavelength_air_angstrom == wavelength).all()
    assert (controller.session.spectrum.flux == flux).all()


def test_failed_or_unavailable_fit_cannot_be_accepted(tmp_path):
    controller = build_review_document(Document(), session_for("missing"),
                                       review_path=tmp_path / "review.json")
    controller.widgets["line_selector"].value = "line3"
    assert controller.widgets["accept_line"].disabled
    controller.decide_selected("accepted")
    assert controller.session.decisions["line3"].state == "pending"
    assert controller.session.reviewed.temperature is None


def test_invalid_mask_bounds_survive_navigation_and_block_saving(controller, tmp_path):
    controller.widgets["line_selector"].value = "line4"
    rest = controller.session.fits["line4"].rest_wavelength_air_angstrom
    controller.widgets["mask_lower"].value = str(rest+0.4)
    controller.widgets["mask_upper"].value = str(rest+1.1)
    click(controller.widgets["mask_add"])
    controller.decide_selected("accepted")
    controller.widgets["mask_lower"].value = "invalid mask boundary"
    assert controller.session.decisions["line4"].state == "pending"
    assert controller.widgets["accept_line"].disabled
    controller.widgets["line_selector"].value = "line2"
    controller.widgets["line_selector"].value = "line4"
    assert controller.widgets["mask_lower"].value == "invalid mask boundary"
    assert controller.widgets["mask_upper"].value == repr(rest+1.1)
    click(controller.widgets["save_review"])
    assert not (tmp_path / "review.json").exists()
    controller.widgets["mask_lower"].value = str(rest+0.45)
    assert not controller.widgets["accept_line"].disabled
    assert controller.session.controls["line4"].excluded_regions == ((rest+0.45, rest+1.1),)


def test_multiple_numeric_masks_keep_the_selected_interval(controller):
    controller.widgets["line_selector"].value = "line4"
    rest = controller.session.fits["line4"].rest_wavelength_air_angstrom
    for lo, hi in ((rest+0.4, rest+1.1), (rest-1.3, rest-1.2)):
        controller.widgets["mask_selector"].value = ""
        controller.widgets["mask_lower"].value = str(lo)
        controller.widgets["mask_upper"].value = str(hi)
        click(controller.widgets["mask_add"])
    assert len(controller.session.controls["line4"].excluded_regions) == 2
    assert controller.widgets["mask_selector"].value == "1"
    controller.widgets["mask_upper"].value = str(rest-1.15)
    assert controller.widgets["mask_selector"].value == "1"
    assert controller.session.controls["line4"].excluded_regions[1] == (rest-1.3, rest-1.15)


def test_save_restore_updates_calibration_header_and_export_source_protection(controller, tmp_path):
    different = session_for(configuration="nandakumar_no_feh", feh=None)
    different.decide("line1", "accepted", note="Analytical restored example")
    source = tmp_path / "loaded-review.json"
    different.save(source)
    controller.widgets["review_path"].value = str(source)
    click(controller.widgets["load_review"])
    assert controller.session.configuration == "nandakumar_no_feh"
    assert "nandakumar_no_feh" in controller.title.text
    assert "nandakumar_feh" not in controller.title.text
    assert controller.session.decisions["line1"].state == "accepted"
    assert source.resolve() in controller.export_protected_paths
    controller.widgets["review_path"].value = str(tmp_path / "different-name.json")
    assert source.resolve() in controller.export_protected_paths


def test_html_labels_and_status_escape_input_text():
    controller = build_review_document(
        Document(), session_for(), spectrum_label="<script>bad</script>",
    )
    assert "<script>bad" not in controller.title.text
    assert "&lt;script&gt;bad" in controller.title.text


def test_new_tabs_are_isolated_and_start_from_the_latest_saved_review(tmp_path):
    server = create_inspection_server(session_for(), review_path=tmp_path / "review.json")
    try:
        application = server._tornado.applications["/"].application
        first = application.create_document()
        click(first.get_model_by_name("accept_line"))
        assert "Accepted lines (1)" in first.get_model_by_name("reviewed_summary").text
        separate = application.create_document()
        assert "Missing" in separate.get_model_by_name("reviewed_summary").text
        click(first.get_model_by_name("save_review"))
        restored = application.create_document()
        assert "Accepted lines (1)" in restored.get_model_by_name("reviewed_summary").text
        assert "Missing" in separate.get_model_by_name("reviewed_summary").text
    finally:
        server.stop()
        server.io_loop.close(all_fds=True)


@pytest.mark.parametrize("port", [-1, 65536, True, 1.5, "5006"])
def test_server_rejects_invalid_port(port):
    with pytest.raises(ValueError, match="port"):
        create_inspection_server(session_for(), port=port)


def test_loopback_server_starts_serves_local_assets_and_closes(monkeypatch):
    monkeypatch.setenv("BOKEH_RESOURCES", "cdn")
    ready = queue.Queue()
    exceptions = []

    def run():
        server = None
        try:
            server = create_inspection_server(
                session_for(), spectrum_label="Analytical server test",
            )
            server.start()
            ready.put(server)
            server.io_loop.start()
        except Exception as exc:
            exceptions.append(exc)
            ready.put(None)
        finally:
            if server is not None:
                server.stop()
                server.io_loop.close(all_fds=True)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    server = ready.get(timeout=15)
    assert server is not None, exceptions
    try:
        assert server.address == "127.0.0.1"
        assert server._tornado.resources().mode == "server"
        assert server._tornado.websocket_origins == {f"127.0.0.1:{server.port}"}
        url = f"http://127.0.0.1:{server.port}/"
        with urllib.request.urlopen(url, timeout=10) as response:
            content = response.read().decode()
        assert "/static/js/bokeh" in content
        assert "cdn.bokeh.org" not in content
        with urllib.request.urlopen(url + "static/js/bokeh.min.js", timeout=10) as response:
            assert response.status == 200
        client = pull_session(url=url)
        assert client.document.get_model_by_name("line_selector") is not None
        assert client.document.get_model_by_name("reviewed_summary") is not None
        client.close()
    finally:
        server.io_loop.add_callback(server.io_loop.stop)
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert not exceptions


def test_manual_constant_correction_updates_only_selected_line_and_review(controller):
    import numpy as np

    original_flux = controller.session.spectrum.flux.copy()
    original_fits = dict(controller.session.fits)
    controller.decide_selected("accepted")
    controller.widgets["continuum_level"].value = "1.035"
    controller.widgets["continuum_enabled"].active = [0]
    correction = controller.session.controls["line1"].continuum
    assert correction.enabled and correction.mode == "constant"
    assert correction.level == 1.035
    assert controller.session.decisions["line1"].state == "pending"
    assert controller.session.reviewed.temperature is None
    assert all(controller.session.fits[line] is fit for line, fit in original_fits.items()
               if line != "line1")
    np.testing.assert_array_equal(controller.session.spectrum.flux, original_flux)
    assert "Before correction:" in controller.details.text
    assert "Current:" in controller.details.text
    assert controller.detail_plot.continuum.data["y"][0] == 1.035
    assert "continuum-placement uncertainty" in controller.details.text
    click(controller.widgets["accept_line"])
    assert controller.session.decisions["line1"].state == "accepted"
    controller.widgets["continuum_enabled"].active = []
    assert controller.session.decisions["line1"].state == "pending"
    assert controller.session.fits["line1"].area_angstrom == pytest.approx(
        original_fits["line1"].area_angstrom, abs=1e-12,
    )
    assert controller.session.controls["line1"].continuum.level == 1.035


def test_linear_mouse_anchors_are_manual_and_residuals_use_corrected_data(controller):
    import numpy as np
    from bokeh.events import Tap

    controller.widgets["continuum_mouse"].value = "anchor1"
    rest = controller.session.fits["line1"].rest_wavelength_air_angstrom
    controller.doc.callbacks.trigger_event(Tap(model=controller.detail_plot.profile,
                                               x=rest-2.8, y=1.015, sx=10, sy=10))
    controller.widgets["continuum_mouse"].value = "anchor2"
    controller.doc.callbacks.trigger_event(Tap(model=controller.detail_plot.profile,
                                               x=rest+2.8, y=1.045, sx=10, sy=10))
    correction = controller.session.controls["line1"].continuum
    assert correction.anchors == ((rest-2.8, 1.015), (rest+2.8, 1.045))
    assert correction.enabled and correction.mode == "linear"
    assert "continuum_anchor_extrapolation" in controller.session.fits["line1"].flags
    plots = controller.detail_plot
    data_x = np.asarray(plots.used.data["x"])
    expected_c = 1.015 + (data_x-(rest-2.8))*(1.045-1.015)/5.6
    spectrum = controller.session.spectrum
    indices = np.searchsorted(spectrum.wavelength_air_angstrom, data_x)
    expected_y = spectrum.flux[indices]/expected_c
    np.testing.assert_allclose(plots.used.data["y"], expected_y, atol=1e-13)
    from sc_teff_method.fitting import gaussian_absorption
    np.testing.assert_allclose(
        plots.residual_used.data["y"],
        expected_y-gaussian_absorption(data_x, controller.session.fits["line1"].parameters),
        atol=1e-13,
    )
    assert len(plots.anchors.data["x"]) == 2
    # Taps may move anchors; they never silently add exclusion intervals.
    assert not controller.session.controls["line1"].excluded_regions
    controller.widgets["continuum_anchor2_y"].value = "1.025"
    assert controller.session.controls["line1"].continuum.anchors[1][1] == 1.025


@pytest.mark.parametrize("bad", ["0", "-0.2", "nan", "inf", "not a continuum"])
def test_invalid_continuum_draft_invalidates_approval_survives_navigation_and_resets(
    controller, tmp_path, bad,
):
    controller.widgets["continuum_enabled"].active = [0]
    controller.decide_selected("accepted")
    previous = controller.session.fits["line1"]
    controller.widgets["continuum_level"].value = bad
    assert controller.session.fits["line1"] is previous
    assert controller.session.decisions["line1"].state == "pending"
    assert controller.widgets["accept_line"].disabled
    controller.widgets["line_selector"].value = "line2"
    controller.widgets["line_selector"].value = "line1"
    assert controller.widgets["continuum_level"].value == bad
    click(controller.widgets["save_review"])
    assert not (tmp_path / "review.json").exists()
    click(controller.widgets["reset_continuum"])
    assert not controller.widgets["accept_line"].disabled
    assert not controller.session.controls["line1"].continuum.enabled
    assert controller.session.decisions["line1"].state == "pending"


def test_remove_replace_anchor_and_restore_correction_review(controller, tmp_path):
    from bokeh.events import Tap

    controller.widgets["continuum_mode"].value = "linear"
    controller.widgets["continuum_enabled"].active = [0]
    controller.widgets["continuum_anchor2_y"].value = "1.02"
    controller.decide_selected("accepted")
    click(controller.widgets["continuum_remove_anchor2"])
    assert controller.widgets["accept_line"].disabled
    assert controller.session.decisions["line1"].state == "pending"
    assert controller.widgets["continuum_anchor2_x"].value == ""
    controller.widgets["continuum_mouse"].value = "anchor2"
    rest = controller.session.fits["line1"].rest_wavelength_air_angstrom
    controller.doc.callbacks.trigger_event(Tap(model=controller.detail_plot.profile,
                                               x=rest+3.0, y=1.03, sx=10, sy=10))
    assert not controller.widgets["accept_line"].disabled
    controller.decide_selected("accepted")
    before = controller.session.fits["line1"]
    click(controller.widgets["save_review"])
    assert (tmp_path / "review.json").exists()
    click(controller.widgets["reset_continuum"])
    click(controller.widgets["load_review"])
    assert controller.session.fits["line1"].area_angstrom == before.area_angstrom
    assert controller.session.decisions["line1"].state == "accepted"
    assert controller.widgets["continuum_enabled"].active == [0]
    assert controller.widgets["continuum_anchor2_y"].value == "1.03"


def test_disable_invalid_anchor_draft_restores_fitting_without_changing_original(controller):
    controller.widgets["continuum_mode"].value = "linear"
    controller.widgets["continuum_enabled"].active = [0]
    controller.decide_selected("accepted")
    controller.widgets["continuum_anchor2_y"].value = "0"
    assert controller.widgets["accept_line"].disabled
    controller.widgets["continuum_enabled"].active = []
    assert not controller.widgets["accept_line"].disabled
    assert not controller.session.controls["line1"].continuum.enabled
    assert controller.session.decisions["line1"].state == "pending"
    assert "refitted" in controller.message.text


def test_continuum_reset_preserves_mask_and_window(controller):
    rest = controller.session.fits["line1"].rest_wavelength_air_angstrom
    controller.widgets["window_lower"].value = str(rest-2.8)
    controller.widgets["mask_lower"].value = str(rest+1.2)
    controller.widgets["mask_upper"].value = str(rest+1.7)
    click(controller.widgets["mask_add"])
    before = controller.session.controls["line1"]
    controller.widgets["continuum_level"].value = "1.035"
    controller.widgets["continuum_enabled"].active = [0]
    click(controller.widgets["reset_continuum"])
    after = controller.session.controls["line1"]
    assert after.window == before.window
    assert after.excluded_regions == before.excluded_regions
    assert after.initial == before.initial
    assert not after.continuum.enabled


def test_late_continuum_value_input_event_does_not_reinvalidate_a_committed_edit(controller):
    controller.widgets["continuum_enabled"].active = [0]
    controller.widgets["continuum_level"].value = "1.025"
    assert not controller.widgets["accept_line"].disabled
    controller.decide_selected("accepted")
    # Bokeh may deliver a last value_input update after the committed value event.
    controller._continuum_draft_callback("continuum_level")("value_input", "1.0", "1.025")
    assert controller.session.decisions["line1"].state == "accepted"
    assert not controller.widgets["accept_line"].disabled
    # A different visible draft still invalidates scientific approval immediately.
    controller._continuum_draft_callback("continuum_level")("value_input", "1.025", "0")
    assert controller.session.decisions["line1"].state == "pending"
    assert controller.widgets["accept_line"].disabled


def test_anchor_commit_preserves_another_pending_field_and_blocks_approval(controller, tmp_path):
    controller.widgets["line_selector"].value = "line3"
    controller.widgets["continuum_mode"].value = "linear"
    controller.widgets["continuum_enabled"].active = [0]
    controller.decide_selected("accepted")
    anchor2 = controller.widgets["continuum_anchor2_y"]
    anchor2.value_input = "1.025"
    controller.widgets["continuum_anchor1_y"].value = "1.005"
    # The first server refit must not rewrite the other TextInput or lose its draft.
    assert anchor2.value == "1.0"
    assert anchor2.value_input == "1.025"
    assert controller._continuum_drafts["line3"]["continuum_anchor2_y"] == "1.025"
    assert controller.widgets["accept_line"].disabled
    click(controller.widgets["save_review"])
    assert not (tmp_path / "review.json").exists()
    anchor2.value = "1.025"
    assert not controller.widgets["accept_line"].disabled
    assert controller.session.controls["line3"].continuum.anchors[0][1] == 1.005
    assert controller.session.controls["line3"].continuum.anchors[1][1] == 1.025
    assert "line3" not in controller._continuum_drafts


def test_other_control_commit_preserves_pending_continuum_and_window_drafts(controller):
    controller.widgets["continuum_enabled"].active = [0]
    controller.widgets["continuum_level"].value_input = "1.045"
    before = controller.session.fits["line1"]
    controller.widgets["window_lower"].value = str(before.window[0]+0.1)
    assert controller.widgets["continuum_level"].value == "1.0"
    assert controller.widgets["continuum_level"].value_input == "1.045"
    assert controller.widgets["accept_line"].disabled
    controller.widgets["continuum_level"].value = "1.045"
    assert not controller.widgets["accept_line"].disabled
    next_lower = str(before.window[0]+0.2)
    controller.widgets["window_lower"].value_input = next_lower
    controller.widgets["continuum_level"].value = "1.035"
    assert controller._drafts["line1"]["window_lower"] == next_lower
    assert controller.widgets["accept_line"].disabled
    controller.widgets["line_selector"].value = "line2"
    controller.widgets["line_selector"].value = "line1"
    assert controller.widgets["window_lower"].value == next_lower
    click(controller.widgets["refit_line"])
    assert not controller.widgets["accept_line"].disabled
    assert controller.session.fits["line1"].window[0] == float(next_lower)


def test_mask_commit_keeps_uncommitted_continuum_draft(controller):
    rest = controller.session.fits["line1"].rest_wavelength_air_angstrom
    controller.widgets["mask_lower"].value = str(rest+1.1)
    controller.widgets["mask_upper"].value = str(rest+1.6)
    click(controller.widgets["mask_add"])
    controller.widgets["continuum_level"].value_input = "1.025"
    controller.widgets["mask_upper"].value = str(rest+1.7)
    assert controller._continuum_drafts["line1"]["continuum_level"] == "1.025"
    assert controller.widgets["accept_line"].disabled
    controller.widgets["continuum_level"].value = "1.025"
    assert not controller.widgets["accept_line"].disabled
    assert controller.session.controls["line1"].excluded_regions[0] == (rest+1.1, rest+1.7)


def test_older_commit_cannot_clear_a_newer_draft_in_the_same_field(controller):
    field = controller.widgets["continuum_level"]
    field.value_input = "1.025"
    # Batched browser events may deliver a newer input before an older blur commit.
    field.value = "1.02"
    assert controller._pending_inputs["line1"]["continuum_level"] == "1.025"
    assert controller._continuum_drafts["line1"]["continuum_level"] == "1.025"
    assert controller.widgets["accept_line"].disabled
    field.value = "1.025"
    assert not controller.widgets["accept_line"].disabled
    assert controller.session.controls["line1"].continuum.level == 1.025
