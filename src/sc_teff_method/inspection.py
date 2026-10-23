"""Local Bokeh views over the existing Gaussian fitter and manual review model.

Only loopback traffic is used. Plot resources are served by the local Bokeh
server; this module contains no spectrum, observational template or remote URL.
"""

from __future__ import annotations

import html
import math
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import numpy as np

try:
    from bokeh.application import Application
    from bokeh.application.handlers.function import FunctionHandler
    from bokeh.events import SelectionGeometry, Tap
    from bokeh.layouts import column, row
    from bokeh.models import (
        BasicTicker,
        BasicTickFormatter,
        BoxSelectTool,
        Button,
        CheckboxGroup,
        ColumnDataSource,
        Div,
        PanTool,
        PrintfTickFormatter,
        Range1d,
        ResetTool,
        Select,
        Span,
        TextAreaInput,
        TextInput,
        WheelZoomTool,
    )
    from bokeh.plotting import figure
    from bokeh.resources import Resources
    from bokeh.server.server import BaseServer
    from bokeh.server.tornado import BokehTornado
    from bokeh.server.util import bind_sockets
    from tornado.httpserver import HTTPServer
    from tornado.ioloop import IOLoop
except ImportError as exc:
    raise ImportError("Interactive inspection requires 'sc-teff-method[inspect]'") from exc

from ._data import LINE_IDS
from .api import estimate_temperature
from .continuum import ContinuumCorrection, evaluate_continuum
from .fitting import GaussianParameters, gaussian_absorption
from .review import ReviewSession

_COLORS = {"pending": "#a26a08", "accepted": "#17754d", "rejected": "#a63635"}
_FIELD_NAMES = (
    "window_lower", "window_upper", "initial_amplitude", "initial_centre", "initial_sigma",
)
_CONTINUUM_FIELDS = (
    "continuum_level", "continuum_anchor1_x", "continuum_anchor1_y",
    "continuum_anchor2_x", "continuum_anchor2_y",
)


def _escape(value) -> str:
    return html.escape(str(value), quote=True)


def _number(value, precision: int = 6) -> str:
    return "missing" if value is None else f"{value:.{precision}g}"


def _named(model, name: str):
    model.name = name
    model.css_classes = [name]
    return model


class _RegionPlot:
    """Linked profile/residual plots, using PR 2's Gaussian model for curves."""

    def __init__(self, width: int, *, detail: bool = False, name: str):
        self.detail = detail
        self.native = ColumnDataSource({"x": [], "y": []})
        self.used = ColumnDataSource({"x": [], "y": []})
        self.removed = ColumnDataSource({"x": [], "y": []})
        self.model = ColumnDataSource({"x": [], "y": []})
        self.continuum = ColumnDataSource({"x": [], "y": []})
        self.anchors = ColumnDataSource({"x": [], "y": []})
        self.residual_used = ColumnDataSource({"x": [], "y": []})
        self.residual_removed = ColumnDataSource({"x": [], "y": []})
        self.shading = ColumnDataSource({"left": [], "right": [], "top": [], "bottom": []})
        x_range = Range1d(0, 1)
        self.mask_tool = BoxSelectTool(dimensions="width") if detail else None
        tools = ([PanTool(dimensions="width"), self.mask_tool,
                  WheelZoomTool(dimensions="width"), ResetTool()] if detail else "tap")
        self.profile = figure(
            width=width, height=215 if detail else 123, x_range=x_range,
            tools=tools, toolbar_location="above" if detail else None,
            active_drag=self.mask_tool if detail else "auto", name=name + "_profile",
            min_border_left=39, min_border_right=8, min_border_top=5, min_border_bottom=0,
            output_backend="canvas",
        )
        self.profile.css_classes = [self.profile.name]
        self.profile.quad(left="left", right="right", top="top", bottom="bottom",
                          source=self.shading, fill_color="#d4514b", fill_alpha=0.08,
                          line_alpha=0)
        self.profile.line("x", "y", source=self.native, color="#a9afb6", line_alpha=0.7)
        self.profile.scatter("x", "y", source=self.native, color="#a9afb6", size=3)
        used = self.profile.scatter("x", "y", source=self.used, color="#247397", size=3)
        self.profile.scatter("x", "y", source=self.removed, color="#bf3f39", marker="x", size=6)
        self.profile.line("x", "y", source=self.model, color="#d8871d", line_width=2)
        self.profile.line("x", "y", source=self.continuum, color="#33845a",
                          line_width=2, line_dash="dashed")
        self.profile.scatter("x", "y", source=self.anchors, color="#33845a",
                             marker="diamond", size=10 if detail else 6)
        if self.mask_tool is not None:
            self.mask_tool.renderers = [used]
        self.residual = figure(
            width=width, height=85 if detail else 65, x_range=x_range, tools="",
            toolbar_location=None, name=name + "_residual",
            min_border_left=46, min_border_right=8, min_border_top=3, min_border_bottom=23,
            output_backend="canvas",
        )
        self.residual.css_classes = [self.residual.name]
        self.residual.scatter("x", "y", source=self.residual_used, color="#247397", size=3)
        self.residual.scatter("x", "y", source=self.residual_removed,
                              color="#bf3f39", marker="x", size=5)
        self.residual.add_layout(Span(location=0, dimension="width", line_color="#a0a7ae"))
        self.spans = []
        for plot in (self.profile, self.residual):
            rest = Span(location=0, dimension="height", line_color="#626b75",
                        line_dash="dashed", line_width=1)
            centre = Span(location=0, dimension="height", line_color="#d8871d", line_width=1)
            lower = Span(location=0, dimension="height", line_color="#8d6ba3", line_dash="dotted")
            upper = Span(location=0, dimension="height", line_color="#8d6ba3", line_dash="dotted")
            for span in (rest, centre, lower, upper):
                plot.add_layout(span)
            self.spans.append((rest, centre, lower, upper))
            plot.xaxis.formatter = PrintfTickFormatter(format="%.1f" if detail else "%.0f")
            plot.xaxis.ticker = BasicTicker(desired_num_ticks=4 if detail else 3,
                                            num_minor_ticks=0)
            plot.yaxis.ticker = BasicTicker(desired_num_ticks=3, num_minor_ticks=0)
            plot.xaxis.major_label_text_font_size = "8pt"
            plot.yaxis.major_label_text_font_size = "8pt"
            plot.axis.axis_label_text_font_size = "8pt"
            plot.grid.grid_line_alpha = 0.20
            plot.outline_line_color = "#d4dbe1"
        self.profile.yaxis.formatter = BasicTickFormatter(use_scientific=False, precision=2)
        self.residual.yaxis.ticker = BasicTicker(desired_num_ticks=2, num_minor_ticks=0)
        self.residual.yaxis.formatter = BasicTickFormatter(
            use_scientific=True, power_limit_low=-3, power_limit_high=4,
        )
        self.profile.add_layout(Span(location=1, dimension="width", line_color="#b7bdc4",
                                     line_dash="dotted"))
        self.profile.xaxis.visible = False
        self.profile.yaxis.axis_label = "Flux" if detail else None
        self.residual.yaxis.axis_label = "F−model" if detail else None
        self.residual.xaxis.axis_label = "Rest air wavelength (Å)" if detail else None
        self.layout = column(self.profile, self.residual, spacing=1, margin=0)

    def update(self, spectrum, fit, corrected=None, correction=None) -> None:
        x = spectrum.wavelength_air_angstrom
        y = spectrum.flux if corrected is None else corrected.flux
        original_y = spectrum.flux
        lower, upper = fit.window
        padding = 0.12 * (upper - lower)
        region = (x >= lower-padding) & (x <= upper+padding)
        finite = region & np.isfinite(original_y)
        masked = np.array(spectrum.mask, copy=True)
        for lo, hi in fit.excluded_regions:
            masked |= (x >= lo) & (x <= hi)
        used = region & (x >= lower) & (x <= upper) & ~masked & np.isfinite(y)
        removed = region & (x >= lower) & (x <= upper) & masked
        if fit.status == "excluded":
            removed |= region & (x >= lower) & (x <= upper)
            used[:] = False
        self.native.data = {"x": x[finite], "y": original_y[finite]}
        self.used.data = {"x": x[used], "y": y[used]}
        self.removed.data = {"x": x[removed], "y": np.where(np.isfinite(y[removed]), y[removed], 1)}
        values = np.concatenate((y[region & np.isfinite(y)], original_y[finite]))
        if correction is not None and correction.enabled and correction.mode != "none":
            continuum_x = np.linspace(lower, upper, 150)
            continuum_y = evaluate_continuum(continuum_x, correction)
            self.continuum.data = {"x": continuum_x, "y": continuum_y}
            self.anchors.data = {
                "x": [point[0] for point in correction.anchors],
                "y": [point[1] for point in correction.anchors],
            } if correction.mode == "linear" else {"x": [], "y": []}
            values = np.concatenate((values, continuum_y[np.isfinite(continuum_y)]))
        else:
            self.continuum.data = {"x": [], "y": []}
            self.anchors.data = {"x": [], "y": []}
        ymin = min(float(values.min()), 1.0)-0.05 if values.size else 0.0
        ymax = max(float(values.max()), 1.0)+0.05 if values.size else 1.1
        self.shading.data = {
            "left": [r[0] for r in fit.excluded_regions],
            "right": [r[1] for r in fit.excluded_regions],
            "top": [ymax] * len(fit.excluded_regions),
            "bottom": [ymin] * len(fit.excluded_regions),
        }
        self.profile.x_range.start, self.profile.x_range.end = lower-padding, upper+padding
        if fit.parameters is not None:
            model_x = np.linspace(lower, upper, 350)
            self.model.data = {"x": model_x, "y": gaussian_absorption(model_x, fit.parameters)}
            self.residual_used.data = {
                "x": x[used], "y": y[used]-gaussian_absorption(x[used], fit.parameters),
            }
            keep_removed = removed & np.isfinite(y)
            self.residual_removed.data = {
                "x": x[keep_removed],
                "y": y[keep_removed]-gaussian_absorption(x[keep_removed], fit.parameters),
            }
        else:
            self.model.data = {"x": [], "y": []}
            self.residual_used.data = {"x": [], "y": []}
            self.residual_removed.data = {"x": [], "y": []}
        for rest, centre, left, right in self.spans:
            rest.location = fit.rest_wavelength_air_angstrom
            centre.visible = fit.parameters is not None
            if fit.parameters is not None:
                centre.location = fit.parameters.centre_air_angstrom
            left.location, right.location = lower, upper


class ReviewDocument:
    """Bokeh callbacks adapting committed controls to one ReviewSession."""

    def __init__(self, doc, session: ReviewSession, *, review_path: str | Path,
                 output_dir: str | Path, protected_paths: Iterable[str | Path],
                 export_protected_paths: Iterable[str | Path], spectrum_label: str):
        self.doc, self.session = doc, session
        self.protected_paths = tuple(Path(path).expanduser() for path in protected_paths)
        self.export_protected_paths = {
            Path(path).expanduser().resolve() for path in export_protected_paths
        }
        self.spectrum_label = spectrum_label
        self.widgets = {}
        self._loading = False
        self._drafts = {}
        self._mask_drafts = {}
        self._mask_selection = {}
        self._continuum_drafts = {}
        self._pending_inputs = {}
        self._note_errors = {}
        self._invalid_lines = set()
        self._selected = LINE_IDS[0]
        self._persisted = None
        self._plot_fits = {}
        self._detail_fit = None

        def widget(model, name):
            self.widgets[name] = _named(model, name)
            return model

        self.title = Div(text="", width=1400)
        self.summary = widget(Div(width=1400, styles={
            "background-color": "#edf5f8", "padding": "11px", "border-radius": "5px",
        }), "reviewed_summary")
        caution = Div(text=(
            "<b>Scientific review is required.</b> Convergence is not acceptance. "
            "Only explicitly accepted valid fits enter the reviewed temperature. "
            "Frozen subset RMSE does not validate post-hoc line selection "
            "or newly chosen continuum corrections."
        ), width=1400, styles={"color": "#725619", "font-size": "12px"})
        legend = Div(text=(
            "<span style='color:#a9afb6'>— Original input</span> · "
            "<span style='color:#247397'>● Corrected fitting pixels</span> · "
            "<span style='color:#33845a'>-- User continuum / ◆ anchors</span> · "
            "<span style='color:#d8871d'>— Gaussian / fitted centre</span> · "
            "grey dashed: laboratory centre · purple dotted: window · "
            "<span style='color:#bf3f39'>× excluded pixels</span> · Gaussian continuum fixed at 1"
        ), width=1400, styles={"font-size": "11px"})
        self.message = widget(Div(text="", width=1400), "review_message")
        self.scientific_warnings = widget(Div(width=1400), "scientific_warnings")
        self.overview = {}
        cards = []
        for line in LINE_IDS:
            button = widget(Button(label=line, width=130, height=25), "inspect_" + line)
            rest = session.fits[line].rest_wavelength_air_angstrom
            line_label = Div(text=f"{rest:.3f} Å", width=150, styles={"font-size": "11px"})
            facts = Div(width=300, styles={"font-size": "11px", "line-height": "1.25"})
            flags = Div(width=300, styles={"font-size": "10px", "line-height": "1.2"})
            plots = _RegionPlot(305, name="overview_" + line)
            card = column(row(button, line_label, spacing=5), facts, plots.layout, flags,
                          width=310, spacing=2, margin=(0, 0, 8, 0),
                          styles={"border-bottom": "1px solid #d4dbe1", "padding-bottom": "5px"})
            self.overview[line] = (button, facts, flags, plots, card)
            cards.append(card)

            def select(event=None, selected=line):
                self.widgets["line_selector"].value = selected

            button.on_click(select)
            plots.profile.on_event(Tap, select)
        overview_grid = column(*(row(*cards[i:i+3], spacing=8) for i in range(0, 9, 3)),
                               spacing=3, width=945)
        selector = widget(Select(title="Detailed line", options=list(LINE_IDS), value=LINE_IDS[0],
                                 width=440), "line_selector")
        self.detail_plot = _RegionPlot(450, detail=True, name="selected")
        detail_hint = Div(text=(
            "Drag horizontally to exclude pixels. Select a continuum mouse action below "
            "and tap the plot to set an anchor or constant level. All coordinates are rest air Å. "
            "Changes refit this line only from the original spectrum."
        ), width=440, styles={"font-size": "11px", "color": "#526171"})
        self.details = widget(Div(width=440, styles={"font-size": "12px"}), "fit_details")
        lower = widget(TextInput(title="Window lower (Å)", width=215), "window_lower")
        upper = widget(TextInput(title="Window upper (Å)", width=215), "window_upper")
        automatic = widget(CheckboxGroup(labels=["Automatic Gaussian initial guess"], active=[0],
                                          width=440), "automatic_initial")
        amp = widget(TextInput(title="Initial depth A", width=130), "initial_amplitude")
        centre = widget(TextInput(title="Initial centre (Å)", width=170), "initial_centre")
        sigma = widget(TextInput(title="Initial sigma (Å)", width=130), "initial_sigma")
        refit = widget(Button(label="Refit line", width=215), "refit_line")
        reset = widget(Button(label="Reset line defaults", width=215), "reset_line")
        masks = widget(Select(title="Excluded intervals", options=[], value="", width=440),
                       "mask_selector")
        mask_lower = widget(TextInput(title="Mask lower (Å)", width=215), "mask_lower")
        mask_upper = widget(TextInput(title="Mask upper (Å)", width=215), "mask_upper")
        add = widget(Button(label="Add interval", width=140), "mask_add")
        update = widget(Button(label="Update interval", width=140), "mask_update")
        remove = widget(Button(label="Remove interval", width=140), "mask_remove")
        continuum_title = Div(text="<b>Manual local continuum</b> · flux / C(λ)", width=440)
        continuum_enabled = widget(CheckboxGroup(labels=["Enable user continuum correction"],
                                                  active=[], width=440), "continuum_enabled")
        continuum_mode = widget(Select(title="Correction mode", options=["constant", "linear"],
                                       value="constant", width=215), "continuum_mode")
        continuum_level = widget(TextInput(title="Constant continuum level C", value="1.0",
                                           width=215), "continuum_level")
        anchor1_x = widget(TextInput(title="Anchor 1 wavelength (Å)", width=215),
                           "continuum_anchor1_x")
        anchor1_y = widget(TextInput(title="Anchor 1 continuum C", width=215),
                           "continuum_anchor1_y")
        anchor2_x = widget(TextInput(title="Anchor 2 wavelength (Å)", width=215),
                           "continuum_anchor2_x")
        anchor2_y = widget(TextInput(title="Anchor 2 continuum C", width=215),
                           "continuum_anchor2_y")
        continuum_mouse = widget(Select(title="Mouse action in detailed profile",
                                        options=[("mask", "Drag: exclude pixels"),
                                                 ("constant", "Tap: set constant level"),
                                                 ("anchor1", "Tap: add / move anchor 1"),
                                                 ("anchor2", "Tap: add / move anchor 2")],
                                        value="mask", width=440), "continuum_mouse")
        remove_anchor1 = widget(Button(label="Remove anchor 1", width=140),
                                "continuum_remove_anchor1")
        remove_anchor2 = widget(Button(label="Remove anchor 2", width=140),
                                "continuum_remove_anchor2")
        reset_continuum = widget(Button(label="Reset to unity", width=140), "reset_continuum")
        continuum_hint = Div(text=(
            "Anchors and levels are entirely user chosen; no continuum optimisation occurs. "
            "Formal Gaussian errors exclude continuum-placement uncertainty. "
            "A better-looking fit does not validate the correction."
        ), width=440, styles={"font-size": "11px", "color": "#725619"})
        note = widget(TextInput(title="Review reason / note (optional)", width=440,
                                max_length=4000), "review_note")
        accept = widget(Button(label="Accept fit", button_type="success", width=150), "accept_line")
        reject = widget(Button(label="Reject", button_type="danger", width=140), "reject_line")
        pending = widget(Button(label="Pending", width=140), "pending_line")
        detail = column(selector, self.detail_plot.layout, detail_hint, self.details,
                        row(lower, upper, spacing=10), automatic,
                        row(amp, centre, sigma, spacing=5), row(refit, reset, spacing=10),
                        masks, row(mask_lower, mask_upper, spacing=10),
                        row(add, update, remove, spacing=5),
                        continuum_title, continuum_enabled,
                        row(continuum_mode, continuum_level, spacing=10),
                        row(anchor1_x, anchor1_y, spacing=10),
                        row(anchor2_x, anchor2_y, spacing=10), continuum_mouse,
                        row(remove_anchor1, remove_anchor2, reset_continuum, spacing=5),
                        continuum_hint, note,
                        row(accept, reject, pending, spacing=5), width=450, spacing=5)

        review = widget(TextInput(title="Review JSON path", value=str(review_path), width=400),
                        "review_path")
        save = widget(Button(label="Save review JSON", width=170), "save_review")
        load = widget(Button(label="Load same-spectrum review", width=220), "load_review")
        overwrite = widget(CheckboxGroup(labels=["Allow replacing existing review / output files"],
                                          active=[], width=400), "overwrite")
        output = widget(TextInput(title="Output directory", value=str(output_dir), width=400),
                        "output_dir")
        export = widget(Button(label="Export reviewed results", button_type="primary", width=220),
                        "export_results")
        formats = widget(Select(title="Figure format", options=["png", "pdf", "both"],
                                value="png", width=120), "plot_format")
        notes = widget(TextAreaInput(title="Session notes (saved with the review)",
                                     value=session.notes, width=1400, rows=2,
                                     max_length=10000), "review_notes")
        file_title = Div(
            text="<h3 style='margin:5px 0'>Save, restore and reproduce</h3>", width=945,
        )
        files = column(file_title,
                       row(review, save, load, spacing=10), overwrite,
                       row(output, export, formats, spacing=10), notes, spacing=6, width=1400)
        footnote = Div(text=(
            "Formal Gaussian errors exclude continuum and profile systematics. "
            "Calibration applicability depends on measurement and observational compatibility. "
            "Synthetic examples establish software correctness, not transfer accuracy. "
            "This application uses only a local loopback server; no uploads or cloud service. "
            "Each browser tab has its own review. Save before closing or reloading."
        ), width=1400, styles={"font-size": "11px", "color": "#526171"})
        root = column(self.title, self.summary, caution, legend, self.message,
                      row(overview_grid, detail, spacing=25), files,
                      self.scientific_warnings, footnote,
                      spacing=10, width=1430, margin=20, name="inspection_root")
        doc.title = "Sc I line-fitting review — local"
        selector.on_change("value", self._select_changed)
        for name in _FIELD_NAMES:
            self.widgets[name].on_change(
                "value", self._committed_callback(name, self._controls_changed)
            )
            self.widgets[name].on_change("value_input", self._draft_callback(name))
        automatic.on_change("active", self._controls_changed)
        continuum_enabled.on_change("active", self._continuum_changed)
        continuum_mode.on_change("value", self._continuum_changed)
        for name in _CONTINUUM_FIELDS:
            self.widgets[name].on_change(
                "value", self._committed_callback(name, self._continuum_changed)
            )
            self.widgets[name].on_change("value_input", self._continuum_draft_callback(name))
        continuum_mouse.on_change("value", self._continuum_mouse_changed)
        remove_anchor1.on_click(lambda: self.remove_continuum_anchor(1))
        remove_anchor2.on_click(lambda: self.remove_continuum_anchor(2))
        reset_continuum.on_click(self.reset_continuum_selected)
        masks.on_change("value", self._mask_selected)
        mask_lower.on_change(
            "value", self._committed_callback("mask_lower", self._mask_field_changed)
        )
        mask_upper.on_change(
            "value", self._committed_callback("mask_upper", self._mask_field_changed)
        )
        mask_lower.on_change("value_input", self._mask_draft_callback("lower"))
        mask_upper.on_change("value_input", self._mask_draft_callback("upper"))
        note.on_change("value", self._note_changed)
        notes.on_change("value", self._session_notes_changed)
        refit.on_click(self.apply_controls)
        reset.on_click(self.reset_selected)
        add.on_click(self.add_mask)
        update.on_click(self.update_mask)
        remove.on_click(self.remove_mask)
        accept.on_click(lambda: self.decide_selected("accepted"))
        reject.on_click(lambda: self.decide_selected("rejected"))
        pending.on_click(lambda: self.decide_selected("pending"))
        save.on_click(self.save_review)
        load.on_click(self.load_review)
        export.on_click(self.export_results)
        self.detail_plot.profile.on_event(SelectionGeometry, self._mask_geometry)
        self.detail_plot.profile.on_event(Tap, self._continuum_tap)
        self.refresh()
        # Bokeh registers model-event routes when models attach to a document.
        # All button/geometry callbacks must exist before this attachment.
        doc.add_root(root)

    @property
    def selected_line(self) -> str:
        return self.widgets["line_selector"].value

    def _error(self, exc) -> None:
        self.message.text = f"<span style='color:#a63635'><b>Not applied:</b> {_escape(exc)}</span>"

    def _success(self, text: str) -> None:
        self.message.text = f"<span style='color:#17754d'>{_escape(text)}</span>"

    def _clear_approval(self) -> None:
        line = self.selected_line
        if self.session.decisions[line].state != "pending":
            self.session.decide(line, "pending", note="")
        self._invalid_lines.add(line)
        self.refresh_results()

    def _committed_callback(self, field, handler):
        def callback(attr, old, new):
            if not self._loading:
                pending = self._pending_inputs.get(self.selected_line, {})
                if pending.get(field) == new:
                    pending.pop(field)
            handler(attr, old, new)
        return callback

    def _store_pending_drafts(self, *, clear_applied=False) -> None:
        """Keep only event-touched text drafts over current committed widget values.

        A commit to one field must never clear another field's pending edit. Empty
        value_input defaults are not drafts; only actual input events mark fields.
        """
        line = self.selected_line
        pending = self._pending_inputs.get(line, {})
        ordinary = {name: pending[name] for name in _FIELD_NAMES if name in pending}
        if ordinary:
            self._drafts[line] = {
                **{name: self.widgets[name].value for name in _FIELD_NAMES},
                "automatic_initial": bool(self.widgets["automatic_initial"].active),
                **ordinary,
            }
        elif clear_applied:
            self._drafts.pop(line, None)
        continuum = {name: pending[name] for name in _CONTINUUM_FIELDS if name in pending}
        if continuum:
            self._continuum_drafts[line] = {**self._continuum_values(), **continuum}
        elif clear_applied:
            self._continuum_drafts.pop(line, None)
        masks = {name.removeprefix("mask_"): pending[name]
                 for name in ("mask_lower", "mask_upper") if name in pending}
        if masks:
            self._mask_drafts[line] = {
                "selected": self.widgets["mask_selector"].value,
                "lower": self.widgets["mask_lower"].value,
                "upper": self.widgets["mask_upper"].value,
                **masks,
            }
        elif clear_applied:
            self._mask_drafts.pop(line, None)

    def _control_availability(self) -> None:
        automatic = bool(self.widgets["automatic_initial"].active)
        for name in ("initial_amplitude", "initial_centre", "initial_sigma"):
            self.widgets[name].disabled = automatic
        linear = self.widgets["continuum_mode"].value == "linear"
        for name in _CONTINUUM_FIELDS:
            self.widgets[name].disabled = linear if name == "continuum_level" else not linear

    def _sync_mask_selector(self, *, populate_bounds=False) -> None:
        """Update the interval list without overwriting unrelated text inputs."""
        regions = self._regions()
        selected = self._mask_selection.get(self.selected_line, "0" if regions else "")
        if selected and not 0 <= int(selected) < len(regions):
            selected = "0" if regions else ""
        self._loading = True
        self.widgets["mask_selector"].options = [("", "New interval / drag to add")] + [
            (str(i), f"{i+1}: {lo:.7f} – {hi:.7f} Å") for i, (lo, hi) in enumerate(regions)
        ]
        self.widgets["mask_selector"].value = selected
        if populate_bounds:
            self.widgets["mask_lower"].value = repr(regions[int(selected)][0]) if selected else ""
            self.widgets["mask_upper"].value = repr(regions[int(selected)][1]) if selected else ""
        self._loading = False

    def _draft_callback(self, field):
        def callback(attr, old, new):
            self._field_draft_changed(field, new)
        return callback

    def _field_draft_changed(self, field, new) -> None:
        if self._loading or new == self.widgets[field].value:
            return
        self._pending_inputs.setdefault(self.selected_line, {})[field] = new
        self._store_pending_drafts()
        self._clear_approval()

    def _controls_changed(self, attr, old, new) -> None:
        if not self._loading:
            self.apply_controls()

    def _select_changed(self, attr, old, new) -> None:
        self._selected = new
        self.message.text = ""
        self.refresh()

    def _regions(self):
        return list(self.session.controls[self.selected_line].excluded_regions)

    def _line_arguments(self):
        line = self.selected_line
        window = (float(self.widgets["window_lower"].value),
                  float(self.widgets["window_upper"].value))
        if self.session.controls[line].window is None and window == self.session.fits[line].window:
            window = None
        initial = None
        if not self.widgets["automatic_initial"].active:
            initial = GaussianParameters(
                float(self.widgets["initial_amplitude"].value),
                float(self.widgets["initial_centre"].value),
                float(self.widgets["initial_sigma"].value),
            )
        return window, initial

    def apply_controls(self, *, regions=None, continuum=None) -> bool:
        self._clear_approval()
        try:
            window, initial = self._line_arguments()
            if regions is None:
                regions = self._regions()
                draft = self._mask_drafts.get(self.selected_line)
                if draft is not None:
                    interval = (float(draft["lower"]), float(draft["upper"]))
                    if draft["selected"]:
                        regions[int(draft["selected"])] = interval
                    else:
                        regions.append(interval)
            self.session.edit_line(
                self.selected_line, window=window, initial=initial,
                excluded_regions=regions,
                continuum=self._continuum_arguments() if continuum is None else continuum,
            )
        except (ValueError, TypeError) as exc:
            self._drafts[self.selected_line] = {
                **{name: self.widgets[name].value for name in _FIELD_NAMES},
                "automatic_initial": bool(self.widgets["automatic_initial"].active),
            }
            self._store_pending_drafts()
            self._error(exc)
            self.refresh_results()
            return False
        pending = self._pending_inputs.get(self.selected_line, {})
        for field, value in tuple(pending.items()):
            if self.widgets[field].value == value:
                pending.pop(field)
        self._store_pending_drafts(clear_applied=True)
        if pending:
            self._invalid_lines.add(self.selected_line)
        else:
            self._invalid_lines.discard(self.selected_line)
        self._success("Line refitted; scientific review is pending.")
        # Refitting can finish while the browser is already editing another field.
        # Do not send unrelated TextInput.value patches that overwrite that input.
        self._control_availability()
        self._sync_mask_selector()
        self.refresh_results()
        return True

    def reset_selected(self) -> None:
        self.session.reset_line(self.selected_line)
        self._drafts.pop(self.selected_line, None)
        self._mask_drafts.pop(self.selected_line, None)
        self._mask_selection.pop(self.selected_line, None)
        self._continuum_drafts.pop(self.selected_line, None)
        self._pending_inputs.pop(self.selected_line, None)
        self._note_errors.pop(self.selected_line, None)
        self._invalid_lines.discard(self.selected_line)
        self._success("Line reset to default window, automatic start, unity continuum "
                      "and no added mask intervals.")
        self.refresh()

    def _continuum_values(self) -> dict:
        return {
            "enabled": bool(self.widgets["continuum_enabled"].active),
            "mode": self.widgets["continuum_mode"].value,
            **{name: self.widgets[name].value for name in _CONTINUUM_FIELDS},
        }

    def _continuum_arguments(self) -> ContinuumCorrection:
        values = self._continuum_values()
        if values["mode"] == "constant":
            return ContinuumCorrection(enabled=values["enabled"], mode="constant",
                                       level=float(values["continuum_level"]))
        return ContinuumCorrection(
            enabled=values["enabled"], mode="linear",
            anchors=((float(values["continuum_anchor1_x"]),
                      float(values["continuum_anchor1_y"])),
                     (float(values["continuum_anchor2_x"]),
                      float(values["continuum_anchor2_y"]))),
        )

    def _continuum_draft_callback(self, field):
        def callback(attr, old, new):
            if self._loading or new == self.widgets[field].value:
                return
            self._pending_inputs.setdefault(self.selected_line, {})[field] = new
            self._store_pending_drafts()
            self._clear_approval()
        return callback

    def _continuum_changed(self, attr, old, new) -> None:
        if self._loading:
            return
        self._continuum_drafts[self.selected_line] = self._continuum_values()
        self._store_pending_drafts()
        if attr == "active" and not new:
            pending = self._pending_inputs.get(self.selected_line, {})
            had_draft = any(name in pending for name in _CONTINUUM_FIELDS)
            for name in _CONTINUUM_FIELDS:
                pending.pop(name, None)
            try:
                correction = self._continuum_arguments()
            except (TypeError, ValueError):
                # Disabling is always possible, even with incomplete anchor drafts.
                correction = replace(self.session.controls[self.selected_line].continuum,
                                     enabled=False)
                had_draft = True
            if had_draft:
                # Disabling explicitly discards invalid continuum drafts only.
                self._populate_continuum(correction)
            self.apply_controls(continuum=correction)
        else:
            self.apply_controls()

    def _populate_continuum(self, correction, *, include_draft=False) -> None:
        anchors = correction.anchors
        fit = self.session.fits[self.selected_line]
        values = {
            "enabled": correction.enabled,
            "mode": correction.mode if correction.mode != "none" else "constant",
            "continuum_level": repr(correction.level),
            "continuum_anchor1_x": repr(anchors[0][0]) if anchors else repr(fit.window[0]),
            "continuum_anchor1_y": repr(anchors[0][1]) if anchors else "1.0",
            "continuum_anchor2_x": repr(anchors[1][0]) if len(anchors) > 1 else repr(fit.window[1]),
            "continuum_anchor2_y": repr(anchors[1][1]) if len(anchors) > 1 else "1.0",
        }
        if include_draft:
            values.update(self._continuum_drafts.get(self.selected_line, {}))
        was_loading = self._loading
        self._loading = True
        self.widgets["continuum_enabled"].active = [0] if values["enabled"] else []
        self.widgets["continuum_mode"].value = values["mode"]
        for name in _CONTINUUM_FIELDS:
            self.widgets[name].value = values[name]
        self._loading = was_loading

    def _continuum_mouse_changed(self, attr, old, new) -> None:
        self.detail_plot.profile.toolbar.active_drag = (
            self.detail_plot.mask_tool if new == "mask" else None
        )

    def _continuum_tap(self, event) -> None:
        action = self.widgets["continuum_mouse"].value
        if action == "mask":
            return
        if (not math.isfinite(event.x) or not math.isfinite(event.y)
                or event.x <= 0 or event.y <= 0):
            self._error("Continuum anchors need finite positive wavelengths and levels.")
            return
        # The clicked coordinates are user input, never inferred from flux pixels.
        self._loading = True
        self.widgets["continuum_enabled"].active = [0]
        if action == "constant":
            self.widgets["continuum_mode"].value = "constant"
            self.widgets["continuum_level"].value = repr(float(event.y))
        else:
            self.widgets["continuum_mode"].value = "linear"
            anchor = "1" if action == "anchor1" else "2"
            self.widgets[f"continuum_anchor{anchor}_x"].value = repr(float(event.x))
            self.widgets[f"continuum_anchor{anchor}_y"].value = repr(float(event.y))
        self._loading = False
        pending = self._pending_inputs.get(self.selected_line, {})
        touched = (["continuum_level"] if action == "constant" else
                   [f"continuum_anchor{anchor}_x", f"continuum_anchor{anchor}_y"])
        for name in touched:
            pending.pop(name, None)
        self._continuum_changed(None, None, None)

    def remove_continuum_anchor(self, index: int) -> None:
        self._loading = True
        self.widgets["continuum_mode"].value = "linear"
        self.widgets[f"continuum_anchor{index}_x"].value = ""
        self.widgets[f"continuum_anchor{index}_y"].value = ""
        self._loading = False
        self._continuum_changed(None, None, None)

    def reset_continuum_selected(self) -> None:
        # Apply unity through the same adapter, preserving ordinary fitting controls.
        self._loading = True
        self.widgets["continuum_enabled"].active = []
        self.widgets["continuum_mode"].value = "constant"
        self.widgets["continuum_level"].value = "1.0"
        lower, upper = self.session.fits[self.selected_line].window
        for index, wavelength in ((1, lower), (2, upper)):
            self.widgets[f"continuum_anchor{index}_x"].value = repr(wavelength)
            self.widgets[f"continuum_anchor{index}_y"].value = "1.0"
        self._loading = False
        self._continuum_drafts.pop(self.selected_line, None)
        for name in _CONTINUUM_FIELDS:
            self._pending_inputs.get(self.selected_line, {}).pop(name, None)
        self.apply_controls()
        if self.selected_line not in self._invalid_lines:
            self._success("Local continuum reset to unity; scientific review is pending.")

    def _mask_selected(self, attr, old, new) -> None:
        if self._loading:
            return
        if self.selected_line in self._mask_drafts:
            self._loading = True
            self.widgets["mask_selector"].value = old
            self._loading = False
            self._error("Apply or reset the current mask draft before selecting another interval.")
            return
        self._mask_selection[self.selected_line] = new
        regions = self._regions()
        self._loading = True
        if new and int(new) < len(regions):
            self.widgets["mask_lower"].value = repr(regions[int(new)][0])
            self.widgets["mask_upper"].value = repr(regions[int(new)][1])
        else:
            self.widgets["mask_lower"].value = ""
            self.widgets["mask_upper"].value = ""
        self._loading = False

    def _mask_field_changed(self, attr, old, new) -> None:
        if self._loading:
            return
        selected = self.widgets["mask_selector"].value
        if selected or self.selected_line in self._mask_drafts:
            self._mask_drafts[self.selected_line] = {
                "selected": selected,
                "lower": self.widgets["mask_lower"].value,
                "upper": self.widgets["mask_upper"].value,
            }
            if selected:
                self.update_mask()
            else:
                self.add_mask()

    def _mask_draft_callback(self, boundary):
        def callback(attr, old, new):
            if self._loading:
                return
            selected = self.widgets["mask_selector"].value
            if not selected and self.selected_line not in self._mask_drafts:
                return  # A not-yet-added interval is only a prospective mask.
            draft = dict(self._mask_drafts.get(self.selected_line, {
                "selected": selected, "lower": self.widgets["mask_lower"].value,
                "upper": self.widgets["mask_upper"].value,
            }))
            if new == self.widgets[f"mask_{boundary}"].value:
                return
            draft[boundary] = new
            self._mask_drafts[self.selected_line] = draft
            self._pending_inputs.setdefault(self.selected_line, {})[f"mask_{boundary}"] = new
            self._clear_approval()
        return callback

    def _mask_numbers(self):
        return (float(self.widgets["mask_lower"].value), float(self.widgets["mask_upper"].value))

    def add_mask(self) -> None:
        draft = self._mask_drafts.get(self.selected_line)
        if self.selected_line in self._invalid_lines and (draft is None or draft["selected"]):
            self._error("Apply or reset the current fitting draft before adding another mask.")
            return
        self._clear_approval()
        self._mask_drafts[self.selected_line] = {
            "selected": "", "lower": self.widgets["mask_lower"].value,
            "upper": self.widgets["mask_upper"].value,
        }
        try:
            interval = self._mask_numbers()
        except (ValueError, TypeError) as exc:
            self._error(exc)
            return
        regions = self._regions()
        self._mask_selection[self.selected_line] = str(len(regions))
        self.apply_controls(regions=[*regions, interval])

    def update_mask(self) -> None:
        selected = self.widgets["mask_selector"].value
        if not selected:
            self._error("Select an interval to update, or use Add interval.")
            return
        self._clear_approval()
        self._mask_drafts[self.selected_line] = {
            "selected": selected, "lower": self.widgets["mask_lower"].value,
            "upper": self.widgets["mask_upper"].value,
        }
        try:
            regions = self._regions()
            regions[int(selected)] = self._mask_numbers()
        except (ValueError, TypeError, IndexError) as exc:
            self._error(exc)
            return
        self.apply_controls(regions=regions)

    def remove_mask(self) -> None:
        selected = self.widgets["mask_selector"].value
        if not selected:
            self._error("Select an interval to remove.")
            return
        regions = self._regions()
        del regions[int(selected)]
        self._mask_drafts.pop(self.selected_line, None)
        self._mask_selection[self.selected_line] = str(min(int(selected), len(regions)-1))
        for name in ("mask_lower", "mask_upper"):
            self._pending_inputs.get(self.selected_line, {}).pop(name, None)
        if self.apply_controls(regions=regions):
            self._sync_mask_selector(populate_bounds=True)

    def _mask_geometry(self, event) -> None:
        if (self.widgets["continuum_mouse"].value != "mask"
                or not event.final or event.geometry.get("type") != "rect"):
            return
        lower, upper = sorted((float(event.geometry["x0"]), float(event.geometry["x1"])))
        if not math.isfinite(lower) or not math.isfinite(upper) or lower == upper:
            return
        if self.selected_line in self._invalid_lines:
            self._error("Apply or reset the current fitting draft before dragging another mask.")
            return
        regions = self._regions()
        self._mask_selection[self.selected_line] = str(len(regions))
        if self.apply_controls(regions=[*regions, (lower, upper)]):
            self._sync_mask_selector(populate_bounds=True)

    def _note_changed(self, attr, old, new) -> None:
        if self._loading:
            return
        try:
            self.session.decide(
                self.selected_line, self.session.decisions[self.selected_line].state, note=new,
            )
        except (TypeError, ValueError) as exc:
            self._note_errors[self.selected_line] = new
            self._error(exc)
        else:
            self._note_errors.pop(self.selected_line, None)
        self.refresh_results()

    def _session_notes_changed(self, attr, old, new) -> None:
        if self._loading:
            return
        try:
            self.session.notes = new
        except (TypeError, ValueError) as exc:
            self._note_errors["session"] = new
            self._error(exc)
        else:
            self._note_errors.pop("session", None)

    def decide_selected(self, state: str) -> None:
        if state == "accepted" and self.selected_line in self._invalid_lines:
            self._error("Apply or reset the draft fitting changes before accepting this line.")
            return
        try:
            self.session.decide(self.selected_line, state, note=self.widgets["review_note"].value)
        except (ValueError, TypeError) as exc:
            self._error(exc)
            return
        self._success(f"{self.selected_line}: {state}. Numerical quality flags are retained.")
        self.refresh_results()

    def refresh_results(self) -> None:
        estimate = self.session.reviewed
        velocity = self.session.spectrum.rv_kms
        rv = "not applied" if velocity is None else f"{velocity:+.3f} km/s"
        self.title.text = (
            "<h1 style='margin:0;font-size:24px'>Sc I line-fitting review</h1>"
            f"<p style='margin:5px 0;color:#526171'>{_escape(self.spectrum_label)} · "
            f"{_escape(self.session.input_identity['filename'])} · "
            f"{_escape(self.session.configuration)} · [Fe/H] {_number(self.session.feh)} · "
            f"rest air Å coordinates · RV correction: {_escape(rv)}</p>"
        )
        temperature, performance = estimate.temperature, estimate.performance
        if temperature is None:
            headline = "Missing — accept at least one valid fit"
            scatter = rmse = bias = n = "—"
        else:
            headline = f"{temperature.temperature_k:.2f} K"
            scatter = _number(temperature.line_scatter_k, 5)
            rmse, bias, n = (f"{performance.rmse_k:.2f}", f"{performance.bias_k:.2f}",
                             str(performance.n_stars))
        included = ", ".join(estimate.included_line_ids) or "none"
        self.summary.text = (
            f"<span style='font-size:21px'><b>Reviewed Teff:</b> {_escape(headline)}</span>"
            f"<br/><b>Accepted lines ({len(estimate.included_line_ids)}):</b> {_escape(included)}"
            f"<br/>Line-to-line sample scatter: {scatter} K · "
            f"Empirical exact-subset RMSE: {rmse} K · N={n} · bias={bias} K. "
            "<b>These are different diagnostics, not a total temperature error bar.</b>"
        )
        if temperature is not None and temperature.warnings:
            self.summary.text += "<br/>" + "<br/>".join(
                _escape(w.message) for w in temperature.warnings
            )
        for line, (button, facts, flags, plots, card) in self.overview.items():
            fit, decision = self.session.fits[line], self.session.decisions[line]
            state = decision.state.capitalize()
            if line in self._invalid_lines:
                state += " · unapplied edit"
            button.label = f"{line} · {state}"
            button.button_type = {"pending": "default", "accepted": "success",
                                  "rejected": "danger"}[decision.state]
            facts.text = (
                f"<b>{_escape(fit.status)}</b> · area {_number(fit.area_angstrom)} Å · "
                f"Teff {_number(estimate.line_temperatures_k.get(line), 6)} K"
            )
            flags.text = ("<span style='color:#a63635'>Flags: " +
                          _escape(", ".join(fit.flags)) + "</span>" if fit.flags else
                          "<span style='color:#526171'>No numerical quality flags</span>")
            if fit.status in ("failed", "unavailable", "excluded"):
                flags.text += "<br/>" + _escape(fit.message)
            card.styles = {"border-bottom": "1px solid #d4dbe1", "padding-bottom": "5px",
                           "border-left": f"3px solid {_COLORS[decision.state]}"}
            if self._plot_fits.get(line) is not fit:
                plots.update(self.session.spectrum, fit, self.session.corrected_spectrum(line),
                             self.session.controls[line].continuum)
                self._plot_fits[line] = fit
        line = self.selected_line
        fit, decision = self.session.fits[line], self.session.decisions[line]
        valid = (fit.status == "converged" and fit.area_angstrom is not None
                 and math.isfinite(fit.area_angstrom) and fit.area_angstrom > 0)
        self.widgets["accept_line"].disabled = not valid or line in self._invalid_lines
        self.widgets["accept_line"].label = "Accept flagged fit" if fit.flags else "Accept fit"
        if (self._detail_fit is None or self._detail_fit[0] != line
                or self._detail_fit[1] is not fit):
            self.detail_plot.update(self.session.spectrum, fit,
                                    self.session.corrected_spectrum(line),
                                    self.session.controls[line].continuum)
            self._detail_fit = (line, fit)
        parameters = fit.parameters
        status = f"<b>{line}: {_escape(decision.state.upper())}</b> · {_escape(fit.status)}"
        if line in self._invalid_lines:
            status += (
                "<br/><b style='color:#a63635'>Unapplied draft; previous acceptance cleared.</b>"
            )
        status += (
            f"<br/>Area {_number(fit.area_angstrom, 9)} Å; formal error "
            f"{_number(fit.area_error_angstrom, 6)} Å. Native pixels {fit.n_pixels}; "
            f"excluded {fit.n_excluded}; residual RMS {_number(fit.residual_rms, 4)}."
        )
        if parameters is not None:
            status += (
                f"<br/>Depth {_number(parameters.amplitude, 7)}; centre "
                f"{parameters.centre_air_angstrom:.7f} Å; sigma "
                f"{_number(parameters.sigma_angstrom, 7)} Å."
            )
            errors = fit.parameter_errors
            if errors is not None:
                status += (
                    f"<br/>Formal parameter errors: depth {_number(errors.amplitude, 4)}, "
                    f"centre {_number(errors.centre_air_angstrom, 4)} Å, "
                    f"sigma {_number(errors.sigma_angstrom, 4)} Å."
                )
            else:
                status += "<br/>Formal parameter errors are unavailable."
        correction = self.session.controls[line].continuum
        before = self.session.uncorrected_fits[line]
        before_temperature = None
        if before.area_angstrom is not None and before.area_angstrom > 0:
            before_temperature = estimate_temperature(
                {line: before.area_angstrom}, configuration=self.session.configuration,
                feh=self.session.feh,
                reference_temperature_k=self.session.reference_temperature_k,
            ).temperature_k
        status += (
            f"<br/><b>Before correction:</b> W {_number(before.area_angstrom, 8)} Å; "
            f"line Teff {_number(before_temperature, 7)} K. "
            f"<b>Current:</b> W {_number(fit.area_angstrom, 8)} Å; "
            f"line Teff {_number(estimate.line_temperatures_k.get(line), 7)} K."
            "<br/>User continuum: "
            f"{_escape(correction.mode) if correction.enabled else 'disabled / unity'}. "
            "Formal covariance excludes continuum-placement uncertainty."
        )
        status += (
            f"<br/>Weighting: {_escape(fit.weighting)}; "
            f"window: {'provisional default' if fit.window_provisional else 'user override'}."
        )
        status += "<br/>" + _escape("Flags: " + ", ".join(fit.flags) if fit.flags else
                                     "No numerical quality flags; inspection is still required.")
        selected_warnings = [warning.message for warning in estimate.calibration_warnings
                             if warning.line_id == line]
        if selected_warnings:
            status += "<br/>" + "<br/>".join(_escape(text) for text in selected_warnings)
        self.details.text = status
        self.scientific_warnings.text = (
            f"<details><summary>Scientific warnings ({len(estimate.warnings)})</summary><ul>" +
            "".join(f"<li>{_escape(message)}</li>" for message in estimate.warnings) +
            "</ul></details>"
        )

    def refresh(self) -> None:
        self._loading = True
        line = self.selected_line
        fit, controls = self.session.fits[line], self.session.controls[line]
        initial = controls.initial or fit.parameters or GaussianParameters(
            0.3, fit.rest_wavelength_air_angstrom, 0.35,
        )
        values = {
            "window_lower": repr(fit.window[0]), "window_upper": repr(fit.window[1]),
            "initial_amplitude": repr(initial.amplitude),
            "initial_centre": repr(initial.centre_air_angstrom),
            "initial_sigma": repr(initial.sigma_angstrom),
            "automatic_initial": controls.initial is None,
        }
        values.update(self._drafts.get(line, {}))
        for name in _FIELD_NAMES:
            self.widgets[name].value = values[name]
        self.widgets["automatic_initial"].active = [0] if values["automatic_initial"] else []
        for name in ("initial_amplitude", "initial_centre", "initial_sigma"):
            self.widgets[name].disabled = values["automatic_initial"]
        regions = self._regions()
        self.widgets["mask_selector"].options = [("", "New interval / drag to add")] + [
            (str(i), f"{i+1}: {lo:.7f} – {hi:.7f} Å") for i, (lo, hi) in enumerate(regions)
        ]
        selected = self._mask_selection.get(line, "0" if regions else "")
        if selected and not 0 <= int(selected) < len(regions):
            selected = "0" if regions else ""
        mask_values = {
            "selected": selected,
            "lower": repr(regions[int(selected)][0]) if selected else "",
            "upper": repr(regions[int(selected)][1]) if selected else "",
        }
        mask_values.update(self._mask_drafts.get(line, {}))
        self.widgets["mask_selector"].value = mask_values["selected"]
        self.widgets["mask_lower"].value = mask_values["lower"]
        self.widgets["mask_upper"].value = mask_values["upper"]
        self._populate_continuum(controls.continuum, include_draft=True)
        self._control_availability()
        self.widgets["review_note"].value = self._note_errors.get(
            line, self.session.decisions[line].note,
        )
        self.widgets["review_notes"].value = self._note_errors.get("session", self.session.notes)
        self._loading = False
        self.refresh_results()

    def save_review(self) -> None:
        try:
            if self._invalid_lines:
                raise ValueError("Apply or reset all draft fitting changes before saving.")
            if self._note_errors:
                raise ValueError("Correct the invalid review notes before saving.")
            self.session.save(self.widgets["review_path"].value,
                              overwrite=bool(self.widgets["overwrite"].active),
                              protected_paths=self.protected_paths)
        except (ValueError, TypeError, OSError) as exc:
            self._error(exc)
            return
        self._success("Review JSON saved. Reopen using the same unchanged spectrum.")
        if self._persisted is not None:
            self._persisted(self)

    def load_review(self) -> None:
        try:
            self.session = ReviewSession.load(
                self.session.spectrum, self.widgets["review_path"].value,
                input_identity=self.session.input_identity,
            )
        except (ValueError, TypeError, OSError) as exc:
            self._error(exc)
            return
        self._drafts.clear()
        self._mask_drafts.clear()
        self._continuum_drafts.clear()
        self._pending_inputs.clear()
        self._mask_selection.clear()
        self._note_errors.clear()
        self._invalid_lines.clear()
        self.export_protected_paths.add(
            Path(self.widgets["review_path"].value).expanduser().resolve()
        )
        self._success("Review restored after identity verification; fits recomputed from memory.")
        self.refresh()
        if self._persisted is not None:
            self._persisted(self)

    def export_results(self) -> None:
        from .review_exports import write_review_outputs

        try:
            if self._invalid_lines:
                raise ValueError("Apply or reset all draft fitting changes before exporting.")
            if self._note_errors:
                raise ValueError("Correct the invalid review notes before exporting.")
            paths = set(self.export_protected_paths)
            current_review = Path(self.widgets["review_path"].value).expanduser()
            if current_review.exists():
                paths.add(current_review.resolve())
            write_review_outputs(
                self.session, self.widgets["output_dir"].value,
                overwrite=bool(self.widgets["overwrite"].active),
                plot_format=self.widgets["plot_format"].value,
                spectrum_label=self.spectrum_label,
                protected_paths=(*self.protected_paths, *paths),
            )
        except (ValueError, TypeError, OSError) as exc:
            self._error(exc)
            return
        self._success(
            "Reviewed outputs exported: review.json, summary.json, lines.csv and diagnostics."
        )


def build_review_document(
    doc,
    session: ReviewSession,
    *,
    review_path: str | Path = "review.json",
    output_dir: str | Path = "review-results",
    protected_paths: Iterable[str | Path] = (),
    export_protected_paths: Iterable[str | Path] = (),
    spectrum_label: str = "Input spectrum",
) -> ReviewDocument:
    """Build one local review document; return its controller for integration tests."""
    return ReviewDocument(doc, session, review_path=review_path, output_dir=output_dir,
                          protected_paths=protected_paths,
                          export_protected_paths=export_protected_paths,
                          spectrum_label=spectrum_label)


def create_inspection_server(
    session: ReviewSession,
    *,
    port: int = 0,
    review_path: str | Path = "review.json",
    output_dir: str | Path = "review-results",
    protected_paths: Iterable[str | Path] = (),
    export_protected_paths: Iterable[str | Path] = (),
    spectrum_label: str = "Input spectrum",
) -> BaseServer:
    """Create a loopback-only server with local assets and independent tab state."""
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer from 0 through 65535")
    baseline = {
        "payload": session.to_dict(), "review_path": review_path,
        "export_paths": tuple(export_protected_paths),
    }

    def saved(controller):
        baseline["payload"] = controller.session.to_dict()
        baseline["review_path"] = controller.widgets["review_path"].value
        baseline["export_paths"] = tuple(controller.export_protected_paths)

    def make_document(doc):
        local_session = ReviewSession.from_dict(
            session.spectrum, baseline["payload"], input_identity=session.input_identity,
        )
        controller = build_review_document(
            doc, local_session, review_path=baseline["review_path"], output_dir=output_dir,
            protected_paths=protected_paths, export_protected_paths=baseline["export_paths"],
            spectrum_label=spectrum_label,
        )
        controller._persisted = saved

    class LocalResources(BokehTornado):
        def resources(self, absolute_url=None):
            # Explicit mode takes precedence over BOKEH_RESOURCES and CDN settings.
            return Resources(mode="server", root_url="/", minified=True)

    loop = IOLoop()
    sockets, actual_port = bind_sockets("127.0.0.1", port)
    try:
        application = LocalResources(
            {"/": Application(FunctionHandler(make_document))},
            extra_websocket_origins=[f"127.0.0.1:{actual_port}"],
            absolute_url=f"http://127.0.0.1:{actual_port}",
            unused_session_lifetime_milliseconds=15000, check_unused_sessions_milliseconds=1000,
        )
        http = HTTPServer(application, xheaders=False)
        http.add_sockets(sockets)
        return BaseServer(loop, application, http)
    except Exception:
        for sock in sockets:
            sock.close()
        loop.close(all_fds=True)
        raise


def serve_inspection(session: ReviewSession, *, open_browser: bool = True, **kwargs) -> None:
    """Run a local server until Ctrl-C; browser requests and JavaScript stay local."""
    server = create_inspection_server(session, **kwargs)
    server.start()
    url = f"http://127.0.0.1:{server.port}/"
    print(f"Sc I review: {url}", flush=True)
    print("Local only. Save review JSON before stopping; press Ctrl-C to close.", flush=True)
    if open_browser:
        server.io_loop.add_callback(server.show, "/", browser=None)
    try:
        server.io_loop.start()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        server.io_loop.close(all_fds=True)
