# -*- coding: utf-8 -*-
"""Display rendering helpers for Aline, Bline, and Cscan workflows."""

import numpy as np

from Generaic_functions import RGBImagePlot, RGBOverlayPlot, fastLinePlot, LinePlot


def display_array(array):
    if isinstance(array, np.ndarray) and array.dtype.kind == "c":
        return np.abs(array)
    return array


def dynamic_alpha(ui):
    return ui.DynContrast.value() / 100.0 if hasattr(ui, "DynContrast") else 0.5


def render_xz_pixmap(ui, intensity, dynamic=None):
    intensity = display_array(intensity)
    dynamic = display_array(dynamic)
    use_dynamic = ui.DynCheckBox.isChecked()
    if use_dynamic and dynamic is not None and np.size(dynamic) > 0:
        return RGBOverlayPlot(
            intensity,
            dynamic,
            ui.XZmin.value(),
            ui.XZmax.value(),
            alpha=dynamic_alpha(ui),
        )
    return RGBImagePlot(matrix1=intensity, m=ui.XZmin.value(), M=ui.XZmax.value())


def set_xy_projection(ui, intensity, dynamic=None):
    if intensity is None:
        return
    intensity = display_array(intensity)
    dynamic = display_array(dynamic)
    use_dynamic = ui.DynCheckBox.isChecked()
    if use_dynamic and dynamic is not None and np.size(dynamic) > 0:
        pixmap = RGBOverlayPlot(
            intensity,
            dynamic,
            ui.Intmin.value(),
            ui.Intmax.value(),
            alpha=dynamic_alpha(ui),
        )
    else:
        pixmap = RGBImagePlot(matrix1=intensity, m=ui.Intmin.value(), M=ui.Intmax.value())
    ui.XYplane.setPixmap(pixmap)


def render_aodo_waveform_ready(ui, payload):
    ao_waveform = payload.get("ao_waveform", None)
    do_waveform = payload.get("do_waveform", None)
    if ao_waveform is None or do_waveform is None:
        return
    ao_waveform = np.asarray(ao_waveform, dtype=np.float32)
    do_waveform = np.asarray(do_waveform, dtype=np.float32)
    wave_min = float(min(np.min(ao_waveform), np.min(do_waveform)))
    wave_max = float(max(np.max(ao_waveform), np.max(do_waveform)))
    margin = 1.0 if wave_max <= wave_min else 0.05 * (wave_max - wave_min)
    pixmap = LinePlot(
        ao_waveform,
        do_waveform,
        wave_min - margin,
        wave_max + margin,
    )
    ui.XwaveformLabel.setPixmap(pixmap)


def render_aline_ready(ui, payload):
    aline = payload.get("aline", None)
    if aline is None:
        return
    aline = display_array(aline)
    pixmap = fastLinePlot(
        aline,
        width=ui.XZplane.width(),
        height=ui.XZplane.height(),
        m=ui.XZmin.value(),
        M=ui.XZmax.value(),
    )
    ui.XZplane.setPixmap(pixmap)


def render_bline_ready(ui, payload):
    bline = payload.get("bline", None)
    if bline is None:
        return
    dyn = payload.get("dyn", None)
    ui.XZplane.setPixmap(render_xz_pixmap(ui, bline, dyn))
    appended = payload.get("appended", None)
    if appended is not None and np.size(appended) > 0:
        set_xy_projection(ui, appended, payload.get("appended_dyn", None))


def render_cscan_ready(ui, payload):
    bline = payload.get("bline", None)
    dynb = payload.get("dynb", None)
    aip = payload.get("aip", None)
    dyn = payload.get("dyn", None)

    if bline is not None:
        ui.XZplane.setPixmap(render_xz_pixmap(ui, bline, dynb))
    set_xy_projection(ui, aip, dyn)
