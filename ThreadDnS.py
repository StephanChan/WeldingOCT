
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 18:26:44 2023

@author: admin
"""

from PyQt5.QtCore import  QThread
from Generaic_functions import findchangept
import numpy as np
import traceback
global SCALE
SCALE =1000
import matplotlib.pyplot as plt
import datetime
import os
from pathlib import Path
# from libtiff import TIFF
import tifffile as TIFF
import time
from ActionTypes import AcqTypes, DnSActions, EXIT_ACTION
from HardwareSpecs import TIFF_APPEND_WRITES_DEFAULT
from DataShape import data_shape

ALINE_MODES = (
    AcqTypes.FINITE_ALINE,
    AcqTypes.CONTINUOUS_ALINE,
)

BLINE_MODES = (
    AcqTypes.FINITE_BLINE,
    AcqTypes.CONTINUOUS_BLINE,
    AcqTypes.TRIGGERED_ACQUIRE,
)

CSCAN_MODES = (
    AcqTypes.FINITE_CSCAN,
    AcqTypes.CONTINUOUS_CSCAN,
)

TRIGGERED_PREALLOC_ALINES = 50000
TRIGGERED_PREVIEW_DOWNSAMPLE = 20

class DnSThread(QThread):
    def __init__(self):
        super().__init__()
        self.AIP = []
        self.Dyn = []
        # self.totalTiles = 0
        self.display_actions = 0
        self.active_tasks = 0
        self.tiff_append_writes = TIFF_APPEND_WRITES_DEFAULT
        self._tiff_initialized_files = set()
        self.MeanVolume = []
        self.DynamicVolume = []
        self.TriggeredBlineBuffer = []
        self.TriggeredFilledAlines = 0
        self.TriggeredBlinePreview = []
        self.TriggeredPreviewCanvas = []
        self.triggered_preview_interval_s = 0.25
        self.triggered_preview_downsample = TRIGGERED_PREVIEW_DOWNSAMPLE
        self._last_triggered_preview_t = 0.0
        self._triggered_first_bline_reported = False

    def reset_dynamic_accumulators(self):
        self.AIP = []
        self.Dyn = []
        self.DynBline = []
        self.MeanVolume = []
        self.DynamicVolume = []
        self.TriggeredBlineBuffer = []
        self.TriggeredFilledAlines = 0
        self.TriggeredBlinePreview = []
        self.TriggeredPreviewCanvas = []
        self._last_triggered_preview_t = 0.0
        self._triggered_first_bline_reported = False
        
    def run(self):
        # self.Dynmax = self.ui.Dynmax.value()
        # self.Dynmin = self.ui.Dynmin.value()
        
        self.QueueOut()
        
    def QueueOut(self):
        self.item = self.queue.get()
        while self.item.action != EXIT_ACTION:
            self.active_tasks += 1
            start=time.time()
            self.current_acq_mode = self.item.acq_mode
            try:
                if self.item.action in (
                    AcqTypes.FINITE_ALINE,
                    AcqTypes.CONTINUOUS_ALINE,
                ):
                    self.display_actions += 1
                    self.Process_aline(self.item.data, self.item.raw, self.current_acq_mode, self.item.gpu_avg_count)
                    self._emit_display(kind="aline")
                elif self.item.action in BLINE_MODES:
                    self.Process_bline(self.item.data, self.item.raw, self.item.dynamic, self.current_acq_mode, self.item.gpu_avg_count)
                    if self.item.action != AcqTypes.TRIGGERED_ACQUIRE:
                        self.display_actions += 1
                        self._emit_display(kind="bline")
                    elif self.should_emit_triggered_preview():
                        self.update_triggered_preview()
                        self.display_actions += 1
                        self._emit_display(kind="triggered_bline")
                elif self.item.action in (
                    AcqTypes.FINITE_CSCAN,
                    AcqTypes.CONTINUOUS_CSCAN,
                ):
                    self.display_actions += 1
                    if self.realtime_cscan_dynamic_enabled(self.current_acq_mode, self.item.dynamic):
                        self.Process_Cscan_RealtimeDynamic(
                            self.item.data,
                            self.item.dynamic,
                            self.current_acq_mode,
                            self.item.gpu_avg_count,
                        )
                    elif self.current_dynamic_enabled():
                        self.Process_Cscan_Dynamic(self.item.data, self.item.dynamic, self.current_acq_mode, self.item.gpu_avg_count)
                    else:
                        self.Process_Cscan(self.item.data, self.item.raw, self.current_acq_mode, self.item.gpu_avg_count)
                    self._emit_display(kind="cscan")
                    
                elif self.item.action == DnSActions.CLEAR:
                    self.reset_dynamic_accumulators()
                elif self.item.action == DnSActions.DISPLAY_COUNTS:
                    self.print_display_counts(self.item.context)
                elif self.item.action == DnSActions.FINALIZE_TRIGGERED_ACQUIRE:
                    self.FinalizeTriggeredAcquire()
                    self.display_actions += 1
                    self._emit_display(kind="triggered_bline")
                else:
                    message = f"Unknown display/save command: {self.item.action}"
                    print(message)
                    self.emit_status(message)
                    # self.ui.PrintOut.append(message)
                if time.time()-start>1:
                    print('time for DnS:',round(time.time()-start,3))
            except Exception as error:
                message = "Display/save processing failed. This item was skipped."
                print(message)
                self.emit_status(message)
                # self.ui.PrintOut.append(message)
                print(traceback.format_exc())
            finally:
                self.active_tasks = max(0, self.active_tasks - 1)
            self.item = self.queue.get()
            
        self.emit_status("Display/save thread exited.")

    def is_idle(self):
        return self.queue.qsize() == 0 and self.active_tasks == 0

    def emit_status(self, message):
        if message is None:
            return
        self.ui_bridge.status_message.emit(str(message))

    def current_dynamic_enabled(self):
        return self.ui.DynCheckBox.isChecked()

    def current_save_enabled(self):
        return self.ui.Save.isChecked()

    def current_aline_avg(self):
        return max(1, int(self.ui.AlineAVG.value()))

    def current_y_pixels(self):
        return max(1, int(self.ui.Ypixels.value()))

    def current_bline_avg(self):
        return max(1, int(self.ui.BlineAVG.value()))

    def realtime_cscan_dynamic_enabled(self, acq_mode, dynamic):
        return (
            self.current_dynamic_enabled()
            and acq_mode in CSCAN_MODES
            and self.ui.RealtimeDynCheckBox.isChecked()
            and np.size(dynamic) > 0
        )

    def write_stack_tiff(self, filename, stack, count=None):
        frame_count = int(stack.shape[0]) if count is None else int(count)
        for ii in range(frame_count):
            self.write_tiff_frame(filename, stack[ii], append_if_exists=True)

    @staticmethod
    def display_data(data):
        if isinstance(data, np.ndarray) and data.dtype.kind == 'c':
            return np.abs(data)
        return data

    @staticmethod
    def save_data(data):
        if not (isinstance(data, np.ndarray) and data.dtype.kind == 'c'):
            return data
        z_pixels = data.shape[-1]
        interleaved = np.empty(data.shape[:-1] + (z_pixels * 2,), dtype=np.float32)
        interleaved[..., :z_pixels] = np.abs(data).astype(np.float32, copy=False)
        interleaved[..., z_pixels:] = np.angle(data).astype(np.float32, copy=False)
        return interleaved

    def reset_tiff_output(self, filename):
        path = Path(filename)
        try:
            if path.exists():
                path.unlink()
        except OSError as error:
            message = f"Failed to reset TIFF output {filename}: {error}"
            print(message)
            self.emit_status(message)
        self._tiff_initialized_files.discard(str(path.resolve()))

    def write_tiff_frame(self, filename, image, append_if_exists=True):
        path = Path(filename)
        resolved = str(path.resolve())
        if self.tiff_append_writes:
            TIFF.imwrite(filename, image, append=append_if_exists)
            if append_if_exists:
                self._tiff_initialized_files.add(resolved)
            return

        first_write = resolved not in self._tiff_initialized_files
        if first_write:
            self.reset_tiff_output(filename)
        TIFF.imwrite(filename, image, append=(append_if_exists and not first_write))
        if append_if_exists:
            self._tiff_initialized_files.add(resolved)

    def _emit_display(self, kind: str):
        """
        Emit display payloads to GUI thread via ui_bridge.
        This thread must NOT create QPixmap or touch widgets for rendering.
        """
        bridge = getattr(self, "ui_bridge", None)
        if bridge is None:
            return

        acq_mode = self.current_acq_mode
        use_dynamic = self.current_dynamic_enabled()

        if kind == "aline" and hasattr(self, "Aline") and np.size(self.Aline) > 0:
            bridge.aline_ready.emit({"mode": acq_mode, "aline": np.array(self.Aline, copy=True)})

        if kind == "bline" and hasattr(self, "Bline") and np.size(self.Bline) > 0:
            dyn = None
            if use_dynamic and hasattr(self, "DynBline") and np.size(self.DynBline) > 0:
                dyn = np.array(self.DynBline, copy=True)
            bridge.bline_ready.emit({"mode": acq_mode, "bline": np.array(self.Bline, copy=True), "dyn": dyn})

        if kind == "triggered_bline" and hasattr(self, "Bline") and np.size(self.Bline) > 0:
            dyn = None
            if use_dynamic and hasattr(self, "DynBline") and np.size(self.DynBline) > 0:
                dyn = np.array(self.DynBline, copy=True)
            appended = None
            if hasattr(self, "TriggeredBlinePreview") and np.size(self.TriggeredBlinePreview) > 0:
                appended = np.array(self.TriggeredBlinePreview, copy=True)
            bridge.bline_ready.emit(
                {
                    "mode": acq_mode,
                    "bline": np.array(self.Bline, copy=True),
                    "dyn": dyn,
                    "appended": appended,
                }
            )

        if (
            kind == "cscan"
            and hasattr(self, "Bline")
            and hasattr(self, "AIP")
            and np.size(self.Bline) > 0
            and np.size(self.AIP) > 0
        ):
            dynb = None
            dyn = None
            if use_dynamic and hasattr(self, "DynBline") and np.size(self.DynBline) > 0:
                dynb = np.array(self.DynBline, copy=True)
            if use_dynamic and hasattr(self, "Dyn") and np.size(self.Dyn) > 0:
                dyn = np.array(self.Dyn, copy=True)
            bridge.cscan_ready.emit(
                {
                    "mode": acq_mode,
                    "bline": np.array(self.Bline, copy=True),
                    "dynb": dynb,
                    "aip": np.array(self.AIP, copy=True),
                    "dyn": dyn,
                }
            )

    def print_display_counts(self, display_name = ''):
        message = f"{self.display_actions} {display_name} display update(s) completed."
        print(message)
        # self.ui.PrintOut.append(message)
        self.display_actions = 0
        
    def Process_aline(self, data, raw = False, acq_mode=None, gpu_avg_count=1):
        display_data = self.display_data(data)
        shape = data_shape(self.ui, display_data, raw, acq_mode, gpu_avg_count)
        Zpixels = shape.z_pixels
        Xpixels = shape.x_pixels
        # Bline averaging
        if display_data.shape[0] > 1:
            Ascan = np.mean(display_data,0)
        else:
            Ascan = display_data[0]
        # Aline averaging if needed
        aline_avg = self.current_aline_avg()
        if aline_avg > 1:
            Ascan = Ascan.reshape([Xpixels//aline_avg, aline_avg, Zpixels])
            Ascan = np.mean(Ascan,1)
            Xpixels = Xpixels//aline_avg
            
        self.Aline = Ascan[Xpixels//2]
        if self.current_save_enabled():
            self.Save(data=data, raw=raw, acq_mode=acq_mode, gpu_avg_count=gpu_avg_count)
            
    
    def Process_bline(self, data, raw = False, dynamic = [], acq_mode=None, gpu_avg_count=1):
        display_data = self.display_data(data)
        shape = data_shape(self.ui, display_data, raw, acq_mode, gpu_avg_count)
        Zpixels = shape.z_pixels
        Xpixels = shape.x_pixels
        if self.current_dynamic_enabled() or raw:
            if display_data.shape[0] > 1:
                Bline=np.mean(display_data,0)
            else:
                Bline = display_data[0]
        else:
            Bline = display_data[0]
        # Aline averaging if needed
        aline_avg = self.current_aline_avg()
        if aline_avg > 1:
            Bline = Bline.reshape([Xpixels//aline_avg, aline_avg, Zpixels])
            Bline = np.mean(Bline,1)
            Xpixels = Xpixels//aline_avg
        self.Bline = np.transpose(Bline)
        if self.current_dynamic_enabled() and len(dynamic)>0:
            self.DynBline = np.transpose(dynamic)
        else:
            self.DynBline = []
            self.Dyn = []

        if acq_mode == AcqTypes.TRIGGERED_ACQUIRE:
            self.append_triggered_bline(Bline)
            return
        
        if self.current_save_enabled():
            self.Save(data=data, dynamic=dynamic, raw=raw, acq_mode=acq_mode, gpu_avg_count=gpu_avg_count)

    def ensure_triggered_buffer_capacity(self, bline):
        bline = np.asarray(bline)
        x_new, z_pixels = bline.shape
        required = int(self.TriggeredFilledAlines) + int(x_new)

        needs_new = (
            not isinstance(self.TriggeredBlineBuffer, np.ndarray)
            or self.TriggeredBlineBuffer.ndim != 2
            or self.TriggeredBlineBuffer.shape[1] != z_pixels
        )
        if needs_new:
            capacity = max(TRIGGERED_PREALLOC_ALINES, required)
            self.TriggeredBlineBuffer = np.zeros((capacity, z_pixels), dtype=bline.dtype)
            self.TriggeredFilledAlines = 0
            self.TriggeredPreviewCanvas = []
        elif required > self.TriggeredBlineBuffer.shape[0]:
            capacity = max(required, self.TriggeredBlineBuffer.shape[0] * 2)
            grown = np.zeros((capacity, z_pixels), dtype=self.TriggeredBlineBuffer.dtype)
            grown[: self.TriggeredFilledAlines, :] = self.TriggeredBlineBuffer[: self.TriggeredFilledAlines, :]
            self.TriggeredBlineBuffer = grown
            self.TriggeredPreviewCanvas = []

        return required

    def append_triggered_bline(self, bline):
        bline = np.asarray(bline)
        self.ensure_triggered_buffer_capacity(bline)
        x_new = int(bline.shape[0])
        x0 = int(self.TriggeredFilledAlines)
        x1 = x0 + x_new
        if not self._triggered_first_bline_reported:
            edge_width = min(20, x_new)
            edge_mean = float(np.mean(bline[:edge_width, :])) if edge_width > 0 else 0.0
            full_mean = float(np.mean(bline)) if bline.size > 0 else 0.0
            print(
                "triggeredAcquire first B-line check: "
                f"first {edge_width} A-line mean={edge_mean:.3f}, "
                f"full B-line mean={full_mean:.3f}"
            )
            self._triggered_first_bline_reported = True
        self.TriggeredBlineBuffer[x0:x1, :] = bline
        self.TriggeredFilledAlines = x1

    def should_emit_triggered_preview(self):
        now = time.monotonic()
        if now - self._last_triggered_preview_t < self.triggered_preview_interval_s:
            return False
        self._last_triggered_preview_t = now
        return True

    def ensure_triggered_preview_canvas(self):
        if self.TriggeredFilledAlines <= 0 or not isinstance(self.TriggeredBlineBuffer, np.ndarray):
            self.TriggeredBlinePreview = []
            return None, 1
        downsample = max(1, int(self.triggered_preview_downsample))
        preview_width = int(np.ceil(self.TriggeredBlineBuffer.shape[0] / float(downsample)))
        z_pixels = int(self.TriggeredBlineBuffer.shape[1])
        if (
            not isinstance(self.TriggeredPreviewCanvas, np.ndarray)
            or self.TriggeredPreviewCanvas.shape != (preview_width, z_pixels)
            or self.TriggeredPreviewCanvas.dtype != self.TriggeredBlineBuffer.dtype
        ):
            self.TriggeredPreviewCanvas = np.zeros(
                (preview_width, z_pixels),
                dtype=self.TriggeredBlineBuffer.dtype,
            )
        return self.TriggeredPreviewCanvas, downsample

    def downsample_triggered_preview(self, data, downsample):
        if downsample <= 1 or data.shape[0] <= 1:
            return data
        complete_rows = (data.shape[0] // downsample) * downsample
        blocks = []
        if complete_rows > 0:
            blocks.append(
                data[:complete_rows, :].reshape(-1, downsample, data.shape[1]).mean(axis=1)
            )
        if complete_rows < data.shape[0]:
            blocks.append(data[complete_rows:, :].mean(axis=0, keepdims=True))
        if len(blocks) == 1:
            return blocks[0]
        return np.vstack(blocks)

    def update_triggered_preview(self):
        canvas, downsample = self.ensure_triggered_preview_canvas()
        if canvas is None:
            return
        filled_preview_cols = int(np.ceil(self.TriggeredFilledAlines / float(downsample)))
        if filled_preview_cols > 0:
            valid = self.TriggeredBlineBuffer[: self.TriggeredFilledAlines, :]
            preview = self.downsample_triggered_preview(valid, downsample)
            canvas[: preview.shape[0], :] = preview
        self.TriggeredBlinePreview = np.transpose(canvas)
        print(
            "triggeredAcquire appended A-lines: "
            f"{self.TriggeredFilledAlines} "
            f"(preview downsample={downsample}, preview columns={filled_preview_cols}/{canvas.shape[0]})"
        )

    def FinalizeTriggeredAcquire(self):
        if self.TriggeredFilledAlines <= 0 or not isinstance(self.TriggeredBlineBuffer, np.ndarray):
            self.Bline = []
            self.DynBline = []
            self.TriggeredBlinePreview = []
            self.TriggeredPreviewCanvas = []
            return

        valid = self.TriggeredBlineBuffer[: self.TriggeredFilledAlines, :]
        downsample = max(1, int(self.triggered_preview_downsample))
        final_preview = self.downsample_triggered_preview(valid, downsample)
        self.TriggeredBlinePreview = np.transpose(final_preview)
        print(
            "triggeredAcquire final appended A-lines: "
            f"{self.TriggeredFilledAlines} (display downsample={downsample})"
        )

        if self.current_save_enabled():
            bundle = self.item.filename_bundle or {}
            filename = bundle.get("filename")
            if not filename:
                raise RuntimeError("Missing triggeredAcquire filename bundle.")
            self.write_tiff_frame(filename, np.transpose(valid), append_if_exists=False)

            
    def Process_Cscan_Dynamic(self, data, dynamic=[], acq_mode=None, gpu_avg_count=1):
        # print(dynamic.shape)
        display_data = self.display_data(data)
        shape = data_shape(self.ui, display_data, False, acq_mode, gpu_avg_count)
        Zpixels = shape.z_pixels
        Xpixels = shape.x_pixels
        Ypixels = self.current_y_pixels()
        dynamic_bline_idx = int(self.item.dynamic_bline_idx or 0)
        # Bline averaging
        if display_data.shape[0] > 1:
            Bline=np.mean(display_data,0)
        else:
            Bline = display_data[0]
        # Aline averaging if needed
        aline_avg = self.current_aline_avg()
        if aline_avg > 1:
            Bline = Bline.reshape([Xpixels//aline_avg, aline_avg, Zpixels])
            Bline = np.mean(Bline,1)
            Xpixels = Xpixels//aline_avg

        self.Bline = np.transpose(Bline)
        
        # print('Bline:', self.Bline[Zpixels//2:Zpixels//2+5, Xpixels//2])
        if len(dynamic)>0:
            self.DynBline = np.transpose(dynamic)
            # print('DynBline:', self.DynBline[Zpixels//2:Zpixels//2+5, Xpixels//2])
        else:
            self.DynBline = []
        
        if dynamic_bline_idx == 0:
            self.AIP = np.zeros([Ypixels, Xpixels])
        if len(dynamic)>0:
            if dynamic_bline_idx == 0:
                self.Dyn = np.zeros([Ypixels, Xpixels])
                
        # print(Bline.shape, self.AIP.shape)
        print('Ypixel: ', dynamic_bline_idx + 1, ' / ', Ypixels)
        self.AIP[dynamic_bline_idx, :] = np.mean(Bline,1)
        if len(dynamic)>0:
            self.Dyn[dynamic_bline_idx, :] = np.mean(dynamic,1)
        if self.current_save_enabled():
            self.Save(data=data, dynamic=dynamic, acq_mode=acq_mode, gpu_avg_count=gpu_avg_count)
        
        
    def Process_Cscan(self, data, raw = False, acq_mode=None, gpu_avg_count=1):
        display_data = self.display_data(data)
        shape = data_shape(self.ui, display_data, raw, acq_mode, gpu_avg_count)
        Zpixels = shape.z_pixels
        Xpixels = shape.x_pixels
        Ypixels = shape.y_pixels
        # Raw data still needs repeat-frame grouping. Processed data should already be averaged in GPU.
        bline_avg = self.current_bline_avg()
        if raw and bline_avg > 1:
            # reshape into Ypixels x Xpixels x Zpixels
            Cscan = display_data.reshape([Ypixels, bline_avg, Xpixels,Zpixels])
            Cscan=np.mean(Cscan,1)
        else:
            Cscan = display_data.copy()
        # Aline averaging if needed
        aline_avg = self.current_aline_avg()
        if aline_avg > 1:
            Cscan = Cscan.reshape([Ypixels, Xpixels//aline_avg, aline_avg, Zpixels])
            Cscan = np.mean(Cscan,2)
            Xpixels = Xpixels//aline_avg
        # print(data[10,100,50:60])
        self.Bline = np.transpose(Cscan[Ypixels//2,:,:]).copy()# has to be first index, otherwise the memory space is not continuous
        self.AIP = np.mean(Cscan,2)
        self.DynBline = []
        self.Dyn = []
        
        if self.current_save_enabled():
            self.Save(data=data, raw=raw, acq_mode=acq_mode, gpu_avg_count=gpu_avg_count)

    def Process_Cscan_RealtimeDynamic(self, data, dynamic=[], acq_mode=None, gpu_avg_count=1):
        display_data = self.display_data(data)
        shape = data_shape(self.ui, display_data, False, acq_mode, gpu_avg_count)
        zpixels = shape.z_pixels
        xpixels = shape.x_pixels
        ypixels = self.current_y_pixels()
        dynamic_bline_idx = int(self.item.dynamic_bline_idx or 0)

        if display_data.shape[0] > 1:
            bline = np.mean(display_data, 0)
        else:
            bline = display_data[0]

        dyn_slice = np.asarray(dynamic, dtype=np.float32)
        aline_avg = self.current_aline_avg()
        if aline_avg > 1:
            bline = bline.reshape([xpixels // aline_avg, aline_avg, zpixels]).mean(axis=1)
            dyn_slice = dyn_slice.reshape([xpixels // aline_avg, aline_avg, zpixels]).mean(axis=1)
            xpixels = xpixels // aline_avg

        if (
            not isinstance(self.MeanVolume, np.ndarray)
            or self.MeanVolume.shape != (ypixels, xpixels, zpixels)
            or dynamic_bline_idx == 0
        ):
            self.MeanVolume = np.zeros((ypixels, xpixels, zpixels), dtype=np.float32)
            self.DynamicVolume = np.zeros((ypixels, xpixels, zpixels), dtype=np.float32)
            self.AIP = np.zeros((ypixels, xpixels), dtype=np.float32)
            self.Dyn = np.zeros((ypixels, xpixels), dtype=np.float32)

        self.Bline = np.transpose(bline)
        self.DynBline = np.transpose(dyn_slice)
        self.MeanVolume[dynamic_bline_idx, :, :] = bline
        self.DynamicVolume[dynamic_bline_idx, :, :] = dyn_slice
        self.AIP[dynamic_bline_idx, :] = np.mean(bline, axis=1)
        self.Dyn[dynamic_bline_idx, :] = np.mean(dyn_slice, axis=1)
        print('Ypixel: ', dynamic_bline_idx + 1, ' / ', ypixels)
        if dynamic_bline_idx + 1 == ypixels:
            if self.current_save_enabled():
                self.SaveRealtimeCscanDynamicVolumes(acq_mode)
            
    def Focusing(self, cscan):
         print(cscan.shape)
         bscan = cscan.mean(0)
         ascan = bscan.mean(0)
         print(ascan.shape)
         surfHeight = findchangept(ascan,1)
         self.ui.SurfHeight.setValue(surfHeight)
         message = 'Detected surface height: '+str(surfHeight)
         print(message)

    def SaveRealtimeCscanDynamicVolumes(self, acq_mode):
        bundle = self.item.filename_bundle
        dynamic_filename = bundle.get("dynamic_filename")
        mean_filename = bundle.get("mean_filename")
        if not dynamic_filename or not mean_filename:
            raise RuntimeError("Missing realtime cscan dynamic filename bundle.")
        try:
            TIFF.imwrite(dynamic_filename, np.asarray(self.DynamicVolume, dtype=np.float32), append=False)
            TIFF.imwrite(mean_filename, np.asarray(self.MeanVolume, dtype=np.float32), append=False)
        finally:
            pass


    def writeTiff(self,filename, image, overlap):
        tif = TIFF.open(filename, mode=overlap)
        tif.write_image(image)
        tif.close()
    def Save(self, data=[], dynamic=[], raw=False, acq_mode=None, gpu_avg_count=1):
        if getattr(self.item, "skip_save", False):
            return
        shape = data_shape(self.ui, data, raw, acq_mode, gpu_avg_count)
        data_to_save = self.save_data(data)
        Zpixels = shape.z_pixels
        Xpixels = shape.x_pixels
        Yrpt = shape.repeat_count
        Ypixels = shape.y_pixels
        bundle = self.item.filename_bundle or {}
        if acq_mode in ALINE_MODES:
            filename = bundle.get("filename")
            if not filename:
                raise RuntimeError("Missing aline filename bundle.")
            self.write_stack_tiff(filename, data_to_save, Yrpt)
                
        elif acq_mode in BLINE_MODES:
            filename = bundle.get("filename")
            dyn_filename = bundle.get("dynamic_filename")
            if not filename:
                raise RuntimeError("Missing bline filename bundle.")
            if dyn_filename is not None and np.size(dynamic) > 0:
                self.write_tiff_frame(dyn_filename, dynamic, append_if_exists=False)
            self.write_stack_tiff(filename, data_to_save, Yrpt)
                
        elif acq_mode in CSCAN_MODES:
            if self.current_dynamic_enabled():
                if self.ui.RealtimeDynCheckBox.isChecked():
                    return
                bline_filename = bundle.get("filename")
                dyn_filename = bundle.get("dynamic_filename")
                if not bline_filename or not dyn_filename:
                    raise RuntimeError("Missing cscan dynamic filename bundle.")
                self.write_stack_tiff(bline_filename, data_to_save, Yrpt)
                if np.size(dynamic) > 0:
                    self.write_tiff_frame(dyn_filename, dynamic, append_if_exists=True)
            else:
                filename = bundle.get("filename")
                if not filename:
                    raise RuntimeError("Missing cscan filename bundle.")
                self.write_stack_tiff(filename, data_to_save, Ypixels)

    def WriteData(self, data, filename):
        filePath = self.ui.DIR.toPlainText()
        filePath = filePath + "/" + filename
        # print(filePath)
        import time
        start = time.time()
        fp = open(filePath, 'wb')
        data.tofile(fp)
        fp.close()
        if time.time()-start > 1:
            message = 'Saving took '+str(round(time.time()-start,3))+' s.'
            print(message)
            # self.ui.PrintOut.append(message)
        
    def WriteAgar(self, data, context):
        [Ystep, Xstep] = context
        slice_num = self.ui.SliceN.value()
        filename = 'slice-'+str(slice_num)+'-agarTiles X-'+str(Xstep)+'-by Y-'+str(Ystep)+'-.bin'
        filePath = self.ui.DIR.toPlainText()
        filePath = filePath + "/" + filename
        # print(filePath)
        fp = open(filePath, 'wb')
        data.tofile(fp)
        fp.close()
        
