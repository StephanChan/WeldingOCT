# -*- coding: utf-8 -*-
"""Acquisition coordinator for Aline, Bline, and Cscan workflows."""

from PyQt5.QtCore import QThread
import datetime
import time
from queue import Empty
import traceback

import matplotlib.pyplot as plt
import numpy as np

from Generaic_functions import findchangept
from ActionFields import DnSActionField, AODOActionField, GPUActionField, DActionField
from ActionTypes import AcqTypes, DnSActions, EXIT_ACTION, GPUActions, WeaverActions
from CameraUi import camera_pixel_format, camera_sample_count


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

CONTINUOUS_MODES = (
    AcqTypes.CONTINUOUS_ALINE,
    AcqTypes.CONTINUOUS_BLINE,
    AcqTypes.TRIGGERED_ACQUIRE,
    AcqTypes.CONTINUOUS_CSCAN,
)


class WeaverThread(QThread):
    def __init__(self):
        super().__init__()
        self.exit_message = "Acquisition thread exited."

    def run(self):
        if getattr(self, "file_naming", None) is None:
            raise RuntimeError("WeaverThread.file_naming must be assigned before starting the thread.")
        self.QueueOut()

    def emit_status(self, message):
        if message is None:
            return
        self.ui_bridge.status_message.emit(str(message))

    def finish_with_message(self, message):
        if message is None:
            return
        print(str(message))
        self.emit_status(message)

    def QueueOut(self):
        self.item = self.queue.get()
        while self.item.action != EXIT_ACTION:
            try:
                if self.item.action in CONTINUOUS_MODES:
                    if not self.wait_for_processing_barrier(label=f"starting {self.item.action}"):
                        message = f"{self.item.action} stopped by user."
                        self.finish_with_message(message)
                        raise StopIteration
                    self.InitMemory()
                    if self.item.action == AcqTypes.TRIGGERED_ACQUIRE:
                        message = self.TriggeredAcquire()
                    else:
                        message = self.RptScan(DnS_action=self.item.action, acq_mode=self.item.action)
                    self.finish_with_message(message)

                elif self.item.action in ALINE_MODES + BLINE_MODES + CSCAN_MODES:
                    if not self.wait_for_processing_barrier(label=f"starting {self.item.action}"):
                        message = f"{self.item.action} stopped by user."
                        self.finish_with_message(message)
                        raise StopIteration
                    self.InitMemory()
                    message = self.SingleScan(DnS_action=self.item.action, acq_mode=self.item.action)
                    self.finish_with_message(message)

                elif self.item.action == WeaverActions.ZSTAGE_REPEATIBILITY:
                    message = self.ZstageRepeatibility()
                    self.finish_with_message(message)

                elif self.item.action == WeaverActions.GOTO_ZERO:
                    message = self.Gotozero()
                    self.finish_with_message(message)

                elif self.item.action == WeaverActions.GET_BACKGROUND:
                    message = self.get_background()
                    self.finish_with_message(message)

                elif self.item.action == WeaverActions.GET_SURFACE:
                    message = self.get_surfCurve()
                    self.finish_with_message(message)

                else:
                    message = f"Unknown acquisition command: {self.item.action}"
                    self.finish_with_message(message)

            except StopIteration:
                pass
            except Exception:
                message = f"Acquisition command failed: {self.item.action}"
                self.finish_with_message(message)
                print(traceback.format_exc())

            self.wait_for_processing_barrier(
                label=f"finishing {self.item.action}",
                stop_if_run_unchecked=False,
            )
            self.ui.RunButton.setChecked(False)
            self.ui.RunButton.setText("Go")
            self.ui.PauseButton.setChecked(False)
            self.ui.PauseButton.setText("Pause")
            self.ui.RunButton.setEnabled(True)
            self.ui.PauseButton.setEnabled(True)
            self.GPUQueue.put(GPUActionField(GPUActions.CLEAR))
            if self.ui_bridge is not None:
                self.ui_bridge.acquisition_controls_locked.emit(False)
            self.item = self.queue.get()

        self.emit_status(self.exit_message)

    def drain_queue(self, queue, name, keep=None):
        drained = 0
        kept = []
        while True:
            try:
                item = queue.get_nowait()
            except Empty:
                break
            if keep is not None and keep(item):
                kept.append(item)
            else:
                drained += 1
        for item in kept:
            queue.put(item)
        if drained:
            print(f"Cleared {drained} stale item(s) from {name}.")
        return drained

    def drain_continuous_backlog(self, reason=""):
        def keep_gpu_item(item):
            is_fft_action = getattr(item, "action", None) in {GPUActions.GPU, GPUActions.CPU}
            is_continuous_mode = getattr(item, "DnS_action", None) in CONTINUOUS_MODES
            return not (is_fft_action and is_continuous_mode)

        drained_gpu = self.drain_queue(self.GPUQueue, "GPUQueue", keep=keep_gpu_item)
        drained_camera = self.drain_queue(self.DatabackQueue, "DatabackQueue")
        if drained_gpu or drained_camera:
            suffix = f" ({reason})" if reason else ""
            print(
                "Continuous backlog drain complete"
                f"{suffix}: GPUQueue={drained_gpu}, DatabackQueue={drained_camera}"
            )

    def processing_backlog(self):
        gpu_pending = self.GPUQueue.qsize()
        gpu_active = getattr(getattr(self, "gpu_thread", None), "active_tasks", 0)
        dns_pending = self.DnSQueue.qsize()
        dns_active = getattr(getattr(self, "dns_thread", None), "active_tasks", 0)
        return gpu_pending, gpu_active, dns_pending, dns_active

    def current_acq_mode(self):
        return self.ui.ACQMode.currentText()

    def current_fft_device(self):
        return self.ui.FFTDevice.currentText()

    def current_save_enabled(self):
        return self.ui.Save.isChecked()

    def current_dynamic_enabled(self):
        return self.ui.DynCheckBox.isChecked()

    def current_bline_avg(self):
        return max(1, int(self.ui.BlineAVG.value()))

    def current_y_pixels(self):
        return max(1, int(self.ui.Ypixels.value()))

    def current_aline_avg(self):
        return max(1, int(self.ui.AlineAVG.value()))

    def current_display_x_pixels(self):
        return max(1, int(self.ui.AlinesPerBline.value()))

    def current_alines_per_bline(self):
        return self.current_display_x_pixels() * self.current_aline_avg()

    def current_nsamples(self):
        return camera_sample_count(self.ui)

    def current_depth_range(self):
        return int(self.ui.DepthRange.value())

    def current_realtime_dynamic_enabled(self):
        return self.current_dynamic_enabled() and self.ui.RealtimeDynCheckBox.isChecked()

    def request_run_stop(self):
        bridge = getattr(self, "ui_bridge", None)
        if bridge is not None and hasattr(bridge, "run_button_checked"):
            bridge.run_button_checked.emit(False)
        else:
            self.ui.RunButton.setChecked(False)
            self.ui.RunButton.setText("Go")

    def current_pre_avg_factor(self):
        fft_device = self.current_fft_device()
        if fft_device not in ["GPU", "CPU"]:
            return 1
        if self.current_dynamic_enabled():
            gpu_thread = getattr(self, "gpu_thread", None)
            return max(1, int(getattr(gpu_thread, "gpu_pre_avg_factor", 1)))
        return self.current_bline_avg()

    def current_processed_repeat_count(self, raw_frame_count):
        pre_avg_factor = self.current_pre_avg_factor()
        if pre_avg_factor <= 1:
            return int(raw_frame_count)
        complete_frames = (int(raw_frame_count) // pre_avg_factor) * pre_avg_factor
        if complete_frames < pre_avg_factor:
            return int(raw_frame_count)
        return max(1, complete_frames // pre_avg_factor)

    def finalize_partial_dynamic_naming(self, acq_mode):
        if (
            not self.current_save_enabled()
            or not self.current_dynamic_enabled()
            or self.current_realtime_dynamic_enabled()
        ):
            return
        if self.file_naming.dynamic_bline_idx == 1:
            return
        if acq_mode in CSCAN_MODES:
            self.file_naming.increment_cscan()
            self.file_naming.reset_dynamic_bline_idx()

    def build_filename_bundle(self, DnS_action, acq_mode, memory_slot, raw=False):
        if not self.current_save_enabled():
            return {}
        if self.file_naming is None:
            raise RuntimeError("WeaverThread.file_naming is required when saving is enabled.")

        raw_shape = self.Memory[memory_slot].shape
        raw_frames = int(raw_shape[0])
        x_pixels = int(raw_shape[1])
        z_pixels = self.current_nsamples() if raw else self.current_depth_range()
        repeat_count = raw_frames if raw else self.current_processed_repeat_count(raw_frames)
        y_pixels = self.current_y_pixels()
        dynamic_bline_idx = int(self.file_naming.dynamic_bline_idx)

        if acq_mode in ALINE_MODES:
            filename = self.file_naming.get_filename("aline", acq_mode, [repeat_count, x_pixels, z_pixels])
            self.file_naming.increment_aline()
            return {"filename": filename, "log_filename": filename}

        if acq_mode in BLINE_MODES:
            filename = self.file_naming.get_filename("bline", acq_mode, [repeat_count, x_pixels, z_pixels])
            bundle = {"filename": filename, "log_filename": filename}
            if self.current_realtime_dynamic_enabled():
                bundle["dynamic_filename"] = self.file_naming.get_filename(
                    "bline_dyn",
                    acq_mode,
                    [repeat_count, x_pixels, z_pixels],
                )
            self.file_naming.increment_bline()
            return bundle

        if acq_mode in CSCAN_MODES:
            if self.current_dynamic_enabled():
                if self.current_realtime_dynamic_enabled():
                    dynamic_filename = self.file_naming.get_filename(
                        "cscan_dyn",
                        acq_mode,
                        [y_pixels, x_pixels, z_pixels],
                        ypixels=y_pixels,
                    )
                    mean_filename = self.file_naming.get_filename(
                        "cscan_mean",
                        acq_mode,
                        [y_pixels, x_pixels, z_pixels],
                    )
                    if dynamic_bline_idx == y_pixels:
                        self.file_naming.increment_cscan()
                        self.file_naming.reset_dynamic_bline_idx()
                    else:
                        self.file_naming.increment_dynY()
                    return {
                        "dynamic_filename": dynamic_filename,
                        "mean_filename": mean_filename,
                        "log_filename": dynamic_filename,
                    }

                bline_filename = self.file_naming.get_filename(
                    "cscan_bline",
                    acq_mode,
                    [repeat_count, x_pixels, z_pixels],
                    ypixels=y_pixels,
                )
                dyn_filename = self.file_naming.get_filename(
                    "cscan_dyn",
                    acq_mode,
                    [repeat_count, x_pixels, z_pixels],
                    ypixels=y_pixels,
                )
                self.file_naming.advance_cscan_dynamic_bline(y_pixels)
                return {
                    "filename": bline_filename,
                    "dynamic_filename": dyn_filename,
                    "log_filename": bline_filename,
                }

            filename = self.file_naming.get_filename("cscan", acq_mode, [y_pixels, x_pixels, z_pixels])
            self.file_naming.increment_cscan()
            return {"filename": filename, "log_filename": filename}

        return {}

    def wait_for_processing_barrier(self, label="", poll_interval=0.2, stop_if_run_unchecked=True):
        label = label or "the next sample"
        while self.ui.RunButton.isChecked() or not stop_if_run_unchecked:
            gpu_pending, gpu_active, dns_pending, dns_active = self.processing_backlog()
            if gpu_pending <= 0 and gpu_active <= 0 and dns_pending <= 0 and dns_active <= 0:
                return True
            self.emit_status(
                "Waiting for processing of the previous sample before "
                f"{label}. GPU queue={gpu_pending}, GPU active={gpu_active}, "
                f"DnS queue={dns_pending}, DnS active={dns_active}."
            )
            time.sleep(poll_interval)
            if stop_if_run_unchecked and not self.ui.RunButton.isChecked():
                return False
        return False

    def InitMemory(self):
        samples = self.current_nsamples()
        alines_per_bline = self.current_alines_per_bline()
        configured_bline_avg = self.current_bline_avg()
        configured_y_pixels = self.current_y_pixels()
        configured_dynamic = self.current_dynamic_enabled()
        configured_acq_mode = self.current_acq_mode()

        if camera_pixel_format(self.ui) in ["Mono8"]:
            data_type = np.uint8
        else:
            data_type = np.uint16

        for ii in range(self.memoryCount):
            if configured_acq_mode in ALINE_MODES + BLINE_MODES:
                self.Memory[ii] = np.zeros(
                    [configured_bline_avg, alines_per_bline, samples],
                    dtype=data_type,
                )
                self.NAcq = 1
            elif configured_acq_mode == AcqTypes.CONTINUOUS_CSCAN:
                self.Memory[ii] = np.zeros(
                    [configured_y_pixels * configured_bline_avg, alines_per_bline, samples],
                    dtype=data_type,
                )
                self.NAcq = 1
            elif configured_acq_mode == AcqTypes.FINITE_CSCAN:
                if configured_dynamic:
                    self.Memory[ii] = np.zeros(
                        [configured_bline_avg, alines_per_bline, samples],
                        dtype=data_type,
                    )
                    self.NAcq = configured_y_pixels
                else:
                    self.Memory[ii] = np.zeros(
                        [configured_y_pixels * configured_bline_avg, alines_per_bline, samples],
                        dtype=data_type,
                    )
                    self.NAcq = 1

    def SingleScan(self, DnS_action, acq_mode, context=None, skip_save=False):
        if not self.wait_for_processing_barrier(label=f"starting {DnS_action}"):
            return f"{DnS_action} stopped by user."
        self.drain_continuous_backlog(reason=f"before {DnS_action}")
        fft_device = self.current_fft_device()

        self.DQueue.put(DActionField("ConfigureBoard"))
        self.AODOQueue.put(AODOActionField("ConfigTask"))
        self.StagebackQueue.get()

        self.DQueue.put(DActionField("Acquire"))
        self.DbackQueue.get()

        self.AODOQueue.put(AODOActionField("StartTask"))
        self.StagebackQueue.get()

        message = f"{DnS_action} stopped by user."
        for iAcq in range(self.NAcq):
            dynamic_bline_idx = iAcq if (self.current_dynamic_enabled() and acq_mode in CSCAN_MODES) else None
            while self.ui.RunButton.isChecked():
                try:
                    t0=time.time()
                    an_action = self.DatabackQueue.get(timeout=5)
                    print('time to fetch data: ', round(time.time()-t0,3))
                    memory_slot = an_action.memory_slot
                    filename_bundle = self.build_filename_bundle(
                        DnS_action,
                        acq_mode,
                        memory_slot,
                        raw=(fft_device in ["None"]),
                    )
                    if fft_device in ["None"]:
                        self.data = self.Memory[memory_slot].copy()
                        if np.sum(self.data) < 10:
                            message = "No usable spectral data received."
                            print(message)
                        else:
                            self.DnSQueue.put(
                                DnSActionField(
                                    DnS_action,
                                    acq_mode=acq_mode,
                                    data=self.data,
                                    raw=True,
                                    context=context,
                                    dynamic_bline_idx=dynamic_bline_idx,
                                    filename_bundle=filename_bundle,
                                    skip_save=skip_save,
                                )
                            )
                            message = f"{DnS_action} completed."
                    else:
                        self.GPUQueue.put(
                            GPUActionField(
                                action=fft_device,
                                DnS_action=DnS_action,
                                acq_mode=acq_mode,
                                memory_slot=memory_slot,
                                context=context,
                                dynamic_bline_idx=dynamic_bline_idx,
                                filename_bundle=filename_bundle,
                                skip_save=skip_save,
                            )
                        )
                        message = f"{DnS_action} completed."
                    break
                except Empty:
                    print(f"{DnS_action}: waiting for camera data...")

        self.AODOQueue.put(AODOActionField("tryStopTask"))
        self.AODOQueue.put(AODOActionField("CloseTask"))
        self.StagebackQueue.get()
        self.finalize_partial_dynamic_naming(acq_mode)
        self.wait_for_processing_barrier(label=f"finishing {DnS_action}", stop_if_run_unchecked=False)
        print(message)
        return message

    def RptScan(self, DnS_action, acq_mode):
        if not self.wait_for_processing_barrier(label=f"starting {DnS_action}"):
            return f"{DnS_action} stopped by user."
        fft_device = self.current_fft_device()
        frame_rate = self.ui.FrameRate_DH.value()
        if acq_mode in (AcqTypes.CONTINUOUS_ALINE, AcqTypes.CONTINUOUS_BLINE):
            self.ui.FrameRate_DH.setValue(20)

        self.DQueue.put(DActionField("ConfigureBoard"))
        # self.AODOQueue.put(AODOActionField("ConfigTask"))
        # self.StagebackQueue.get()

        data_backs = 0
        skipped_fft_actions = 0

        self.DQueue.put(DActionField("Acquire"))
        self.DbackQueue.get()

        # self.AODOQueue.put(AODOActionField("StartTask"))
        # self.StagebackQueue.get()

        while self.ui.RunButton.isChecked():
            try:
                t0=time.time()
                an_action = self.DatabackQueue.get(timeout=5)
                print('time to fetch data: ', round(time.time()-t0,3))
                memory_slot = an_action.memory_slot
                data_backs += 1
                
                if fft_device in ["None"]:
                    filename_bundle = self.build_filename_bundle(DnS_action, acq_mode, memory_slot, raw=True)
                    self.data = self.Memory[memory_slot].copy()
                    if np.sum(self.data) < 10:
                        message = "No usable spectral data received."
                        print(message)
                    else:
                        self.DnSQueue.put(
                            DnSActionField(
                                DnS_action,
                                acq_mode=acq_mode,
                                data=self.data,
                                raw=True,
                                filename_bundle=filename_bundle,
                                skip_save=False,
                            )
                        )
                else:
                    if self.GPUQueue.qsize() == 0:
                        filename_bundle = self.build_filename_bundle(DnS_action, acq_mode, memory_slot, raw=False)
                        self.GPUQueue.put(
                            GPUActionField(
                                fft_device,
                                DnS_action=DnS_action,
                                acq_mode=acq_mode,
                                memory_slot=memory_slot,
                                filename_bundle=filename_bundle,
                                skip_save=False,
                            )
                        )
                    else:
                        skipped_fft_actions += 1
            except Empty:
                pass

            if self.ui.PauseButton.isChecked():
                self.AODOQueue.put(AODOActionField("StopTask"))
                while self.ui.PauseButton.isChecked() and self.ui.RunButton.isChecked():
                    time.sleep(0.5)
                if not self.ui.PauseButton.isChecked():
                    self.AODOQueue.put(AODOActionField("StartTask"))
                    self.StagebackQueue.get()

        # self.AODOQueue.put(AODOActionField("tryStopTask"))
        # self.AODOQueue.put(AODOActionField("CloseTask"))
        # self.StagebackQueue.get()
        self.finalize_partial_dynamic_naming(acq_mode)

        message = f"{DnS_action} stopped. Received {data_backs} camera buffer(s)."
        if skipped_fft_actions:
            message += f" Skipped {skipped_fft_actions} stale continuous FFT request(s)."
        print(message)
        self.drain_continuous_backlog(reason=f"after {DnS_action}")
        self.GPUQueue.put(GPUActionField(GPUActions.DISPLAY_FFT_ACTIONS))
        self.GPUQueue.put(GPUActionField(GPUActions.DISPLAY_COUNTS, context=DnS_action))
        self.wait_for_processing_barrier(label=f"finishing {DnS_action}", stop_if_run_unchecked=False)
        self.ui.FrameRate_DH.setValue(frame_rate)
        return message

    def read_triggered_input(self):
        self.AODOQueue.put(AODOActionField("ReadDigitalInput"))
        return bool(self.StagebackQueue.get())

    def build_triggered_finalize_filename_bundle(self, acquired_buffers, raw=False):
        if not self.current_save_enabled():
            return {}
        x_pixels = max(1, int(acquired_buffers)) * self.current_display_x_pixels()
        z_pixels = self.current_nsamples() if raw else self.current_depth_range()
        filename = self.file_naming.get_filename(
            "bline",
            AcqTypes.TRIGGERED_ACQUIRE,
            [1, x_pixels, z_pixels],
        )
        bundle = {"filename": filename, "log_filename": filename}
        self.file_naming.increment_bline()
        return bundle

    def TriggeredAcquire(self):
        DnS_action = AcqTypes.TRIGGERED_ACQUIRE
        acq_mode = AcqTypes.TRIGGERED_ACQUIRE
        if not self.wait_for_processing_barrier(label=f"starting {DnS_action}"):
            return f"{DnS_action} stopped by user."
        self.drain_continuous_backlog(reason=f"before {DnS_action}")

        fft_device = self.current_fft_device()
        data_backs = 0
        skipped_fft_actions = 0
        message = f"{DnS_action} stopped before trigger went high."

        self.DQueue.put(DActionField("ConfigureBoard"))

        self.AODOQueue.put(AODOActionField("ConfigDigitalInput"))
        self.StagebackQueue.get()

        try:
            self.emit_status("Waiting for triggeredAcquire digital input to go high...")
            while self.ui.RunButton.isChecked():
                if self.read_triggered_input():
                    break
                time.sleep(0.01)

            if not self.ui.RunButton.isChecked():
                return f"{DnS_action} stopped by user."

            self.DQueue.put(DActionField("Acquire"))
            self.DbackQueue.get()
            message = f"{DnS_action} stopped."

            while self.ui.RunButton.isChecked():
                try:
                    t0 = time.time()
                    an_action = self.DatabackQueue.get(timeout=5)
                    print("time to fetch data: ", round(time.time() - t0, 3))
                    memory_slot = an_action.memory_slot
                    data_backs += 1

                    if fft_device in ["None"]:
                        self.data = self.Memory[memory_slot].copy()
                        if np.sum(self.data) < 10:
                            message = "No usable spectral data received."
                            print(message)
                        else:
                            self.DnSQueue.put(
                                DnSActionField(
                                    DnS_action,
                                    acq_mode=acq_mode,
                                    data=self.data,
                                    raw=True,
                                    filename_bundle={},
                                    skip_save=False,
                                )
                            )
                    else:
                        if self.GPUQueue.qsize() == 0:
                            self.GPUQueue.put(
                                GPUActionField(
                                    fft_device,
                                    DnS_action=DnS_action,
                                    acq_mode=acq_mode,
                                    memory_slot=memory_slot,
                                    filename_bundle={},
                                    skip_save=False,
                                )
                            )
                        else:
                            skipped_fft_actions += 1

                    if not self.read_triggered_input():
                        message = f"{DnS_action} stopped because digital input went low."
                        self.request_run_stop()
                        break
                except Empty:
                    print(f"{DnS_action}: waiting for camera data...")
                    if not self.read_triggered_input():
                        message = f"{DnS_action} stopped because digital input went low."
                        self.request_run_stop()
                        break

                if self.ui.PauseButton.isChecked():
                    while self.ui.PauseButton.isChecked() and self.ui.RunButton.isChecked():
                        time.sleep(0.5)
        finally:
            self.request_run_stop()
            self.AODOQueue.put(AODOActionField("CloseDigitalInput"))
            self.StagebackQueue.get()
            self.drain_continuous_backlog(reason=f"after {DnS_action}")
            self.GPUQueue.put(GPUActionField(GPUActions.DISPLAY_FFT_ACTIONS))
            self.wait_for_processing_barrier(label=f"finishing {DnS_action}", stop_if_run_unchecked=False)
            final_bundle = self.build_triggered_finalize_filename_bundle(
                data_backs,
                raw=(fft_device in ["None"]),
            )
            self.DnSQueue.put(
                DnSActionField(
                    DnSActions.FINALIZE_TRIGGERED_ACQUIRE,
                    acq_mode=AcqTypes.TRIGGERED_ACQUIRE,
                    filename_bundle=final_bundle,
                )
            )
            self.wait_for_processing_barrier(label=f"displaying {DnS_action}", stop_if_run_unchecked=False)
            self.GPUQueue.put(GPUActionField(GPUActions.DISPLAY_COUNTS, context=DnS_action))
            self.wait_for_processing_barrier(label=f"counting {DnS_action}", stop_if_run_unchecked=False)

        message += f" Received {data_backs} camera buffer(s)."
        if skipped_fft_actions:
            message += f" Skipped {skipped_fft_actions} stale FFT request(s)."
        print(message)
        return message

    def get_background(self):
        acq_mode = self.ui.ACQMode.currentText()
        fft_device = self.ui.FFTDevice.currentText()
        bline_avg = self.ui.BlineAVG.value()
        dyn_checked = self.ui.DynCheckBox.isChecked()
        realtime_dyn_checked = self.ui.RealtimeDynCheckBox.isChecked()

        self.ui.ACQMode.setCurrentText(AcqTypes.FINITE_BLINE)
        self.ui.FFTDevice.setCurrentText("None")
        self.ui.BlineAVG.setValue(10)
        self.ui.DynCheckBox.setChecked(False)
        self.ui.RealtimeDynCheckBox.setChecked(False)
        self.ui.RunButton.setChecked(True)

        self.InitMemory()
        self.SingleScan(
            DnS_action=AcqTypes.FINITE_BLINE,
            acq_mode=AcqTypes.FINITE_BLINE,
            skip_save=True,
        )

        xpixels = self.current_alines_per_bline()
        yrpt = self.ui.BlineAVG.value()
        bline = self.data.reshape([yrpt, xpixels, self.current_nsamples()])
        background = np.float32(np.mean(bline, 0))
        filepath = self.timestamped_file("background", "bin")
        with open(filepath, "wb") as fp:
            background.tofile(fp)

        self.ui.BG_DIR.setText(filepath)
        self.ui.ACQMode.setCurrentText(acq_mode)
        self.ui.FFTDevice.setCurrentText(fft_device)
        self.ui.BlineAVG.setValue(bline_avg)
        self.ui.DynCheckBox.setChecked(dyn_checked)
        self.ui.RealtimeDynCheckBox.setChecked(realtime_dyn_checked)

        return "Background measurement completed."

    def get_background_cscan(self):
        acq_mode = self.ui.ACQMode.currentText()
        fft_device = self.ui.FFTDevice.currentText()
        bline_avg = self.ui.BlineAVG.value()
        dyn_checked = self.ui.DynCheckBox.isChecked()
        realtime_dyn_checked = self.ui.RealtimeDynCheckBox.isChecked()

        self.ui.ACQMode.setCurrentText(AcqTypes.FINITE_CSCAN)
        self.ui.FFTDevice.setCurrentText("None")
        self.ui.BlineAVG.setValue(1)
        self.ui.DynCheckBox.setChecked(False)
        self.ui.RealtimeDynCheckBox.setChecked(False)
        self.ui.RunButton.setChecked(True)

        self.InitMemory()
        self.SingleScan(
            DnS_action=AcqTypes.FINITE_CSCAN,
            acq_mode=AcqTypes.FINITE_CSCAN,
            skip_save=True,
        )

        background = np.float32(np.mean(self.data, axis=0))
        filepath = self.timestamped_file("background", "bin")
        with open(filepath, "wb") as fp:
            background.tofile(fp)

        self.ui.BG_DIR.setText(filepath)
        self.ui.ACQMode.setCurrentText(acq_mode)
        self.ui.FFTDevice.setCurrentText(fft_device)
        self.ui.BlineAVG.setValue(bline_avg)
        self.ui.DynCheckBox.setChecked(dyn_checked)
        self.ui.RealtimeDynCheckBox.setChecked(realtime_dyn_checked)
        return "Background measurement completed."

    def get_surfCurve(self):
        acq_mode = self.ui.ACQMode.currentText()
        fft_device = self.ui.FFTDevice.currentText()
        self.ui.ACQMode.setCurrentText(AcqTypes.FINITE_CSCAN)
        self.ui.FFTDevice.setCurrentText("GPU")
        self.ui.DSing.setChecked(True)
        self.ui.RunButton.setChecked(True)

        self.InitMemory()
        self.SingleScan(DnS_action=AcqTypes.FINITE_CSCAN, acq_mode=AcqTypes.FINITE_CSCAN)
        cscan = self.GPU2weaverQueue.get()

        zpixels = self.ui.DepthRange.value()
        xpixels = self.current_alines_per_bline()
        ypixels = self.ui.Ypixels.value() * self.ui.BlineAVG.value()
        cscan = cscan.reshape([ypixels, xpixels, zpixels])

        bline = np.float32(np.mean(cscan, 0))
        surf_curve = np.zeros([xpixels])

        plt.figure()
        plt.imshow(bline)
        plt.title("Bline for finding surface")
        for xx in range(xpixels):
            surf_curve[xx] = findchangept(bline[xx, :], 1)

        surf_curve = surf_curve - min(surf_curve)
        plt.figure()
        plt.plot(surf_curve)
        plt.title("surface")
        plt.figure()

        filepath = self.ui.DIR.toPlainText() + "/" + "surfCurve.bin"
        with open(filepath, "wb") as fp:
            np.uint16(surf_curve).tofile(fp)

        self.ui.Surf_DIR.setText(filepath)
        self.ui.ACQMode.setCurrentText(acq_mode)
        self.ui.FFTDevice.setCurrentText(fft_device)
        self.ui.DSing.setChecked(False)
        return "Surface curve measurement completed."

    def timestamped_file(self, stem, suffix):
        now = datetime.datetime.now()
        stamp = (
            f"{now.year}-{now.month}-{now.day}-"
            f"{now.hour}-{now.minute}-{now.second}"
        )
        return f"{self.ui.DIR.toPlainText()}/{stem}_{stamp}.{suffix}"

    def ZstageRepeatibility(self):
        return "Z stage repeatability test is not part of the trimmed acquisition workflow."

    def Gotozero(self):
        return "Goto zero is not part of the trimmed acquisition workflow."
