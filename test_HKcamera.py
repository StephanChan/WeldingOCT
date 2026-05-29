# -*- coding: utf-8 -*-
"""
Small Hikrobot/MVS line-scan camera test module.

The module can be imported by other code, or run directly to enumerate cameras,
open one camera, configure common acquisition parameters, grab a few frames, and
optionally save them with the MVS SDK image writer.
"""

import ctypes
import os
import platform
import sys
import time
from ctypes import *
from pathlib import Path


MVS_DEV_ROOT = r"D:\Apps\MVS\Development"
MVS_RUNTIME_DIR = r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"

DEFAULT_DEVICE_INDEX = 0
DEFAULT_NUM_FRAMES = 8
DEFAULT_TIMEOUT_MS = None
DEFAULT_OUTPUT_DIR = "HKCamera_test"

DEFAULT_TRIGGER_MODE = "Off"  # "Off", "Software", "LineStart"
DEFAULT_TRIGGER_SELECTOR = "LineStart"
DEFAULT_TRIGGER_SOURCE = "Line1"
DEFAULT_TRIGGER_ACTIVATION = "RisingEdge"
DEFAULT_LINE_SELECTOR = None
DEFAULT_LINE_MODE = None
DEFAULT_LINE_SOURCE = None
DEFAULT_STROBE_ENABLE = None
DEFAULT_LINE_RATE = 60000

# HiK line-scan camera axes:
# camera Width = samples/depth, camera Height = A-lines per B-line.
DEFAULT_NSAMPLES_HK = 400
DEFAULT_ALINES_PER_BLINE = 2048
DEFAULT_WIDTH = DEFAULT_NSAMPLES_HK
DEFAULT_HEIGHT = DEFAULT_ALINES_PER_BLINE
DEFAULT_OFFSET_X = None
DEFAULT_OFFSET_Y = None
DEFAULT_PIXEL_FORMAT = None
DEFAULT_EXPOSURE_AUTO = "Off"
DEFAULT_EXPOSURE_TIME = None
DEFAULT_GAIN_AUTO = "Off"
DEFAULT_GAIN = None
DEFAULT_DIGITAL_SHIFT = 0.0
DEFAULT_IMAGE_COMPRESSION = None
MIN_GRAB_TIMEOUT_MS = 1000
FRAME_TIMEOUT_MARGIN_MS = 1500


def _bootstrap_mvs_sdk():
    if platform.system() == "Windows":
        ctypes.windll.kernel32.SetDllDirectoryW(MVS_RUNTIME_DIR)
        os.environ["PATH"] = MVS_RUNTIME_DIR + os.pathsep + os.environ.get("PATH", "")
        import_dir = os.path.join(MVS_DEV_ROOT, "Samples", "Python", "MvImport")
        if import_dir not in sys.path:
            sys.path.append(import_dir)
    else:
        import_dir = os.path.join("..", "..", "MvImport")
        if import_dir not in sys.path:
            sys.path.append(import_dir)


_bootstrap_mvs_sdk()
from MvCameraControl_class import *  # noqa: E402,F403


def check_ret(ret, message):
    if ret != 0:
        raise RuntimeError("%s ret[0x%x]" % (message, ret))


def decoding_char(ctypes_char_array):
    byte_str = memoryview(ctypes_char_array).tobytes()
    null_index = byte_str.find(b"\x00")
    if null_index != -1:
        byte_str = byte_str[:null_index]

    for encoding in ("gbk", "utf-8", "latin-1"):
        try:
            return byte_str.decode(encoding)
        except UnicodeDecodeError:
            pass
    return byte_str.decode("latin-1", errors="replace")


def device_summary(device_info):
    layer_type = device_info.nTLayerType
    if layer_type in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
        info = device_info.SpecialInfo.stGigEInfo
        ip = ".".join(
            str((info.nCurrentIp >> shift) & 0xFF)
            for shift in (24, 16, 8, 0)
        )
        return {
            "type": "GigE",
            "model": decoding_char(info.chModelName),
            "serial": decoding_char(info.chSerialNumber),
            "user_name": decoding_char(info.chUserDefinedName),
            "ip": ip,
        }
    if layer_type == MV_USB_DEVICE:
        info = device_info.SpecialInfo.stUsb3VInfo
        return {
            "type": "USB3",
            "model": decoding_char(info.chModelName),
            "serial": decoding_char(info.chSerialNumber),
            "user_name": decoding_char(info.chUserDefinedName),
            "ip": "",
        }
    if layer_type == MV_GENTL_CAMERALINK_DEVICE:
        info = device_info.SpecialInfo.stCMLInfo
        camera_type = "CameraLink"
    elif layer_type == MV_GENTL_CXP_DEVICE:
        info = device_info.SpecialInfo.stCXPInfo
        camera_type = "CXP"
    elif layer_type == MV_GENTL_XOF_DEVICE:
        info = device_info.SpecialInfo.stXoFInfo
        camera_type = "XoF"
    else:
        return {"type": "Unknown", "model": "", "serial": "", "user_name": "", "ip": ""}

    return {
        "type": camera_type,
        "model": decoding_char(info.chModelName),
        "serial": decoding_char(info.chSerialNumber),
        "user_name": decoding_char(info.chUserDefinedName),
        "ip": "",
    }


def enum_devices():
    device_list = MV_CC_DEVICE_INFO_LIST()
    layer_type = (
        MV_GIGE_DEVICE
        | MV_USB_DEVICE
        | MV_GENTL_CAMERALINK_DEVICE
        | MV_GENTL_CXP_DEVICE
        | MV_GENTL_XOF_DEVICE
    )
    ret = MvCamera.MV_CC_EnumDevices(layer_type, device_list)
    check_ret(ret, "enum devices failed")
    return device_list


def print_devices(device_list):
    print("find %d devices" % device_list.nDeviceNum)
    for index in range(device_list.nDeviceNum):
        info = cast(device_list.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents
        summary = device_summary(info)
        print(
            "[%d] %s model=%s serial=%s user=%s ip=%s"
            % (
                index,
                summary["type"],
                summary["model"],
                summary["serial"],
                summary["user_name"],
                summary["ip"],
            )
        )


def check_feature_node_access(cam, node_name):
    access_mode = MV_XML_AccessMode()
    ret = cam.MV_XML_GetNodeAccessMode(node_name, access_mode)
    if ret != 0:
        return False
    return access_mode.value in (AM_WO, AM_RO, AM_RW)


def set_if_not_none(cam, setter, name, value):
    if value is None:
        return
    ret = setter(name, value)
    check_ret(ret, "set %s failed" % name)


def warn_ret(ret, message):
    if ret != 0:
        print("warning: %s ret[0x%x]" % (message, ret))
        return False
    return True


class HKLineScanCamera:
    def __init__(self, device_index=DEFAULT_DEVICE_INDEX):
        self.device_index = int(device_index)
        self.cam = None
        self.device_info = None
        self.is_open = False
        self.is_grabbing = False
        self.decode_hb = False

    def open(self):
        device_list = enum_devices()
        if device_list.nDeviceNum == 0:
            raise RuntimeError("find no device")
        if self.device_index >= device_list.nDeviceNum:
            raise ValueError("device_index %d is out of range" % self.device_index)

        self.device_info = cast(
            device_list.pDeviceInfo[self.device_index],
            POINTER(MV_CC_DEVICE_INFO),
        ).contents

        self.cam = MvCamera()
        ret = self.cam.MV_CC_CreateHandle(self.device_info)
        check_ret(ret, "create handle failed")

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        check_ret(ret, "open device failed")
        self.is_open = True

        if self.device_info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
            packet_size = self.cam.MV_CC_GetOptimalPacketSize()
            if int(packet_size) > 0:
                ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)
                if ret != 0:
                    print("warning: set packet size failed ret[0x%x]" % ret)
            else:
                print("warning: get packet size failed ret[0x%x]" % packet_size)
        return self

    def close(self):
        if self.cam is None:
            return
        if self.is_grabbing:
            self.stop_grabbing()
        if self.is_open:
            ret = self.cam.MV_CC_CloseDevice()
            if ret != 0:
                print("warning: close device failed ret[0x%x]" % ret)
            self.is_open = False
        ret = self.cam.MV_CC_DestroyHandle()
        if ret != 0:
            print("warning: destroy handle failed ret[0x%x]" % ret)
        self.cam = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def set_enum(self, name, value, strict=True):
        ret = self.cam.MV_CC_SetEnumValueByString(name, value)
        if strict:
            check_ret(ret, "set %s to %s failed" % (name, value))
            print("set %s = %s" % (name, value))
            return True
        ok = warn_ret(ret, "set %s to %s failed" % (name, value))
        if ok:
            print("set %s = %s" % (name, value))
        return ok

    def set_bool(self, name, value, strict=True):
        ret = self.cam.MV_CC_SetBoolValue(name, bool(value))
        if strict:
            check_ret(ret, "set %s failed" % name)
            print("set %s = %s" % (name, bool(value)))
            return True
        ok = warn_ret(ret, "set %s failed" % name)
        if ok:
            print("set %s = %s" % (name, bool(value)))
        return ok

    def set_int(self, name, value, strict=True):
        ret = self.cam.MV_CC_SetIntValue(name, int(value))
        if strict:
            check_ret(ret, "set %s failed" % name)
            print("set %s = %d" % (name, int(value)))
            return True
        ok = warn_ret(ret, "set %s failed" % name)
        if ok:
            print("set %s = %d" % (name, int(value)))
        return ok

    def set_float(self, name, value, strict=True):
        ret = self.cam.MV_CC_SetFloatValue(name, float(value))
        if strict:
            check_ret(ret, "set %s failed" % name)
            print("set %s = %.6g" % (name, float(value)))
            return True
        ok = warn_ret(ret, "set %s failed" % name)
        if ok:
            print("set %s = %.6g" % (name, float(value)))
        return ok

    def get_int_feature(self, name):
        value = MVCC_INTVALUE()
        memset(byref(value), 0, sizeof(value))
        ret = self.cam.MV_CC_GetIntValue(name, value)
        if ret != 0:
            return None
        return value

    def get_float_feature(self, name):
        value = MVCC_FLOATVALUE()
        memset(byref(value), 0, sizeof(value))
        ret = self.cam.MV_CC_GetFloatValue(name, value)
        if ret != 0:
            return None
        return value

    def get_string_feature(self, name):
        value = MVCC_STRINGVALUE()
        memset(byref(value), 0, sizeof(value))
        ret = self.cam.MV_CC_GetStringValue(name, value)
        if ret != 0:
            return None
        return decoding_char(value.chCurValue)

    def print_int_feature(self, name):
        label = name
        feature_name = name.strip().split()[-1]
        value = self.get_int_feature(feature_name)
        if value is None:
            print("%s: unavailable" % label)
            return
        print(
            "%s: current=%d min=%d max=%d inc=%d"
            % (label, value.nCurValue, value.nMin, value.nMax, value.nInc)
        )

    def print_float_feature(self, name):
        label = name
        feature_name = name.strip().split()[-1]
        value = self.get_float_feature(feature_name)
        if value is None:
            print("%s: unavailable" % label)
            return
        print(
            "%s: current=%.6g min=%.6g max=%.6g"
            % (label, value.fCurValue, value.fMin, value.fMax)
        )

    def print_basic_status(self):
        print("camera status/readout:")
        for name in ("DeviceModelName", "DeviceSerialNumber", "DeviceUserID"):
            value = self.get_string_feature(name)
            if value:
                print("  %s: %s" % (name, value))
        for name in ("Width", "Height", "OffsetX", "OffsetY", "PayloadSize"):
            self.print_int_feature("  " + name)
        self.print_float_feature("  DeviceTemperature")

    def configure_roi(
        self,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        offset_x=DEFAULT_OFFSET_X,
        offset_y=DEFAULT_OFFSET_Y,
    ):
        print("configure ROI:")
        for name in ("Width", "Height", "OffsetX", "OffsetY"):
            self.print_int_feature("  before " + name)

        # Offsets must usually be set before shrinking/growing ROI.
        if offset_x is not None:
            self.set_int("OffsetX", offset_x, strict=False)
        if offset_y is not None:
            self.set_int("OffsetY", offset_y, strict=False)
        if width is not None:
            self.set_int("Width", width, strict=True)
        if height is not None:
            self.set_int("Height", height, strict=True)

        for name in ("Width", "Height", "OffsetX", "OffsetY"):
            self.print_int_feature("  after " + name)

    def configure_pixel_format(self, pixel_format=DEFAULT_PIXEL_FORMAT):
        print("configure pixel format:")
        if pixel_format:
            self.set_enum("PixelFormat", pixel_format, strict=True)
        else:
            print("PixelFormat unchanged")

    def configure_exposure_gain(
        self,
        exposure_auto=DEFAULT_EXPOSURE_AUTO,
        exposure_time=DEFAULT_EXPOSURE_TIME,
        gain_auto=DEFAULT_GAIN_AUTO,
        gain=DEFAULT_GAIN,
    ):
        print("configure exposure/gain:")
        if exposure_auto is not None:
            self.set_enum("ExposureAuto", exposure_auto, strict=False)
        if exposure_auto == "Off" and exposure_time is not None:
            self.set_float("ExposureTime", exposure_time, strict=True)

        if gain_auto is not None:
            self.set_enum("GainAuto", gain_auto, strict=False)
        if gain_auto == "Off" and gain is not None:
            self.set_float("Gain", gain, strict=True)

        self.print_float_feature("  ExposureTime")
        self.print_float_feature("  Gain")

    def configure_line_rate(self, line_rate=DEFAULT_LINE_RATE, enable=True):
        print("configure line rate:")
        if line_rate is None:
            self.set_bool("AcquisitionLineRateEnable", False, strict=False)
            return
        self.set_int("AcquisitionLineRate", line_rate, strict=True)
        self.set_bool("AcquisitionLineRateEnable", enable, strict=False)
        self.print_int_feature("  AcquisitionLineRate")
        self.print_int_feature("  ResultingLineRate")

    def configure_trigger(
        self,
        trigger_mode=DEFAULT_TRIGGER_MODE,
        trigger_selector=DEFAULT_TRIGGER_SELECTOR,
        trigger_source=DEFAULT_TRIGGER_SOURCE,
        trigger_activation=DEFAULT_TRIGGER_ACTIVATION,
    ):
        print("configure trigger:")
        if trigger_mode == "Off":
            self.set_enum("TriggerSelector", trigger_selector, strict=False)
            ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            check_ret(ret, "set trigger mode off failed")
            print("set TriggerMode = Off")
            return

        if trigger_mode == "Software":
            self.set_enum("TriggerSelector", trigger_selector, strict=False)
            self.set_enum("TriggerSource", "Software", strict=True)
            self.set_enum("TriggerActivation", trigger_activation, strict=False)
            ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
            check_ret(ret, "set trigger mode on failed")
            print("set TriggerMode = On")
            return

        if trigger_mode in ("LineStart", "Hardware"):
            self.set_enum("TriggerSelector", trigger_selector, strict=True)
            self.set_enum("TriggerSource", trigger_source, strict=True)
            self.set_enum("TriggerActivation", trigger_activation, strict=False)
            ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
            check_ret(ret, "set trigger mode on failed")
            print("set TriggerMode = On")
            return

        raise ValueError("unsupported trigger_mode: %s" % trigger_mode)

    def configure_image_compression(self, image_compression=DEFAULT_IMAGE_COMPRESSION):
        print("configure image compression:")
        if image_compression is None:
            print("ImageCompressionMode unchanged")
            self.decode_hb = False
            return
        ok = self.set_enum("ImageCompressionMode", image_compression, strict=False)
        self.decode_hb = ok and image_compression == "HB"

    def configure_digital_io(
        self,
        line_selector=DEFAULT_LINE_SELECTOR,
        line_mode=DEFAULT_LINE_MODE,
        line_source=DEFAULT_LINE_SOURCE,
        strobe_enable=DEFAULT_STROBE_ENABLE,
    ):
        print("configure digital IO:")
        if line_selector is None and line_mode is None and line_source is None:
            print("LineSelector/LineMode/LineSource unchanged")
        if line_selector is not None:
            self.set_enum("LineSelector", line_selector, strict=False)
        if line_mode is not None:
            self.set_enum("LineMode", line_mode, strict=False)
        if line_source is not None:
            self.set_enum("LineSource", line_source, strict=False)
        if strobe_enable is not None:
            self.set_bool("StrobeEnable", strobe_enable, strict=False)

        line_status = self.get_int_feature("LineStatusAll")
        if line_status is not None:
            print("  LineStatusAll: %d" % line_status.nCurValue)

    def configure(
        self,
        trigger_mode=DEFAULT_TRIGGER_MODE,
        trigger_selector=DEFAULT_TRIGGER_SELECTOR,
        trigger_source=DEFAULT_TRIGGER_SOURCE,
        trigger_activation=DEFAULT_TRIGGER_ACTIVATION,
        line_selector=DEFAULT_LINE_SELECTOR,
        line_mode=DEFAULT_LINE_MODE,
        line_source=DEFAULT_LINE_SOURCE,
        strobe_enable=DEFAULT_STROBE_ENABLE,
        line_rate=DEFAULT_LINE_RATE,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        offset_x=DEFAULT_OFFSET_X,
        offset_y=DEFAULT_OFFSET_Y,
        pixel_format=DEFAULT_PIXEL_FORMAT,
        exposure_auto=DEFAULT_EXPOSURE_AUTO,
        exposure_time=DEFAULT_EXPOSURE_TIME,
        gain_auto=DEFAULT_GAIN_AUTO,
        gain=DEFAULT_GAIN,
        digital_shift=DEFAULT_DIGITAL_SHIFT,
        image_compression=DEFAULT_IMAGE_COMPRESSION,
    ):
        if self.cam is None:
            raise RuntimeError("camera is not open")

        self.configure_trigger(
            trigger_mode=trigger_mode,
            trigger_selector=trigger_selector,
            trigger_source=trigger_source,
            trigger_activation=trigger_activation,
        )
        self.configure_digital_io(
            line_selector=line_selector,
            line_mode=line_mode,
            line_source=line_source,
            strobe_enable=strobe_enable,
        )
        self.configure_roi(
            width=width,
            height=height,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        self.configure_pixel_format(pixel_format)
        self.configure_line_rate(line_rate)
        self.configure_exposure_gain(
            exposure_auto=exposure_auto,
            exposure_time=exposure_time,
            gain_auto=gain_auto,
            gain=gain,
        )

        if digital_shift is not None:
            print("configure digital shift:")
            self.set_bool("DigitalShiftEnable", True, strict=False)
            self.set_float("DigitalShift", digital_shift, strict=False)

        self.configure_image_compression(image_compression)
        self.print_basic_status()

    def configure_software_trigger(self):
        ret = self.cam.MV_CC_SetEnumValueByString("ScanMode", "FrameScan")
        if ret == 0:
            print("set frame scan mode")

        if check_feature_node_access(self.cam, "FrameTriggerControl"):
            ret = self.cam.MV_CC_SetBoolValue("FrameTriggerMode", True)
            check_ret(ret, "set frame trigger mode failed")
            ret = self.cam.MV_CC_SetEnumValueByString("FrameTriggerSource", "Software")
            check_ret(ret, "set frame trigger source failed")
        else:
            ret = self.cam.MV_CC_SetEnumValueByString("TriggerSelector", "FrameBurstStart")
            check_ret(ret, "set trigger selector failed")
            ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
            check_ret(ret, "set trigger mode failed")
            ret = self.cam.MV_CC_SetEnumValueByString("TriggerSource", "Software")
            check_ret(ret, "set trigger source failed")

    def configure_line_start_trigger(self):
        ret = self.cam.MV_CC_SetEnumValueByString("TriggerSelector", "LineStart")
        check_ret(ret, "set trigger selector failed")
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
        check_ret(ret, "set trigger mode failed")
        ret = self.cam.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_LINE1)
        check_ret(ret, "set trigger source failed")

    def software_trigger(self):
        if check_feature_node_access(self.cam, "FrameTriggerControl"):
            command = "FrameTriggerSoftware"
        else:
            command = "TriggerSoftware"
        ret = self.cam.MV_CC_SetCommandValue(command)
        check_ret(ret, "software trigger failed")

    def get_payload_size(self):
        param = MVCC_INTVALUE()
        memset(byref(param), 0, sizeof(MVCC_INTVALUE))
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", param)
        check_ret(ret, "get payload size failed")
        return param.nCurValue

    def estimate_frame_timeout_ms(self):
        height = self.get_int_feature("Height")
        line_rate = self.get_int_feature("AcquisitionLineRate")
        if height is None or line_rate is None or line_rate.nCurValue <= 0:
            return MIN_GRAB_TIMEOUT_MS

        frame_time_ms = int((height.nCurValue / float(line_rate.nCurValue)) * 1000.0)
        return max(MIN_GRAB_TIMEOUT_MS, frame_time_ms + FRAME_TIMEOUT_MARGIN_MS)

    def start_grabbing(self):
        ret = self.cam.MV_CC_StartGrabbing()
        check_ret(ret, "start grabbing failed")
        self.is_grabbing = True

    def stop_grabbing(self):
        ret = self.cam.MV_CC_StopGrabbing()
        if ret != 0:
            print("warning: stop grabbing failed ret[0x%x]" % ret)
        self.is_grabbing = False

    def grab_frame(self, timeout_ms=DEFAULT_TIMEOUT_MS, decode_hb=True):
        if timeout_ms is None:
            timeout_ms = self.estimate_frame_timeout_ms()
        frame = MV_FRAME_OUT()
        memset(byref(frame), 0, sizeof(frame))

        ret = self.cam.MV_CC_GetImageBuffer(frame, int(timeout_ms))
        if frame.pBufAddr is None or ret != 0:
            raise TimeoutError(
                "get image buffer failed ret[0x%x] timeout[%d ms]"
                % (ret, int(timeout_ms))
            )

        try:
            if not decode_hb:
                return frame, None

            payload_size = self.get_payload_size()
            decode_param = MV_CC_HB_DECODE_PARAM()
            decode_param.pSrcBuf = frame.pBufAddr
            decode_param.nSrcLen = frame.stFrameInfo.nFrameLen
            decode_param.nDstBufSize = payload_size
            decode_param.pDstBuf = (c_ubyte * payload_size)()

            ret = self.cam.MV_CC_HBDecode(decode_param)
            check_ret(ret, "high bandwidth decode failed")
            return frame, decode_param
        except Exception:
            self.cam.MV_CC_FreeImageBuffer(frame)
            raise

    def free_frame(self, frame):
        ret = self.cam.MV_CC_FreeImageBuffer(frame)
        if ret != 0:
            print("warning: free image buffer failed ret[0x%x]" % ret)

    def save_decoded_frame(self, decode_param, file_path):
        file_path = str(file_path)
        save_param = MV_SAVE_IMAGE_TO_FILE_PARAM_EX()
        memset(byref(save_param), 0, sizeof(save_param))
        save_param.nWidth = decode_param.nWidth
        save_param.nHeight = decode_param.nHeight
        save_param.pData = decode_param.pDstBuf
        save_param.enImageType = MV_Image_Bmp
        save_param.nDataLen = decode_param.nDstBufLen
        save_param.enPixelType = decode_param.enDstPixelType
        save_param.nQuality = 80
        save_param.iMethodValue = 3
        save_param.pcImagePath = ctypes.create_string_buffer(file_path.encode("ascii"))

        ret = self.cam.MV_CC_SaveImageToFileEx(save_param)
        check_ret(ret, "save image failed")

    def save_raw_frame(self, frame, file_path):
        file_path = str(file_path)
        info = frame.stFrameInfo
        save_param = MV_SAVE_IMAGE_TO_FILE_PARAM_EX()
        memset(byref(save_param), 0, sizeof(save_param))
        save_param.nWidth = info.nWidth
        save_param.nHeight = info.nHeight
        save_param.pData = frame.pBufAddr
        save_param.enImageType = MV_Image_Bmp
        save_param.nDataLen = info.nFrameLen
        save_param.enPixelType = info.enPixelType
        save_param.nQuality = 80
        save_param.iMethodValue = 3
        save_param.pcImagePath = ctypes.create_string_buffer(file_path.encode("ascii"))

        ret = self.cam.MV_CC_SaveImageToFileEx(save_param)
        check_ret(ret, "save image failed")

    def grab_test_frames(
        self,
        num_frames=DEFAULT_NUM_FRAMES,
        output_dir=DEFAULT_OUTPUT_DIR,
        timeout_ms=DEFAULT_TIMEOUT_MS,
        save_images=True,
        software_trigger=False,
    ):
        output_path = Path(output_dir)
        if save_images:
            output_path.mkdir(parents=True, exist_ok=True)

        if timeout_ms is None:
            timeout_ms = self.estimate_frame_timeout_ms()
        print("grab timeout = %d ms" % int(timeout_ms))

        self.start_grabbing()
        results = []
        try:
            for index in range(int(num_frames)):
                if software_trigger:
                    self.software_trigger()
                    time.sleep(0.02)

                frame, decoded = self.grab_frame(
                    timeout_ms=timeout_ms,
                    decode_hb=self.decode_hb,
                )
                try:
                    info = frame.stFrameInfo
                    pixel_type = decoded.enDstPixelType if decoded else info.enPixelType
                    print(
                        "frame %d: Width[%d], Height[%d], FrameNum[%d], PixelType[%d]"
                        % (
                            index + 1,
                            info.nWidth,
                            info.nHeight,
                            info.nFrameNum,
                            pixel_type,
                        )
                    )
                    if save_images:
                        image_path = output_path / ("%03d.bmp" % (index + 1))
                        if decoded:
                            self.save_decoded_frame(decoded, image_path)
                        else:
                            self.save_raw_frame(frame, image_path)
                    results.append((info.nWidth, info.nHeight, info.nFrameNum))
                finally:
                    self.free_frame(frame)
        finally:
            self.stop_grabbing()
        return results


def run_test(
    device_index=DEFAULT_DEVICE_INDEX,
    num_frames=DEFAULT_NUM_FRAMES,
    trigger_mode=DEFAULT_TRIGGER_MODE,
    trigger_selector=DEFAULT_TRIGGER_SELECTOR,
    trigger_source=DEFAULT_TRIGGER_SOURCE,
    trigger_activation=DEFAULT_TRIGGER_ACTIVATION,
    line_selector=DEFAULT_LINE_SELECTOR,
    line_mode=DEFAULT_LINE_MODE,
    line_source=DEFAULT_LINE_SOURCE,
    strobe_enable=DEFAULT_STROBE_ENABLE,
    width=DEFAULT_WIDTH,
    height=DEFAULT_HEIGHT,
    offset_x=DEFAULT_OFFSET_X,
    offset_y=DEFAULT_OFFSET_Y,
    pixel_format=DEFAULT_PIXEL_FORMAT,
    exposure_auto=DEFAULT_EXPOSURE_AUTO,
    exposure_time=DEFAULT_EXPOSURE_TIME,
    gain_auto=DEFAULT_GAIN_AUTO,
    gain=DEFAULT_GAIN,
    line_rate=DEFAULT_LINE_RATE,
    digital_shift=DEFAULT_DIGITAL_SHIFT,
    image_compression=DEFAULT_IMAGE_COMPRESSION,
    output_dir=DEFAULT_OUTPUT_DIR,
    save_images=True,
):
    MvCamera.MV_CC_Initialize()
    try:
        version = MvCamera.MV_CC_GetSDKVersion()
        print("SDKVersion[0x%x]" % version)

        device_list = enum_devices()
        print_devices(device_list)

        with HKLineScanCamera(device_index=device_index) as camera:
            camera.configure(
                trigger_mode=trigger_mode,
                trigger_selector=trigger_selector,
                trigger_source=trigger_source,
                trigger_activation=trigger_activation,
                line_selector=line_selector,
                line_mode=line_mode,
                line_source=line_source,
                strobe_enable=strobe_enable,
                line_rate=line_rate,
                width=width,
                height=height,
                offset_x=offset_x,
                offset_y=offset_y,
                pixel_format=pixel_format,
                exposure_auto=exposure_auto,
                exposure_time=exposure_time,
                gain_auto=gain_auto,
                gain=gain,
                digital_shift=digital_shift,
                image_compression=image_compression,
            )
            return camera.grab_test_frames(
                num_frames=num_frames,
                output_dir=output_dir,
                save_images=save_images,
                software_trigger=(trigger_mode == "Software"),
            )
    finally:
        MvCamera.MV_CC_Finalize()

def main():
    # Edit these arguments to mirror the quick Daheng test configuration.
    # Examples:
    #   run_test(width=400, height=2032, pixel_format="Mono8")
    #   # width is NSamples_HK/depth, height is AlinesPerBline.
    #   run_test(trigger_mode="Hardware", trigger_selector="LineStart",
    #            trigger_source="Line1", trigger_activation="RisingEdge")
    #   run_test(exposure_time=1000.0, gain=1.0, line_rate=10000)
    #   run_test(image_compression="Off")  # only if the node is writable
    run_test()


if __name__ == "__main__":
    main()
