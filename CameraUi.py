"""Helpers for reading camera-specific UI settings.

The Camera combobox is intentionally limited to these user-facing names:
Daheng, PhotonFocus, and HiK.
"""

CAMERA_DAHENG = "Daheng"
CAMERA_PHOTONFOCUS = "PhotonFocus"
CAMERA_HIK = "HiK"
SUPPORTED_CAMERA_NAMES = (CAMERA_DAHENG, CAMERA_PHOTONFOCUS, CAMERA_HIK)


def current_camera_name(ui):
    camera = getattr(ui, "Camera", None)
    if camera is None or not hasattr(camera, "currentText"):
        return ""
    return camera.currentText()


def current_camera_is_daheng(ui):
    return current_camera_name(ui) == CAMERA_DAHENG


def current_camera_is_photonfocus(ui):
    return current_camera_name(ui) == CAMERA_PHOTONFOCUS


def current_camera_is_hik(ui):
    return current_camera_name(ui) == CAMERA_HIK


def widget_value(ui, name, default=None):
    widget = getattr(ui, name, None)
    if widget is None:
        return default
    if hasattr(widget, "value"):
        return widget.value()
    if hasattr(widget, "currentText"):
        return widget.currentText()
    if hasattr(widget, "text"):
        return widget.text()
    return default


def spectral_downsample(ui):
    if current_camera_is_hik(ui):
        return max(1, int(widget_value(ui, "SpectralDS_HK", 1)))
    if current_camera_is_photonfocus(ui):
        return max(1, int(widget_value(ui, "SpectralDS_PF", 1)))
    return max(1, int(widget_value(ui, "SpectralDS_DH", 1)))


def raw_camera_sample_count(ui):
    if current_camera_is_hik(ui):
        return int(widget_value(ui, "NSamples_HK", 1024))
    if current_camera_is_photonfocus(ui):
        return int(widget_value(ui, "NSamples_PF", 1024))
    return int(widget_value(ui, "NSamples_DH", 1024))


def camera_sample_count(ui):
    raw_samples = raw_camera_sample_count(ui)
    ds = spectral_downsample(ui)
    if raw_samples % ds != 0:
        raise ValueError(
            "SpectralDS must divide the selected camera sample count: "
            f"raw_samples={raw_samples}, SpectralDS={ds}"
        )
    return raw_samples // ds


def camera_pixel_format(ui):
    if current_camera_is_hik(ui):
        return str(widget_value(ui, "PixelFormat_display_HK", "Mono12"))
    if current_camera_is_photonfocus(ui):
        return str(widget_value(ui, "PixelFormat_display_PF", "Mono12"))
    return str(widget_value(ui, "PixelFormat_display_DH", "Mono12"))


def downsample_spectral_axis(data, ratio, axis):
    ratio = max(1, int(ratio))
    if ratio == 1:
        return data
    samples = int(data.shape[axis])
    if samples % ratio != 0:
        raise ValueError(
            f"SpectralDS={ratio} must divide spectral samples={samples}"
        )
    out_samples = samples // ratio
    moved = data.swapaxes(axis, -1)
    out_shape = moved.shape[:-1] + (out_samples, ratio)
    downsampled = moved.reshape(out_shape).mean(axis=-1)
    downsampled = downsampled.swapaxes(axis, -1)
    return downsampled.astype(data.dtype, copy=False)
