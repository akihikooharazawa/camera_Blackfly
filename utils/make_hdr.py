import numpy as np
import cv2


def make_hdr_debevec(frames, exposures_us, bayer_pattern=cv2.COLOR_BayerRG2BGR,
                     gamma=2.2, max_val=65535):
    """
    Build an HDR image using OpenCV's Debevec pipeline.

    Args:
        frames:        Bayer image stack (H, W, N), dtype uint16.
        exposures_us:  Exposure times in microseconds, shape (N,).
        bayer_pattern: OpenCV Bayer conversion code.
        gamma:         Gamma for Reinhard tone mapping.
        max_val:       Sensor max value (65535 for BayerRG16).

    Returns:
        hdr: HDR radiance map (H, W, 3), float32.
        ldr: Tone-mapped preview (H, W, 3), uint8.
    """
    frames = np.asarray(frames)
    exposures_us = np.asarray(exposures_us, dtype=np.float32)

    if frames.ndim != 3:
        raise ValueError("frames must have shape (H, W, N)")
    if frames.shape[2] != exposures_us.shape[0]:
        raise ValueError("frames.shape[2] must match len(exposures_us)")

    # OpenCV HDR functions expect exposure times in seconds
    times = exposures_us / 1e6

    order = np.argsort(times)
    times = times[order]
    frames = frames[..., order]

    img_list = []
    for i in range(frames.shape[2]):
        bayer16 = np.clip(frames[..., i].astype(np.uint16), 0, max_val)
        bgr16 = cv2.cvtColor(bayer16, bayer_pattern)
        bgr8 = np.round(bgr16.astype(np.float32) * (255.0 / max_val)).astype(np.uint8)
        img_list.append(bgr8)

    calibrate = cv2.createCalibrateDebevec()
    response = calibrate.process(img_list, times=times)

    merge = cv2.createMergeDebevec()
    hdr = merge.process(img_list, times=times, response=response)

    tonemap = cv2.createTonemapReinhard(gamma=gamma)
    ldr = np.clip(tonemap.process(hdr.copy()) * 255.0, 0, 255).astype(np.uint8)

    return hdr.astype(np.float32), ldr


def make_hdr_linear_raw(frames, exposures_us, bayer_pattern=cv2.COLOR_BayerRG2BGR,
                        gamma=2.2, max_val=65535, valid_min=0.02, valid_max=0.98):
    """
    Build an HDR image directly from linear raw Bayer data.

    Normalizes each frame by its exposure time and fuses valid radiance
    estimates across exposures using triangle weighting.

    Args:
        frames:        Bayer image stack (H, W, N), dtype uint16.
        exposures_us:  Exposure times in microseconds, shape (N,).
        bayer_pattern: OpenCV Bayer conversion code.
        gamma:         Gamma for Reinhard tone mapping.
        max_val:       Sensor max value (65535 for BayerRG16).
        valid_min:     Lower normalized threshold (rejects underexposed pixels).
        valid_max:     Upper normalized threshold (rejects saturated pixels).

    Returns:
        hdr: HDR radiance map (H, W, 3), float32.
        ldr: Tone-mapped preview (H, W, 3), uint8.
    """
    frames = np.asarray(frames)
    exposures_us = np.asarray(exposures_us, dtype=np.float32)

    if frames.ndim != 3:
        raise ValueError("frames must have shape (H, W, N)")
    if frames.shape[2] != exposures_us.shape[0]:
        raise ValueError("frames.shape[2] must match len(exposures_us)")

    times = exposures_us / 1e6

    order = np.argsort(times)
    times = times[order]
    frames = frames[..., order]

    radiance_list = []
    weight_list = []

    for i in range(frames.shape[2]):
        bayer16 = np.clip(frames[..., i].astype(np.uint16), 0, max_val)
        bgr16 = cv2.cvtColor(bayer16, bayer_pattern)
        img = bgr16.astype(np.float32) / max_val

        rad = img / times[i]

        weight = 1.0 - np.abs(2.0 * img - 1.0)
        weight = np.where((img > valid_min) & (img < valid_max), weight, 0.0).astype(np.float32)

        radiance_list.append(rad)
        weight_list.append(weight)

    radiance_stack = np.stack(radiance_list, axis=-1)   # (H, W, 3, N)
    weight_stack = np.stack(weight_list, axis=-1)       # (H, W, 3, N)

    weight_sum = np.sum(weight_stack, axis=-1)
    hdr = np.sum(radiance_stack * weight_stack, axis=-1) / np.maximum(weight_sum, 1e-8)

    fallback = np.mean(radiance_stack, axis=-1)
    hdr = np.where(weight_sum > 0, hdr, fallback).astype(np.float32)

    tonemap = cv2.createTonemapReinhard(gamma=gamma)
    ldr = np.clip(tonemap.process(hdr.copy()) * 255.0, 0, 255).astype(np.uint8)

    return hdr, ldr


def make_hdr(frames, exposures_us, method="linear_raw",
             bayer_pattern=cv2.COLOR_BayerRG2BGR, gamma=2.2,
             max_val=65535, valid_min=0.02, valid_max=0.98):
    """
    Build an HDR image from a multi-exposure Bayer stack.

    Args:
        frames:        Bayer image stack (H, W, N), dtype uint16.
        exposures_us:  Exposure times in microseconds, shape (N,).
        method:        'linear_raw' (default) or 'debevec'.
        bayer_pattern: OpenCV Bayer conversion code.
        gamma:         Gamma for Reinhard tone mapping.
        max_val:       Sensor max value (65535 for BayerRG16).
        valid_min/max: Pixel validity thresholds (linear_raw only).

    Returns:
        hdr: HDR radiance map (H, W, 3), float32.
        ldr: Tone-mapped preview (H, W, 3), uint8.
    """
    if method == "linear_raw":
        return make_hdr_linear_raw(frames=frames, exposures_us=exposures_us,
                                   bayer_pattern=bayer_pattern, gamma=gamma,
                                   max_val=max_val, valid_min=valid_min,
                                   valid_max=valid_max)
    elif method == "debevec":
        return make_hdr_debevec(frames=frames, exposures_us=exposures_us,
                                bayer_pattern=bayer_pattern, gamma=gamma,
                                max_val=max_val)
    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'linear_raw' or 'debevec'.")
