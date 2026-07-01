# camera_Blackfly

Capture and HDR tools for a FLIR (Teledyne) Blackfly camera via the Spinnaker
(PySpin) SDK.

## Structure

```
camera_Blackfly/
├── camera/
│   └── wrapper_pyspin.py   # Blackfly camera driver (PySpin wrapper)
├── utils/
│   └── make_hdr.py         # HDR image construction from a multi-exposure stack
├── capture_blackfly.py     # Interactive live view + single/HDR capture
└── capture_blackfly.ipynb  # Notebook version
```

## Requirements

```
numpy
opencv-python
matplotlib
```

PySpin is **not** on PyPI. Install it from the FLIR/Teledyne Spinnaker SDK:

1. Install the Spinnaker SDK: https://www.flir.com/products/spinnaker-sdk/
2. Install the matching Python wheel:
   `pip install spinnaker_python-<version>-cp<pyver>-<platform>.whl`

The wrapper imports it as `import PySpin`.

## Usage

### Interactive capture

```
python capture_blackfly.py
```

Keys: `i` capture (single + HDR), `e`/`d` exposure +/- 1000 us,
`g`/`f` gain +/- 1 dB, `a` toggle auto-exposure, `t` toggle text overlay, `q` quit.

### Camera driver

```python
from camera.wrapper_pyspin import Blackfly

cam = Blackfly(pixel_format="BayerRG16")   # Mono16 / Mono8 / BayerRG8 / BayerRG16

# Auto exposure
cam.set_auto_exposure(True)

# Manual settings (auto-exposure is turned off automatically for manual exposure)
cam.set_exposure(10000)   # microseconds (us)
cam.set_gain(0)           # dB
cam.set_fps(30)           # Hz

# Set frame rate and exposure together, decoupling their mutual constraint:
cam.set_framerate_and_exposure(fps=10, exposure_time=50000)   # 10 fps, 50 ms

# Grab a frame (+ metadata)
img, timestamp, frame_id, exposure, gain = cam.get_next_image(return_metadata=True)

cam.stop()
```

### HDR

```python
import numpy as np
from utils.make_hdr import make_hdr

data = np.load("stack.npz")               # frames (H, W, N) + exposures (us)
hdr, ldr = make_hdr(data["frames"], data["exposures"], method="linear_raw")
```

## Notes / gotchas

### Frame rate vs. exposure vs. bit depth vs. resolution

These four are coupled through the sensor readout and the USB link bandwidth:

- **Exposure limits FPS.** A frame cannot be shorter than its exposure, so the
  max frame rate is roughly `1 / exposure` (e.g. 50 ms exposure -> <= 20 fps).
- **Bit depth and resolution set the bytes per frame.** A 12 MP sensor is
  `3000 x 4000` px = 12 M pixels; that is 24 MB/frame at 16-bit (BayerRG16) or
  12 MB/frame at 8-bit (BayerRG8).
- **USB3 bandwidth (~400 MB/s effective) then caps the frame rate.** At 12 MP:
  16-bit -> ~15 fps max, 8-bit -> ~30 fps max, regardless of exposure.
  To go faster, reduce bytes/frame: use 8-bit, and/or crop (ROI) or bin.

`set_fps()` re-queries the live `AcquisitionFrameRate` max on every call, so it
correctly reflects the current exposure and pixel format.

### FPS and exposure are mutually constraining

Setting `set_fps()` then `set_exposure()` (or vice-versa) can clamp one because
the other was set first. Use `set_framerate_and_exposure(fps, exposure_time)` to
set both independently — it minimizes the exposure, sets the frame rate, then
applies the requested exposure (bounded by `1/fps`).

### Manual exposure requires auto-exposure OFF

`ExposureTime` is read-only while auto-exposure is running. `set_exposure()` now
disables auto-exposure automatically before writing, so manual exposure never
raises a `Node is not writable` access exception.

### Images look yellow/green (no white balance)

The wrapper **disables auto white balance** and sets the red/blue balance ratios
to 1.0 on purpose, to keep the raw sensor response for HDR/analysis. As a result
debayered previews have a yellow/green cast. Apply white balance in
post-processing for display (e.g. a gray-world correction), or enable/set the
camera's `BalanceWhiteAuto` / `BalanceRatio` if you want white-balanced captures.
```

## License

See repository.
