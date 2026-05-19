import time
from datetime import datetime
import numpy as np
import cv2
import os

from camera.wrapper_pyspin import Blackfly


cam = Blackfly(pixel_format="BayerRG16")
cam.set_auto_exposure()

img, _, _, exposure, gain = cam.get_next_image(return_metadata=True)
print(f"shape: {img.shape}, dtype: {img.dtype}")
print(f"    fps: {cam.get_fps():.2f}, exposure: {exposure:.1f} us, gain: {gain:.2f} dB")


# ----------------------------
# HDR helpers
# ----------------------------
def get_exposure_interval(ideal_exposure, num_steps=8, ratio=16):
    min_exp = max(cam.min_exp_val, ideal_exposure / ratio)
    max_exp = min(cam.max_exp_val, ideal_exposure * ratio)
    exposures = [float(e) for e in np.geomspace(min_exp, max_exp, num_steps)]
    print(f"Exposure interval (us): {[f'{e:.1f}' for e in exposures]}")
    return exposures


def capture_hdr(output_folder_path=None):
    cam.set_auto_exposure(False)
    base_exposure = cam.get_exposure()
    exposures = get_exposure_interval(base_exposure)
    capture_time = time.time()
    real_exposures = []

    for exp in exposures:
        cam.set_exposure(exp, check=False)
        time.sleep(0.05)
        frame, _, _, cur_exp, _ = cam.get_next_image(return_metadata=True)
        real_exposures.append(cur_exp)
        print(f"  exposure: {cur_exp:.1f} us, shape: {frame.shape}")

        if output_folder_path is not None:
            os.makedirs(output_folder_path, exist_ok=True)
            timestamp_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(capture_time))
            frame_8bit = (frame / 256).astype(np.uint8)
            frame_color = cv2.cvtColor(frame_8bit, cv2.COLOR_BayerRG2RGB)
            cv2.imwrite(os.path.join(output_folder_path, f'vis_exp{cur_exp/1000:.2f}ms_{timestamp_str}.png'), frame_color)
            np.save(os.path.join(output_folder_path, f'vis_exp{cur_exp/1000:.2f}ms_raw.npy'), frame)

    cam.set_exposure(base_exposure, check=False)
    cam.set_auto_exposure()
    return capture_time, real_exposures


# ----------------------------
# Single + HDR capture
# ----------------------------
def capture_images(output_folder=None):
    if output_folder is None:
        output_folder = os.path.join('results', time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(output_folder, exist_ok=True)

    print("Capturing image")
    frame, vis_tstamp, _, exposure, _ = cam.get_next_image(return_metadata=True)
    timestamp_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(vis_tstamp))
    np.save(os.path.join(output_folder, f'vis_{timestamp_str}_exp{exposure/1000:.2f}ms_raw.npy'), frame)
    frame_8bit = (frame / 256).astype(np.uint8)
    frame_color = cv2.cvtColor(frame_8bit, cv2.COLOR_BayerRG2RGB)
    cv2.imwrite(os.path.join(output_folder, f'vis_{timestamp_str}_exp{exposure/1000:.2f}ms.png'), frame_color)

    print("Capturing HDR images")
    capture_hdr(output_folder)

    print(f"Saved to {output_folder}")
    return output_folder


# ----------------------------
# Live view
# ----------------------------
cv2.namedWindow('Blackfly', cv2.WINDOW_NORMAL)

print("Controls:")
print("  'i' - capture images (single + HDR)")
print("  't' - toggle text display")
print("  'q' - quit")
print("  'e' / 'd' - exposure +1000 / -1000 us")
print("  'g' / 'f' - gain +1 / -1 dB")
print("  'a' - toggle auto exposure")

time_start = None
time_last = None
effective_FPS = None
display_text = True
auto_exposure = True
output_folder = os.path.join('results', time.strftime('%Y%m%d_%H%M%S'))

while True:
    time_last = time_start
    if time_start is not None and time_last is not None:
        time_start = time.time()
        effective_FPS = 1 / (time_start - time_last)
    if time_start is None:
        time_start = time.time()

    frame, _, _, exposure, gain = cam.get_next_image(return_metadata=True)
    if frame is None:
        continue

    display = (frame / 256).astype(np.uint8)
    if len(display.shape) == 2:
        display = cv2.cvtColor(display, cv2.COLOR_BayerRG2RGB)

    display_img = display.copy()
    if display_text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        if effective_FPS is not None:
            cv2.putText(display_img, f'FPS: {effective_FPS:.2f}', (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(display_img, f'Exposure: {exposure/1000:.2f}ms  Gain: {gain:.1f}dB', (10, 60), font, 0.7, (255, 255, 255), 2)
        cv2.putText(display_img, f'AutoExp: {"ON" if auto_exposure else "OFF"}', (10, 90), font, 0.7, (255, 255, 255), 2)

    cv2.imshow('Blackfly', display_img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        cam.set_auto_exposure(False)
        auto_exposure = False
        cam.set_exposure(cam.get_exposure() + 1000, check=False)
    elif key == ord('d'):
        cam.set_auto_exposure(False)
        auto_exposure = False
        cam.set_exposure(cam.get_exposure() - 1000, check=False)
    elif key == ord('g'):
        cam.set_gain(cam.get_gain() + 1)
    elif key == ord('f'):
        cam.set_gain(cam.get_gain() - 1)
    elif key == ord('a'):
        auto_exposure = not auto_exposure
        cam.set_auto_exposure(auto_exposure)
        print(f"Auto exposure: {'ON' if auto_exposure else 'OFF'}")
    elif key == ord('t'):
        display_text = not display_text
    elif key == ord('i'):
        try:
            output_folder = os.path.join('results', datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            output_folder = capture_images(output_folder)

            font = cv2.FONT_HERSHEY_SIMPLEX
            completion_text = "Capture Complete!"
            text_size = cv2.getTextSize(completion_text, font, 1, 2)[0]
            text_x = (display_img.shape[1] - text_size[0]) // 2
            text_y = (display_img.shape[0] + text_size[1]) // 2
            overlay = display_img.copy()
            cv2.rectangle(overlay, (0, 0), (display_img.shape[1], display_img.shape[0]), (0, 0, 0), -1)
            display_img2 = cv2.addWeighted(overlay, 0.5, display_img, 0.5, 0)
            cv2.putText(display_img2, completion_text, (text_x, text_y), font, 1, (255, 255, 255), 2)
            cv2.imshow('Blackfly', display_img2)
            cv2.waitKey(1000)
        except Exception as e:
            print(f"Error: {e}")

cam.stop()
cv2.destroyAllWindows()
