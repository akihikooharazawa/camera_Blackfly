import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import time
import cv2
import PySpin


class Blackfly(object):
    def __init__(self, cam_id=0, pixel_format="Mono16"):
        self.system = PySpin.System.GetInstance()
        cam_list = self.system.GetCameras()
        num_cameras = cam_list.GetSize()
        assert num_cameras > 0 and num_cameras > cam_id, \
            f'Found only {num_cameras} camera(s). Requested cam_id={cam_id}'
        self.cam = cam_list[cam_id]
        cam_list.Clear()

        self.cam.Init()
        self.nodemap = self.cam.GetNodeMap()
        self.set_buffer_handling_mode()
        self.load_userset("UserSet0")
        self.timestamp_offset = self.get_timestamp_offset()
        self.enable_chunk_data()

        # Set Pixel Format
        self.image_processor = None
        if pixel_format == "Mono16":
            self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono16)
            self.bpp = 16
        elif pixel_format == "Mono8":
            self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
            self.bpp = 8
        elif pixel_format == "BayerRG8":
            self.cam.PixelFormat.SetValue(PySpin.PixelFormat_BayerRG8)
            self.bpp = 8
        elif pixel_format == "BayerRG16":
            self.cam.PixelFormat.SetValue(PySpin.PixelFormat_BayerRG16)
            self.bpp = 16
        else:
            raise NotImplementedError(f"Unsupported pixel format: {pixel_format}")

        # Disable auto white balance
        self.cam.BalanceWhiteAuto.SetValue(PySpin.BalanceWhiteAuto_Off)
        self.cam.BalanceRatioSelector.SetValue(PySpin.BalanceRatioSelector_Red)
        self.cam.BalanceRatio.SetValue(1.0)
        self.cam.BalanceRatioSelector.SetValue(PySpin.BalanceRatioSelector_Blue)
        self.cam.BalanceRatio.SetValue(1.0)

        node_exposure = PySpin.CFloatPtr(self.nodemap.GetNode('ExposureTime'))
        self.min_exp_val = node_exposure.GetMin()
        self.max_exp_val = node_exposure.GetMax()

        node_gain = PySpin.CFloatPtr(self.nodemap.GetNode('Gain'))
        self.min_gain_val = node_gain.GetMin()
        self.max_gain_val = node_gain.GetMax()

        node_fps = PySpin.CFloatPtr(self.nodemap.GetNode('AcquisitionFrameRate'))
        self.min_fps_val = node_fps.GetMin()
        self.max_fps_val = node_fps.GetMax()

        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        self.cam.BeginAcquisition()

        self.exposure = self.get_exposure()
        self.gain = self.get_gain()
        self.fps = self.get_fps()

    def stop(self):
        self.cam.EndAcquisition()
        self.cam.DeInit()
        del self.cam
        self.system.ReleaseInstance()

    def set_buffer_handling_mode(self, mode='NewestOnly'):
        sNodemap = self.cam.GetTLStreamNodeMap()
        node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsReadable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
            print('Unable to set stream buffer handling mode.')
            return False
        node_newestonly = node_bufferhandling_mode.GetEntryByName(mode)
        if not PySpin.IsReadable(node_newestonly):
            print('Unable to set stream buffer handling mode.')
            return False
        node_bufferhandling_mode.SetIntValue(node_newestonly.GetValue())

    def load_userset(self, userset_name):
        node_user_set_selector = PySpin.CEnumerationPtr(self.nodemap.GetNode('UserSetSelector'))
        if not PySpin.IsReadable(node_user_set_selector) or not PySpin.IsWritable(node_user_set_selector):
            print('Unable to load UserSet.')
            return False
        node_user_set_value = node_user_set_selector.GetEntryByName(userset_name)
        if not PySpin.IsReadable(node_user_set_value):
            print('Unable to load UserSet.')
            return False
        node_user_set_selector.SetIntValue(node_user_set_value.GetValue())
        node_user_set_load = PySpin.CCommandPtr(self.nodemap.GetNode('UserSetLoad'))
        if not PySpin.IsAvailable(node_user_set_load) or not PySpin.IsWritable(node_user_set_load):
            print('Unable to load UserSet.')
            return False
        node_user_set_load.Execute()

    def get_timestamp_offset(self):
        node_timestamp_latch = PySpin.CCommandPtr(self.nodemap.GetNode('TimestampLatch'))
        if not PySpin.IsAvailable(node_timestamp_latch) or not PySpin.IsWritable(node_timestamp_latch):
            print('Unable to execute TimestampLatch.')
            return 0.0
        system_time = time.time()
        node_timestamp_latch.Execute()
        node_timestamp_latch_value = PySpin.CIntegerPtr(self.nodemap.GetNode('TimestampLatchValue'))
        if not PySpin.IsAvailable(node_timestamp_latch_value) or not PySpin.IsReadable(node_timestamp_latch_value):
            print('Unable to read TimestampLatchValue.')
            return 0.0
        latched_time = node_timestamp_latch_value.GetValue()
        print(f'System Time: {system_time:.3f} s, Timestamp Latch: {latched_time} ns')
        return system_time - latched_time / 1e9

    def enable_chunk_data(self):
        chunk_mode_active = PySpin.CBooleanPtr(self.nodemap.GetNode('ChunkModeActive'))
        if not PySpin.IsWritable(chunk_mode_active):
            print('Unable to activate chunk mode.')
            return False
        chunk_mode_active.SetValue(True)
        chunk_selector = PySpin.CEnumerationPtr(self.nodemap.GetNode('ChunkSelector'))
        for chunk_name in ['Timestamp', 'FrameID', 'ExposureTime', 'Gain']:
            chunk_selector.SetIntValue(chunk_selector.GetEntryByName(chunk_name).GetValue())
            chunk_enable = PySpin.CBooleanPtr(self.nodemap.GetNode('ChunkEnable'))
            if PySpin.IsWritable(chunk_enable):
                chunk_enable.SetValue(True)
        return True

    # --------------- Exposure ---------------
    def set_auto_exposure(self, auto_exp=True):
        if auto_exp:
            self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)
        else:
            self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        return self.get_auto_exposure()

    def get_auto_exposure(self):
        return self.cam.ExposureAuto.GetValue()

    def set_exposure(self, exposure_time, check=True):
        # Manual exposure requires auto-exposure OFF: ExposureTime is read-only
        # while auto-exposure is running, so ensure it is disabled first.
        if self.get_auto_exposure() != PySpin.ExposureAuto_Off:
            self.set_auto_exposure(False)
        exposure_time = max(self.min_exp_val, min(self.max_exp_val, exposure_time))
        self.cam.ExposureTime.SetValue(exposure_time)
        self.exposure = self.get_exposure()
        print(f'Exposure set to {self.exposure:.1f} us')
        if check:
            while True:
                _, _, _, curr_exp, _ = self.get_next_image(return_metadata=True)
                if abs(curr_exp - self.exposure) < 5:
                    break
        return self.exposure

    def get_exposure(self):
        return self.cam.ExposureTime.GetValue()

    # --------------- Gain ---------------
    def set_gain(self, gain):
        gain = max(self.min_gain_val, min(self.max_gain_val, gain))
        self.cam.Gain.SetValue(gain)
        self.gain = self.get_gain()
        print(f'Gain set to {self.gain:.2f} dB')
        while True:
            _, _, _, _, curr_gain = self.get_next_image(return_metadata=True)
            if abs(curr_gain - self.gain) < 0.1:
                break
        return self.gain

    def get_gain(self):
        return self.cam.Gain.GetValue()

    # --------------- FPS ---------------
    def set_fps(self, fps):
        # The achievable frame rate depends on the current exposure and pixel
        # format, so re-query the live node min/max instead of the values cached
        # at __init__ (otherwise a later, shorter exposure cannot raise the fps).
        self.min_fps_val = self.cam.AcquisitionFrameRate.GetMin()
        self.max_fps_val = self.cam.AcquisitionFrameRate.GetMax()
        fps = max(self.min_fps_val, min(self.max_fps_val, fps))
        curr_exp = self.get_exposure()
        self.cam.AcquisitionFrameRate.SetValue(fps)
        self.fps = self.get_fps()
        print(f'FPS set to {self.fps:.2f}')
        self.min_exp_val = self.cam.ExposureTime.GetMin()
        self.max_exp_val = self.cam.ExposureTime.GetMax()
        self.set_exposure(curr_exp)
        return self.fps

    def get_fps(self):
        return self.cam.AcquisitionFrameRate.GetValue()

    def set_framerate_and_exposure(self, fps, exposure_time):
        # FPS and exposure constrain each other (exposure <= 1/fps, and the max fps
        # is limited by the exposure). To set both independently, decouple them:
        #   1) minimize exposure -> the requested fps is always reachable
        #   2) set fps           -> fixes the frame period (1/fps)
        #   3) set exposure      -> apply the real value (bounded by the period)
        self.set_auto_exposure(False)
        self.set_exposure(self.min_exp_val, check=False)
        self.set_fps(fps)
        return self.set_exposure(exposure_time)

    # --------------- Acquisition ---------------
    def get_next_image(self, return_metadata=False):
        try:
            image_result = self.cam.GetNextImage()
            software_tstamp = time.time()

            if image_result.IsIncomplete():
                print(f'Incomplete image: {PySpin.Image_GetImageStatusDescription(image_result.GetImageStatus())}')
                image_result.Release()
                return (None, None, None, None, None) if return_metadata else (None, None)

            metadata = image_result.GetChunkData()
            timestamp = metadata.GetTimestamp() / 1e9 + self.timestamp_offset
            frame_number = metadata.GetFrameID()
            exposure = metadata.GetExposureTime()
            gain = metadata.GetGain()

            if self.image_processor is not None:
                fmt = PySpin.PixelFormat_BGR8 if self.bpp == 8 else PySpin.PixelFormat_BGR16
                image_data = self.image_processor.Convert(image_result, fmt).GetNDArray()
            else:
                image_data = image_result.GetNDArray()

            image_result.Release()

        except PySpin.SpinnakerException as ex:
            print(f'Error: {ex}')
            return (None, None, None, None, None) if return_metadata else (None, None)

        if return_metadata:
            return image_data, timestamp, frame_number, exposure, gain
        else:
            return image_data, software_tstamp


def main():
    cam = Blackfly(pixel_format="BayerRG16")
    cam.set_auto_exposure()

    try:
        while True:
            image_data, _, _, exposure, gain = cam.get_next_image(return_metadata=True)
            if image_data is None:
                continue
            display = (image_data / 256).astype('uint8')
            cv2.imshow('Blackfly', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                cam.set_exposure(cam.get_exposure() + 1000)
            elif key == ord('d'):
                cam.set_exposure(cam.get_exposure() - 1000)
            elif key == ord('g'):
                cam.set_gain(cam.get_gain() + 1)
            elif key == ord('f'):
                cam.set_gain(cam.get_gain() - 1)
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
