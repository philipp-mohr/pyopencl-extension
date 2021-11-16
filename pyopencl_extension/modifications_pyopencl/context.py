import numpy as np
import pyopencl as cl


def get_devices():
    """
    On a computer often multiple chips exist to execute OpenCl code, like Intel, AMD or Nvidia GPUs or FPGAs.
    This function return a list of all available devices.
    """
    platforms = cl.get_platforms()
    devices = [d for p in platforms for d in p.get_devices(device_type=cl.device_type.GPU)]
    # devices = [platform[device[0]].get_devices(device_type=cl.device_type.GPU)[device[1]]][0]
    return devices


class Context(cl.Context):
    def __init__(self, devices, properties=None, dev_type=None):
        if len(devices) > 1:
            raise NotImplementedError('In current implementation only a single device per context is supported. '
                                      'Please request feature if required.')
        super().__init__(devices, properties, dev_type)
        self.time_compilation_ns = 0

    @property
    def device_id(self):
        matches = np.where(np.array([d.int_ptr for d in get_devices()]) == self.int_ptr)
        if not len(matches) == 1:
            raise ValueError('int_ptr is expected to be unique identifier for device. '
                             'Change current implementation to fix issue')
        return matches[0][0]

    def add_time_compilation(self, t_ns):
        self.time_compilation_ns += t_ns
