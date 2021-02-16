__author__ = "piveloper"
__copyright__ = "06.02.2021, piveloper"
__version__ = "1.0"
__email__ = "piveloper@gmail.com"
__doc__ = """This module contains the convolution operation"""

from typing import List, Union

import numpy as np
from pyopencl.array import Array, to_device, zeros
from pyopencl_extension.components.copy_array_region import CopyArrayRegion
from pyopencl_extension import Thread, ClHelpers, Kernel, Program, Global, is_complex_type


class Convolution1D:
    @property
    def impulse_response(self):
        return self.impulse_response_buffer.get()

    def __init__(self, in_buffer: Array, impulse_response: Union[List[complex], np.ndarray, Array]):
        self.thread = Thread.from_buffer(in_buffer)
        self.queue = in_buffer.queue
        self.in_buffer = in_buffer
        if isinstance(impulse_response, Array):
            self.impulse_response_buffer = impulse_response
        else:
            self.impulse_response_buffer = to_device(self.queue, np.array(impulse_response, in_buffer.dtype))
        self.length_impulse_response = self.impulse_response_buffer.shape[0]

        self.in_buffer = in_buffer

        shape = self.in_buffer.shape
        shape_trailing = shape[:-1] + (self.length_impulse_response - 1,)
        shape_out = shape[:-1] + (shape[-1] + shape_trailing[-1],)
        self.out_buffer = zeros(self.queue, shape_out, dtype=in_buffer.dtype)

        axis_conv = self.in_buffer.ndim - 1
        workgroup_size = 2 * self.thread.device.global_mem_cacheline_size
        global_size = (int(self.out_buffer.shape[axis_conv] / workgroup_size + 1.0) * workgroup_size,)
        local_size = (workgroup_size,)

        self.knl = Kernel(name='convolve',
                          args={'x': Global(self.in_buffer),
                                  'h': Global(self.impulse_response_buffer, '__constant'),
                                  'y': Global(self.out_buffer)},
                          # https://en.wikipedia.org/wiki/Convolution
                          # assumes filter dimension to be contiguous in memory
                          # performs full convolution
                          # thus input buffer does have leading and trailing zeros according to filter length
                          body="""
                        int n = get_global_id(AXIS_CONV);
                        if(n<LENGTH_OUT_AXIS_CONV){
                            data_t _sum=${init_sum};
                            for(int m=0;m<LENGTH_IMP_RES;m++){
                                int n_minus_m = n-m;
                                if(0<=n_minus_m && n_minus_m < LENGTH_IN_AXIS_CONV){
                                    _sum = ADD(_sum, MUL(h[m],x[n_minus_m]));
                                }
                            }
                            y[n] = _sum;
                        }
                            """,
                          type_defs={'data_t': (data_t := self.in_buffer.dtype)},
                          replacements={'command_addr': ClHelpers.command_compute_address(self.in_buffer.ndim),
                                          'init_sum': '0' if is_complex_type(data_t) else 'NEW(0,0)'},
                          defines={'AXIS_CONV': axis_conv,
                                     'LENGTH_IN_AXIS_CONV': self.in_buffer.shape[axis_conv],
                                     'LENGTH_OUT_AXIS_CONV': self.out_buffer.shape[axis_conv],
                                     'LENGTH_IMP_RES': self.length_impulse_response},
                          global_size=global_size,
                          local_size=local_size
                          ).compile(thread=self.thread, b_python=False)

    def __call__(self, b_python: bool = False, **kwargs) -> Array:
        self.knl()
        return self.out_buffer
