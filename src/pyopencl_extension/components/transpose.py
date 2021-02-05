import os
from typing import Tuple, Union

from mako.template import Template
from pyopencl import cltypes

import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.array import Array

from pyopencl_extension.framework import ClHelpers, ClProgram, ClKernel, KnlArgBuffer, KnlGridType, ScalarArgTypes, \
    ClInit

__author__ = "piveloper"
__copyright__ = "26.03.2020, piveloper"
__version__ = "1.0"
__email__ = "piveloper@gmail.com"
__doc__ = """This module contains the Transpose class which transposes an array."""

"""
/*
in_buffer.shape =(4,8,10)
axis_order = (1,0,2)
out_buffer.shape = (8,4,10)

s0 = get_global_size(1) = 4
s1 = get_global_size(1) = 8
s2 = get_global_size(2) = 10

Computation grid is spanned by in_buffer.shape:
g0,g1,g2

According to axis order, elements of in_buffer are copied to different positions in memory of out_buffer:
for axis_order = (1,0,2):
out_buffer[g1*s0*s2+g0*s2+g2]=in_buffer[g0*s1*s2+g1*s2+g2]
i_in = g0*s1*s2+g1*s2+g2
i_out = g1*s0*s2+g0*s2+g2

1 D scenario for axis_order = (1,0):
s0 = get_global_size(1) = 4
s1 = get_global_size(1) = 1

i_in = g0*s1+g1
i_out = g1*s0+g0

*/
"""


class Transpose:
    def _command_for_input_address_computation(self):
        """
        //specific implementation axes_order=(1,0)
        int i_in = g0*get_global_size(1)+g1;
        int i_out = g1*get_global_size(0)+g0;

        //specific implementation axes_order=(1,2,0)
        int i_in = g0*get_global_size(1)*get_global_size(2)+g1*get_global_size(2)+g0;
        int i_out = g1*get_global_size(2)*get_global_size(0)+g2*get_global_size(0)+g0;
        """
        n_dim = self.in_buffer.shape.__len__()
        command = '0'
        for i in range(n_dim):
            offset = '1'
            for j in range(i + 1, n_dim):
                offset += '*get_global_size({})'.format(j)
            command += '+get_global_id({})*{}'.format(i, offset)
        return command  # 'get_global_id(0)*get_global_size(1)+get_global_id(1)'

    def _command_for_output_address_computation(self):
        """
        //specific implementation axes_order=(1,0)
        int i_in = g0*get_global_size(1)+g1;
        int i_out = g1*get_global_size(0)+g0;

        //specific implementation axes_order=(1,2,0)
        int i_in = g0*get_global_size(1)*get_global_size(2)+g1*get_global_size(2)+g0;
        int i_out = g1*get_global_size(2)*get_global_size(0)+g2*get_global_size(0)+g0;
        """
        n_dim = self.in_buffer.shape.__len__()
        command = '0'
        for i in range(n_dim):
            offset = '1'
            # axes_order[j]: What has been the axis of input buffer related to axis j of output buffer?
            for j in range(i + 1, n_dim):
                offset += '*get_global_size({})'.format(self.axes_order[j])
            command += '+get_global_id({})*{}'.format(self.axes_order[i], offset)
        return command  # 'get_global_id(1)*get_global_size(0)+get_global_id(0)'

    def __init__(self, in_buffer: cl_array.Array, axes_order: Tuple[int, ...]):
        self.axes_order = axes_order
        shape_out = tuple([list(in_buffer.shape)[i] for i in axes_order])
        self.out_buffer = cl.array.empty(in_buffer.queue, shape_out, dtype=in_buffer.dtype)
        self.in_buffer = in_buffer
        self.knl = ClKernel(name='transpose',
                       args={'in_buffer': KnlArgBuffer(self.in_buffer, 'const'),
                             'out_buffer': KnlArgBuffer(self.out_buffer, '', True)},
                       body=["""
                                int i_in = ${i_in};
                                int i_out = ${i_out};
                                out_buffer[i_out] = in_buffer[i_in];                       
                               """],
                       replacements={'i_in': self._command_for_input_address_computation(),
                                     'i_out': self._command_for_output_address_computation()},
                       global_size=self.in_buffer.shape).compile(cl_init=ClInit.from_buffer(in_buffer))

    def __call__(self, *args, **kwargs):
        self.knl()
        return self.out_buffer