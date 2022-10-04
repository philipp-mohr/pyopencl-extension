from pyopencl_extension import Helpers, Program, Kernel, Global, Scalar, Array, empty
from pyopencl_extension import Types

__author__ = "P.Mohr"
__copyright__ = "26.03.2020, P.Mohr"
__version__ = "1.0"
__email__ = "philipp.mohr@tuhh.de"
__doc__ = """This script includes helpful functions to extended PyOpenCl functionality."""


class SumAlongAxis:
    """
    Responsibility:
    Calculates the sum over all elements of an OpenCl array along an axis.
    """

    def _command_compute_address_in(self):
        """
        e.g. in_buffer = zeros(shape=(100,10)), axis=0
            out_buffer = zeros(shape=(10,))
        //from for loop in kernel: i = glob_id_axis;
        sum + = in_buffer[i* get_global_size(0) + get_global_id(0)];

        e.g. in_buffer = zeros(shape=(100,10)), axis=1
            out_buffer = zeros(shape=(100,))
        //from for loop in kernel: i = glob_id_axis;
        sum + = in_buffer[get_global_id(0)* size_axis + i];

        e.g. in_buffer = zeros(shape=(1000, 100,10)), axis=1
            out_buffer = zeros(shape=(1000, 10))
        //from for loop in kernel: i = glob_id_axis;
        sum + = in_buffer[get_global_id(0)* size_axis * get_global_size(1) + i * get_global_size(1) + get_global_id(1)];
        """
        command = '0'
        subtract_axis = 0
        for axis_in_buffer in range(self.in_buffer.ndim):
            offset = '1'
            # get_global_id(i) * offset
            # classic way: offset = get_global_size(i+1)*...*get_global_size(ndim-1)
            if axis_in_buffer == self.axis:
                subtract_axis = 1  # subtract to account for smaller global_size than in_buffer size.
            for axis_out_buffer in range(axis_in_buffer - subtract_axis + 1, self.out_buffer.ndim):
                offset += '*get_global_size({})'.format(axis_out_buffer)
            # Now insert size axis to account for larger ndim of in_buffer compared to out_buffer
            if self.axis > axis_in_buffer:
                offset += '*{size_axis}'.format(size_axis=self.in_buffer.shape[self.axis])

            if axis_in_buffer == self.axis:
                command += '+i*{}'.format(offset)
            else:
                axis_out_buffer = axis_in_buffer - subtract_axis
                command += '+get_global_id({})*{}'.format(axis_out_buffer, offset)
        return command

    def _get_cl_program(self) -> Program:
        knl = Kernel(name='sum_along_axis',
                     args={'in_buffer': Global(self.in_buffer),
                           'axis': Scalar(Types.int(self.axis)),
                           'out_buffer': Global(self.out_buffer)},
                     body=["""
                 buff_t sum = (buff_t) 0;
                 for(int i=0; i<${size_input_axis}; i++){// i == glob_id_axis
                    sum+=in_buffer[${addr_in}];
                 }                 
                 out_buffer[${addr}] = sum;
                 """],
                     replacements={'size_input_axis': self.in_buffer.shape[self.axis],
                                   'addr': Helpers.command_compute_address(self.out_buffer.ndim),
                                   'addr_in': self._command_compute_address_in()},
                     global_size=self.out_buffer.shape)
        type_defs = {'buff_t': self.in_buffer.dtype}
        return Program(type_defs=type_defs, kernels=[knl])

    def __init__(self, in_buffer: Array, axis: int = 0, out_buffer: Array = None):
        self.in_buffer = in_buffer
        self.axis = axis
        if self.in_buffer.ndim == 1:
            shape_out = (1,)
        else:
            shape_out = tuple([size for i_axis, size in enumerate(self.in_buffer.shape) if i_axis != self.axis])
        if out_buffer is None:
            self.out_buffer = empty(shape_out, self.in_buffer.dtype)
        else:
            if out_buffer.shape != shape_out:
                raise ValueError('out buffer shape does not match required shape')
            self.out_buffer = out_buffer
        self.program = self._get_cl_program().compile()

    def __call__(self, *args, **kwargs):
        self.program.sum_along_axis()
        return self.out_buffer
