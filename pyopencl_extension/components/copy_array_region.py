from copy import copy
from typing import Tuple, Union

import numpy as np
from pyopencl.array import Array, empty, zeros, to_device
from pytest import mark

from pyopencl_extension import Thread, Helpers, Kernel, Program, Global, Scalar, Types

__author__ = "piveloper"
__copyright__ = "26.03.2020, piveloper"
__version__ = "1.0"
__email__ = "piveloper@gmail.com"
__doc__ = """This script implements the functionality to copy certain region_in of cl array on device."""

from pyopencl_extension.types.utilities_np_cl import c_name_from_dtype

TypeSliceFormatCopyArrayRegion = Tuple[Tuple[int, Union[int, None], Union[int, None], int], ...]


# supports sclicing with [start,stop] and [start:stop:step_width]


class _Slice:
    def __getitem__(self, val) -> TypeSliceFormatCopyArrayRegion:
        def convert(axis, _val):
            return (axis,
                    0 if _val.start is None else _val.start,
                    _val.stop,
                    1 if _val.step is None else _val.step)

        if not isinstance(val, tuple):
            return convert(0, val),
        else:
            return tuple([convert(axis, _val) for axis, _val in enumerate(val)])


Slice = _Slice()


def test_slice():
    res = Slice[2:]
    assert res == ((0, 2, None, 1),)
    res = Slice[2::3]
    assert res == ((0, 2, None, 3),)
    res = Slice[2:10:3]
    assert res == ((0, 2, 10, 3),)
    res = Slice[2:, 3:-1:2, :]
    assert res == ((0, 2, None, 1),
                   (1, 3, -1, 2),
                   (2, 0, None, 1))
    res = Slice[0:0]
    assert res == ((0, 0, 0, 1),)


class CopyArrayRegion:
    def _command_for_addr_out_computation(self):
        command = '0'
        for i in range(self.in_buffer.ndim):
            offset = '1'
            for j in range(i + 1, self.in_buffer.ndim):
                offset += '*{}'.format(self.out_buffer.shape[j])
            command += '+(get_global_id({})+{})*{}'.format(i, self.region_out[i][0], offset)
        return command

    def _command_for_addr_in_computation(self):
        command = '0'
        for i in range(self.in_buffer.ndim):
            offset = '1'
            for j in range(i + 1, self.in_buffer.ndim):
                offset += '*{}'.format(self.in_buffer.shape[j])
            command += '+(get_global_id({})+{})*{}'.format(i, self.region_in[i][0], offset)
        return command

    @staticmethod
    def _deal_with_incomplete_regions(region, array: Array):
        # deal with incomplete regions. Fill remaining ranges for axes from input buffer size
        reg = [(0, ax_dim, 1) for ax_dim in array.shape]
        for ax in region:
            reg[ax[0]] = ax[1:]
        return reg

    @staticmethod
    def _deal_with_none_in_stop(region, array: Array):
        # deal with None in stop e.g. Slice[:] -> ((0, 0, None, 1),)
        return [[ax[0], ax[1] if ax[1] is not None else array.shape[i], ax[2]]
                for i, ax in enumerate(region)]

    @staticmethod
    def _deal_with_negative_region(region, array: Array):
        # deal with negative regions
        return [[idx if idx >= 0 else array.shape[i] + idx for idx in ax] for i, ax in
                enumerate(region)]

    def __init__(self, in_buffer: Array,
                 region_in: TypeSliceFormatCopyArrayRegion = None,
                 out_buffer: Array = None,
                 region_out: TypeSliceFormatCopyArrayRegion = None):
        """

        :param in_buffer:
        :param region_in: e.g. region_in=((0,0,1,2),(1,1,3,2)) selects the region_in of array a, like numpy would do with
        a[0:1:2,1:3:2] where 2 is the step width. The first element of tuple selects the axis.
        :param out_buffer: target buffer where data from in_buffer is being copied. (optional)
        :param region_out: specifies the region of out buffer memory where in_buffer data is copied (optional)
        """
        _region_in_original = copy(region_in)  # for debug purposes
        _region_out_original = copy(region_out)  # for debug purposes

        queue = in_buffer.queue
        if region_in is not None:
            region_in = self._deal_with_incomplete_regions(region_in, in_buffer)

        if out_buffer is not None and region_out is not None:
            region_out = self._deal_with_incomplete_regions(region_out, out_buffer)

        if region_in is not None:
            region_in = self._deal_with_none_in_stop(region_in, in_buffer)
        if region_out is not None and out_buffer is not None:
            region_out = self._deal_with_none_in_stop(region_out, out_buffer)

        if region_in is not None:
            region_in = self._deal_with_negative_region(region_in, in_buffer)
        if region_out is not None and out_buffer is not None:
            region_out = self._deal_with_negative_region(region_out, out_buffer)

        self.in_buffer = in_buffer
        if region_in is None:
            self.region_in = [(0, self.in_buffer.shape[i_axis], 1) for i_axis in range(self.in_buffer.ndim)]
        else:
            self.region_in = region_in

        if out_buffer is None and region_out is None:
            shape = [ax[1] - ax[0] for ax in self.region_in]
            self.out_buffer = empty(queue, tuple(shape), dtype=self.in_buffer.dtype)
            self.region_out = [(0, i, 1) for i in shape]  # (tuple([0]*len(shape)),shape)
        elif out_buffer is not None and region_out is not None:
            self.out_buffer = out_buffer
            self.region_out = region_out
        else:
            raise ValueError('Case of input argument combination not supported')

        if self.in_buffer.dtype != self.out_buffer.dtype:
            raise ValueError('in and out buffer must be of same type')

        self.shape_region_out = tuple([ax[1] - ax[0] for ax in self.region_out])
        self.in_buffer = in_buffer

        self.copy_array_region = Kernel(name='copy_array_region',
                                        args={'in_buffer': Global(self.in_buffer, read_only=True),
                                                'out_buffer': Global(self.out_buffer, )},
                                        body=["""
                                  int addr_in = ${command_addr_in};
                                  int addr_out = ${command_addr_out};
                                  out_buffer[addr_out]=in_buffer[addr_in];
                                  """],
                                        replacements={'command_addr_in': self._command_for_addr_in_computation(),
                                                        'command_addr_out': self._command_for_addr_out_computation()},
                                        global_size=self.shape_region_out).compile(
            thread=Thread.from_buffer(in_buffer))

    def __call__(self, *args, **kwargs):
        self.copy_array_region()
        return self.out_buffer


def cl_set(array: Array, region: TypeSliceFormatCopyArrayRegion, value):
    """
    example usage:
    set slice of array with scalar value
    val = 1
    cl_set(ary, Slice[:,2:3], val)

    set slice of array with equally shaped numpy array like the slice
    some_np_array = np.array([[3,4])
    cl_set(ary, Slice[1:2,2:3], some_np_array)


    :param array:
    :param region:
    :param value:
    :return:
    """
    # todo test if array c contiguous
    region_arg = region
    # if slice is contiguous block of memory set it as
    # _buffer_np = np.zeros_like(add_symbols_memory_initialization.out_buffer)
    # _buffer_np[:, memory: -memory] = mapper.alphabet[0]
    # add_symbols_memory_initialization.out_buffer.set(_buffer_np)
    region = CopyArrayRegion._deal_with_incomplete_regions(region_arg, array)
    region = CopyArrayRegion._deal_with_none_in_stop(region, array)
    region = CopyArrayRegion._deal_with_negative_region(region, array)

    # test if requested region is
    for axis, _slice in enumerate(region):
        step_width = _slice[2]
        if abs(_slice[0] * step_width) > array.shape[axis] or abs(_slice[1] * step_width) > array.shape[axis]:
            raise ValueError('Slicing out of array bounds')
    if any([(part[0] - part[1]) == 0 for part in region]):  # check that there is no empty slice
        return
    global_size = np.product([part[1] - part[0] for part in region])
    target_shape = to_device(array.queue, np.array(array.shape).astype(Types.int))
    offset_target = to_device(array.queue, np.array([part[0] for part in region]).astype(Types.int))
    source_shape = to_device(array.queue, np.array([part[1] - part[0] for part in region]).astype(Types.int))
    source_n_dims = len(source_shape)

    if isinstance(value, np.ndarray):
        source = to_device(array.queue, value.astype(array.dtype))
        arg_source = Global(source)
        code_source = 'source[get_global_id(0)]'
    else:
        arg_source = Scalar(array.dtype)
        source = value
        code_source = 'source'

    knl = Kernel('set_cl_array',
                 {'target': Global(array),
                    'target_shape': Global(target_shape),
                    'offset_target': Global(offset_target),
                    'source': arg_source,
                    'source_shape': Global(source_shape),
                    'source_n_dims': Scalar(Types.int)},
                   """
   // id_source = get_global_id(0)
   // id_source points to element of array source which replaces element with id_target in array target.
   // we need to compute id_target from id_source:
   // we assume c-contiguous addressing like:
   // id_source = id0*s1*s2*s3+id1*s2*s3+id2*s3+id3 (here s refers shape of source array)
   // At first we need to compute individual ids of source array from id_source:
   // id3 = int(gid % s3), temp = (gid-id3)/s3
   // id2 = int(temp % s2), temp = (temp-id2)/s2
   // id1 = int(temp % s1), temp = (temp-id1)/s1
   // id0 = int(temp % s0), temp = (temp-id0)/s1
   // Finally, we can determine the id of the target array and copy element to corresponding position:
   // id_target = (id0*offset0t)*s1t*s2t ... (sxt: shape of target array along dim x)
   int id_target = 0; // to be iteratively computed from global id, slice dimensions and ary dimensions
   
   int temp = get_global_id(0);
   int prod_source_id_multiplier = 1;
   int prod_target_id_multiplier = 1;
   
   for(int i=source_n_dims-1; i>=0; i--){ // i=i_axis_source
    int id_source = temp % source_shape[i];
    temp = (int)((temp-id_source)/source_shape[i]);
    prod_source_id_multiplier *= source_shape[i];
    id_target += (offset_target[i]+id_source)*prod_target_id_multiplier;
    prod_target_id_multiplier *= target_shape[i];
   }
   target[id_target] = ${source};
                   """,
                 replacements={'addr': Helpers.command_compute_address(array.ndim),
                                 'source': code_source},
                 global_size=(global_size,)
                 ).compile(Thread.from_buffer(array), emulate=False)
    knl(source=source, source_n_dims=source_n_dims)


@mark.parametrize('s', [[0, 1, 2, 3],
                        [0, 10, 1, 3],
                        [4, 8, 0, 3],
                        [4, -1, 0, 3],
                        [0, 10, 0, 3]])
def test_cl_set(s):
    thread = Thread()
    ary = zeros(thread.queue, (10, 3), Types.double)
    ary_np = ary.get()
    val = 2
    # ClSet(ary)[:, 2:3] = val
    ary_np[s[0]:s[1], s[2]:s[3]] = val
    cl_set(ary, Slice[s[0]:s[1], s[2]:s[3]], val)

    assert np.all(ary_np == ary.get())


def test_cl_set_out_of_bounds():
    thread = Thread()
    ary = zeros(thread.queue, (10, 3), Types.double)
    val = 2
    try:
        cl_set(ary, Slice[:100, :], val)
    except ValueError as err:
        assert str(err) == 'Slicing out of array bounds'


def test_cl_set_many_dimensions():
    s = [0, 1, 2, 3, 0, 1]
    thread = Thread()
    ary = zeros(thread.queue, (10, 3, 4), Types.double)
    ary_np = ary.get()
    val = 2
    # ClSet(ary)[:, 2:3] = val
    ary_np[s[0]:s[1], :, s[4]:s[5]] = val
    cl_set(ary, Slice[s[0]:s[1], :, s[4]:s[5]], val)
    assert np.all(ary_np == ary.get())


class TypeConverter:
    def __init__(self, in_buffer: Array, out_buffer_dtype: np.dtype):
        self.in_buffer = in_buffer
        self.out_buffer = empty(in_buffer.queue, in_buffer.shape, dtype=out_buffer_dtype)
        knl = Kernel(name='type',
                     args={'in_buffer': Global(self.in_buffer, read_only=True),
                             'out_buffer': Global(self.out_buffer)},
                     body=["""
                                   int addr_in = ${command_addr_in};
                                   int addr_out = ${command_addr_out};
                                   out_buffer[addr_out]=convert_${buff_out_t}(in_buffer[addr_in]);
                                   """],
                     replacements={'command_addr_in': Helpers.command_compute_address(self.in_buffer.ndim),
                                     'command_addr_out': Helpers.command_compute_address(self.out_buffer.ndim),
                                     'buff_out_t': c_name_from_dtype(self.out_buffer.dtype)},
                     global_size=self.in_buffer.shape)
        thread = Thread(queue=in_buffer.queue, context=in_buffer.context)
        self.program = Program(kernels=[knl]).compile(thread=thread, emulate=False)

    def __call__(self):
        self.program.type()
        # self.in_buffer.queue.finish()
        return self.out_buffer
