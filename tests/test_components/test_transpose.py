import numpy as np
from pyopencl.array import to_device

from pyopencl_extension.components.transpose import Transpose
from pyopencl_extension.np_cl_types import ClTypes


def test_transpose_single_dimension(cl_init):
    ary_np = np.array([[1, 2, 3]], dtype=ClTypes.int).T
    in_cl = to_device(cl_init.queue, ary_np)
    transpose_on_input_buffer = Transpose(in_cl, (1, 0))
    out_cl = transpose_on_input_buffer()
    assert np.all(out_cl.get() == ary_np.T)


def test_transpose_int(cl_init):
    ary_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=ClTypes.int)
    ary_np = np.hstack([ary_np, ary_np])  # array([[1, 2, 3], [4, 5, 6]])
    in_cl = to_device(cl_init.queue, ary_np)
    transpose_in_buffer = Transpose(in_cl, (1, 0))
    out_cl = transpose_in_buffer()
    assert np.all(out_cl.get() == ary_np.T)


def test_transpose_float(cl_init):
    ary_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=ClTypes.float)
    ary_np = np.hstack([ary_np, ary_np])  # array([[1, 2, 3], [4, 5, 6]])
    in_cl = to_device(cl_init.queue, ary_np)
    transpose_in_buffer = Transpose(in_cl, (1, 0))
    out_cl = transpose_in_buffer()
    assert np.all(out_cl.get() == ary_np.T)


def test_transpose_complex(cl_init):
    ary_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=ClTypes.cfloat)
    ary_np = np.hstack([ary_np, ary_np])  # array([[1, 2, 3], [4, 5, 6]])
    in_cl = to_device(cl_init.queue, ary_np)
    transpose_in_buffer = Transpose(in_cl, (1, 0))
    out_cl = transpose_in_buffer()
    assert np.all(out_cl.get() == ary_np.T)