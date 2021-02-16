import numpy as np
from pyopencl.array import to_device

from pyopencl_extension import Transpose
from pyopencl_extension import Types


def test_transpose_single_dimension(thread):
    ary_np = np.array([[1, 2, 3]], dtype=Types.int).T
    in_cl = to_device(thread.queue, ary_np)
    transpose_on_input_buffer = Transpose(in_cl, (1, 0))
    out_cl = transpose_on_input_buffer()
    assert np.all(out_cl.get() == ary_np.T)


def test_transpose_int(thread):
    ary_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=Types.int)
    ary_np = np.hstack([ary_np, ary_np])  # array([[1, 2, 3], [4, 5, 6]])
    in_cl = to_device(thread.queue, ary_np)
    transpose_in_buffer = Transpose(in_cl, (1, 0))
    out_cl = transpose_in_buffer()
    assert np.all(out_cl.get() == ary_np.T)


def test_transpose_float(thread):
    ary_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=Types.float)
    ary_np = np.hstack([ary_np, ary_np])  # array([[1, 2, 3], [4, 5, 6]])
    in_cl = to_device(thread.queue, ary_np)
    transpose_in_buffer = Transpose(in_cl, (1, 0))
    out_cl = transpose_in_buffer()
    assert np.all(out_cl.get() == ary_np.T)


def test_transpose_complex(thread):
    ary_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=Types.cfloat)
    ary_np = np.hstack([ary_np, ary_np])  # array([[1, 2, 3], [4, 5, 6]])
    in_cl = to_device(thread.queue, ary_np)
    transpose_in_buffer = Transpose(in_cl, (1, 0))
    out_cl = transpose_in_buffer()
    assert np.all(out_cl.get() == ary_np.T)