import numpy as np

from pyopencl_extension import SumAlongAxis, to_device


def test_sum_along_axis(thread):
    ary = np.array([[1, 2, 3], [1, 2, 3]])
    ary_buffer = to_device(thread.queue, ary)
    """"
    sum_along_axis = SumAlongAxis(ary_buffer, axis=0)
    res = sum_along_axis().get()
    ref = ary.sum(axis=0)
    assert np.all(res == ref)
    """

    sum_along_axis = SumAlongAxis(ary_buffer, axis=1)
    res = sum_along_axis().get()
    ref = ary.sum(axis=1)
    assert np.all(res == ref)


def test_sum_along_axis_1d(thread):
    ary = np.array([1, 2, 3])
    ary_buffer = to_device(thread.queue, ary)

    sum_along_axis = SumAlongAxis(ary_buffer, axis=0)
    # res_py = sum_along_axis(emulate=True).get()
    res_cl = sum_along_axis().get()
    ref = ary.sum(axis=0)
    assert np.all(res_cl == ref)
