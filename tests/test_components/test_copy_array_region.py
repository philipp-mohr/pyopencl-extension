import numpy as np

from pyopencl_extension import CopyArrayRegion, Slice, to_device


def test_copy_array_region_on_device(thread):
    ary_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.complex64)
    in_cl = to_device(thread.queue, ary_np)
    copy_region = CopyArrayRegion(in_cl, region_in=Slice[0:1, 1:3])

    out = copy_region()

    out_np = in_cl.get()[0:1, 1:3]

    assert np.all(out.get() == out_np)


def test_copy_array_region_on_device_region_none(thread):
    ary_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.complex64)
    in_cl = to_device(thread.queue, ary_np)
    copy_region = CopyArrayRegion(in_cl, region_in=None)

    out = copy_region()

    out_np = in_cl.get()

    assert np.all(out.get() == out_np)


def test_copy_array_region_on_device_between_buffers(thread):
    ary_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.complex64)
    in_cl = to_device(thread.queue, ary_np)
    out_np = np.zeros(shape=(4, 4), dtype=in_cl)
    out_cl = to_device(thread.queue, out_np)
    copy_region = CopyArrayRegion(in_buffer=in_cl,
                                  region_in=Slice[:1, 1:3],
                                  out_buffer=out_cl,
                                  region_out=Slice[1:2, 2:4])
    out = copy_region()
    out_np[1:2, 2:4] = in_cl.get()[0:1, 1:3]
    assert np.all(out.get() == out_np)


def test_copy_array_region_on_device_negative_indexing(thread):
    ary_np = np.ones(shape=(5, 10), dtype=np.int32)
    in_cl = to_device(thread.queue, ary_np)
    copy_region = CopyArrayRegion(in_buffer=in_cl,
                                  region_in=Slice[:, 2:-1])
    out = copy_region()
    out_np = in_cl.get()[:, 2:-1]
    assert np.all(out.get() == out_np)


def test_copy_array_region_on_device_given_axis_index(thread):
    ary_np = np.ones(shape=(5, 10), dtype=np.int32)
    in_cl = to_device(thread.queue, ary_np)
    copy_region = CopyArrayRegion(in_buffer=in_cl,
                                  region_in=Slice[2:-1, :])
    out = copy_region()
    out_np = in_cl.get()[2:-1, :]
    assert np.all(out.get() == out_np)

# todo
# def test_step_width_larger_one(thread):
#     ary_np = np.array([np.arange(10, dtype=np.int32)] * 2)
#     in_cl = cl.array.to_device(thread.queue, ary_np)
#     out_cl = cl.array.to_device(thread.queue, ary_np)
#     copy_region = CopyArrayRegion(in_buffer=in_cl,
#                                   region_in=Slice[:, ::2],
#                                   out_buffer=out_cl[:, 1::2])
#     out = copy_region()
#     out_np = in_cl.get()[:, ::2]
#     assert np.all(out.get() == out_np)
