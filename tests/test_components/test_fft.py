__author__ = "piveloper"
__copyright__ = "10.02.2021, piveloper"
__version__ = "1.0"
__email__ = "piveloper@gmail.com"
__doc__ = """This module contains tests for the FFT operation. Radix 2 and """

from pyopencl_extension.components.fft import Fft
from pyopencl_extension import Types, to_device, Thread, set_b_use_existing_file_for_emulation

import time
from pytest import mark
import numpy as np


def get_in_data_cplx(M, N, dtype=Types.cdouble):
    np.random.seed(0)
    in_data_np = np.random.random((M, N,)).astype(dtype)
    in_data_np.imag = np.random.random((M, N,)).astype(in_data_np.imag.dtype)
    return in_data_np


def get_in_data_cplx_radix(radix=2, exponent=10):
    np.random.seed(0)
    return get_in_data_cplx(50, radix ** exponent)


@mark.parametrize("in_data_np", [
    #np.random.random((2 ** 14,)).astype(Types.cdouble),
    get_in_data_cplx(1, 512, dtype=Types.cdouble),
    get_in_data_cplx(20, 10000, dtype=Types.cdouble),
    #np.random.random((1,2 ** 9,)).astype(Types.cdouble),
    get_in_data_cplx_radix(radix=16, exponent=3),
    get_in_data_cplx_radix(radix=2, exponent=10),
    get_in_data_cplx_radix(radix=4, exponent=5),
    get_in_data_cplx_radix(radix=8, exponent=4)])
def test_fft(in_data_np):
    atol = 1e-4 if in_data_np.dtype == Types.cfloat else 1e-8
    import numpy as np

    thread = Thread(b_profiling_enable=False)
    in_data_cl = to_device(thread.queue, in_data_np)

    fft_cl = Fft(in_data_cl, b_python=False)

    # zero padding data for numpy
    axis = 1
    N = in_data_np.shape[axis]
    if not np.log2(N).is_integer():  # if not power of 2, pad accordingly
        N = 2 ** int(np.log2(N) + 1)
    in_data_np_power_of_two = np.zeros((in_data_np.shape[0], N), in_data_np.dtype)
    in_data_np_power_of_two[:, :in_data_np.shape[axis]] = in_data_np

    def measure(call):
        attempts = 3
        ts = []
        for i in range(attempts):
            t1 = time.time()
            call()
            t2 = time.time()
            ts.append(t2 - t1)
        return min(ts)

    import pyfftw
    t_fftw = measure(lambda: pyfftw.interfaces.numpy_fft.fft(in_data_np_power_of_two, axis=-1))
    t_np = measure(lambda: np.fft.fft(in_data_np_power_of_two, axis=-1))
    fft_in_data_np = np.fft.fft(in_data_np_power_of_two, axis=-1)

    def fft_call():
        fft_in_data_cl = fft_cl()
        fft_in_data_cl.queue.finish()
    t_cl = measure(fft_call)
    fft_in_data_cl = fft_cl()

    if in_data_np.size < 1024:
        # Test against emulation (commented since it is slower)
        set_b_use_existing_file_for_emulation(False)
        fft_cl_py = Fft(in_data_cl, b_python=True)

        fft_in_data_cl_py = fft_cl_py()
        a = fft_in_data_cl_py.get().view(Types.cdouble)
        b = fft_in_data_cl.get().view(Types.cdouble)
        c = fft_in_data_np.view(Types.cdouble)
        assert np.allclose(a, b)
        assert np.allclose(c, b)
        assert np.allclose(c, a)


    # import matplotlib.pyplot as plt
    # plt.plot(fft_in_data_np.flatten())
    # plt.plot(fft_in_data_cl_emulation.get().flatten())
    # plt.show()
    assert np.allclose(fft_in_data_np, fft_in_data_cl.get(), atol=atol)
    # benchmark using reikna
    if False:  # change to true to run against reikna's fft. Note: Reikna takes quite some optimization time before run
        from reikna.cluda import any_api
        from reikna.fft import FFT
        import numpy
        api = any_api()
        thr = api.Thread.create()
        data = in_data_np
        dtype = data.dtype
        axes = (1,)
        fft = FFT(data, axes=axes)
        fftc = fft.compile(thr)
        data_dev = thr.to_device(data)
        res_dev = thr.empty_like(data_dev)
        ts = []
        for i in range(attempts):
            t1 = time.time()
            fftc(res_dev, data_dev)
            thr.synchronize()
            t2 = time.time()
            ts.append(t2 - t1)
        fwd_ref = numpy.fft.fftn(data, axes=axes).astype(dtype)
        tnp = time.time()
        fwd_ref = numpy.fft.fftn(data, axes=axes).astype(dtype)
        tnp = time.time() - tnp
        # numpy.fft.fftn(data[:, :, 0], axes=(1,))
        treikna_min = min(ts)
        assert np.allclose(fft_in_data_np, res_dev.get())
