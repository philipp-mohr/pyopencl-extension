import time

import numpy as np
from pyopencl.array import to_device, empty_like, empty
from scipy.signal import fftconvolve

from pyopencl_extension import Thread, Types, Kernel, Global, Profiling, ClHelpers, Function, Global, \
    Scalar
from pyopencl_extension.components.convolve import Convolution1D


def get_fft_function(size_input, type_input, n_axes):
    N = size_input
    if not np.log2(N).is_integer():
        raise ValueError('input size must be power of 2')
    defines = {
        'N_STAGES': int(np.log2(N)),
        'N_INPUTS': int(np.log2(N)),
        'AXIS': n_axes,
               }
    typedefs = {'cplx_t': type_input}
    # Work group size is assumed to be N
    func = Function('radix2_fft',
                    {'x': Global(type_input),
                       'i_x': Scalar(Types.int)
                       },
                      """
               local cplx_t[N_INPUTS] a;
               local cplx_t[N_INPUTS] b;
               b = x;//interleave x according to radix2 requirements
               for(int i_stage=0; i_stage<N_STAGES; i_stage++){
                    // prepare input
                    if(i_x==0){
                        a[i_x] = MUL( CPLX_EXP(2*PI*0)) , b[i_x] );
                    }else
                    barrier(CLK_LOCAL_MEM_FENCE);
                    b[i_x] = a[i_x+1] + a[i_x];
               }
               """,
                    defines=defines,
                    type_defs=typedefs)
    return local_size_last

def test_cl_fft():
    # initialize context and queue
    thread = Thread()
    queue = thread.queue
    signal_t = Types.int
    x = to_device(queue, (10 * (np.random.random((1000000,)) - 0.5)).astype(signal_t))
    get_fft_function(1024)


def test_convolution():
    n_repetitions = 10

    # initialize context and queue
    thread = Thread(b_profiling_enable=True)
    queue = thread.queue

    signal_t = Types.int
    x = to_device(queue, (10 * (np.random.random((1000000,)) - 0.5)).astype(signal_t))
    h = to_device(queue, (10 * (np.random.random((1000,)) - 0.5)).astype(signal_t))
    convolve = Convolution1D(x, h)
    convolve()

    cl_t = time.time()
    for _ in range(n_repetitions):
        convolve()
    queue.finish()
    cl_t = time.time() - cl_t

    # reference
    np_x = x.get()
    np_h = h.get()
    np_t = time.time()
    for _ in range(n_repetitions):
        np_y = fftconvolve(np_x, np_h)
        # np_y = np.convolve(np_x, np_h)
    np_t = time.time() - np_t

    diff = convolve.out_buffer.get() - np_y
    performance_improvement = np_t / cl_t
    assert np.allclose(convolve.out_buffer.get(), np_y, atol=1e-7)

    #Profiling(queue).show_histogram_cumulative_kernel_times()
