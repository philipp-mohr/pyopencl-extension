import time
from math import log

__author__ = "piveloper"
__copyright__ = "10.02.2021, piveloper"
__version__ = "1.0"
__email__ = "piveloper@gmail.com"
__doc__ = """This module contains the FFT operation"""

from pyopencl.array import zeros_like, Array, zeros
from pytest import mark

from pyopencl_extension import Function, Kernel, Scalar, Types, Global, to_device, Scalar, \
    Global, Program, Thread, Private, set_b_use_existing_file_for_emulation, Local
from pyopencl_extension.components.copy_array_region import CopyArrayRegion, Slice
import numpy as np


class Fft:
    """
    Implementation of reference algorithm [1, Fig. 2]

    [1] N. K. Govindaraju, B. Lloyd, Y. Dotsenko, B. Smith, und J. Manferdelli, „High performance discrete Fourier
    transforms on graphics processors“, in 2008 SC - International Conference for High Performance Computing,
    Networking, Storage and Analysis, Austin, TX, Nov. 2008, S. 1–12, doi: 10.1109/SC.2008.5213922.
    """

    def __call__(self, *args, **kwargs):
        self._copy_in_buffer()
        data_in = self._data_in
        data_out = self._data_out
        Ns = 1
        n_stages = int(log(self.N, self.R))
        for i in range(n_stages):
            self.cl_program.gpu_fft(Ns=Ns,
                                    data_in=data_in,
                                    data_out=data_out)
            Ns = Ns * self.R
            data_in, data_out = data_out, data_in  # swap
        return data_in.view(self.in_buffer.dtype).reshape(self._out_shape)

    def __init__(self, in_buffer: Array, b_python=False):
        # just for debugging:
        set_b_use_existing_file_for_emulation(True)
        self.b_python = b_python

        # fir internal data types to input data type
        if in_buffer.dtype == Types.cfloat:
            data_t = Types.float2
            real_t = Types.double
        elif in_buffer.dtype == Types.cdouble:
            data_t = Types.double2
            real_t = Types.double
        else:
            raise ValueError()

        # input ist zero padded such that its length becomes a power of 2.
        # There exists fast FFT versions for non power of two, but those are currently not implemented.
        # For further information read here:
        # https://math.stackexchange.com/questions/77118/non-power-of-2-ffts
        # https://stackoverflow.com/questions/13841296/fft-in-numpy-python-when-n-is-not-a-power-of-2
        if in_buffer.ndim == 1:
            b_1d_input = True
            in_buffer = in_buffer.reshape((1, in_buffer.shape[0]))
        else:
            b_1d_input = False
        N = in_buffer.shape[1]
        M = in_buffer.shape[0]
        if not np.log2(N).is_integer():  # if not power of 2, pad accordingly
            N = 2 ** int(np.log2(N) + 1)
        if b_1d_input:
            self._out_shape = (N,)
        else:
            self._out_shape = (M, N)

        _data_in = zeros(in_buffer.queue, (M, N), in_buffer.dtype)
        self._copy_in_buffer = CopyArrayRegion(in_buffer=in_buffer, region_in=Slice[:, :],
                                               region_out=Slice[:, :in_buffer.shape[1]], out_buffer=_data_in)
        data_in = _data_in.view(data_t)
        data_out = zeros_like(data_in)
        typedefs = {'data_t': data_t,
                    'real_t': real_t}
        T = min(int(N / 2), 4 * Thread.from_buffer(in_buffer).device.global_mem_cacheline_size)
        defines = {'R': (R := 2),  # currently only Radix 2 fft is supported
                   'N': N,
                   'M': M,  # number of FFTs to perform simultaneously
                   'T': T  # work group size)
                   }
        knl_gpu_fft = Kernel('gpu_fft',
                             {'Ns': Scalar(Types.int),
                              # In each iteration, the algorithm can be thought of combining the radix R FFTs on
                              # subsequences of length Ns into the FFT of a new sequence of length RNs by
                              # performing an FFT of length R on the corresponding elements of the subsequences.
                              'data_in': Global(data_in, '__const'),
                              'data_out': Global(data_out)},
                             """
                 local real_t shared[2*T*R];
                 int t = get_local_id(2);
                 int b = get_global_id(1); 
                 int idx_butterfly = b*T + t;               
                 if(idx_butterfly<(int)(N/2)){ // in case N/2 is not multiple of Threads T, deactivate certain work items
                    int j = idx_butterfly; // as proposed in [1, p.3, text section A]
                    fft_iteration(j, Ns, data_in, data_out, shared);
                 }
                 """,
                             global_size=(M, max(1, int(N / (R * T))), T),
                             local_size=(1, 1, T))
        func_fft_iteration = Function('fft_iteration',
                                      {'j': Scalar(Types.int),
                                       'Ns': Scalar(Types.int),
                                       'data0': Global(data_in.dtype, '__const'),
                                       'data1': Global(data_out.dtype),
                                       'shared': Local(real_t)},
                                      """
        int offset_block = get_global_id(0)*N;
        private data_t v[R];
        int idxS = offset_block + j; // idxS=idxSource
        real_t angle = -2*PI*(j % Ns)/(Ns*R);
        for(int r=0; r<R; r++){
            v[r] = data0[(int)(idxS + r* N/R)];
            //v[r]*=(data_t)(cos(r*angle), sin(r*angle)); //todo how to assign vector dtype?
            real_t cos_angle = cos(r*angle);
            real_t sin_angle = sin(r*angle);
            v[r] = (data_t)(v[r].s0*cos_angle-v[r].s1 *sin_angle,
                            v[r].s0*sin_angle+v[r].s1 *cos_angle);
        }
        ${fft_radix_R}(v);
        
        /* According to [1, p.4] commented block is replaced below, such that global memory writes are coalesced
        // changed line 27 of [1, Figure 2] to work with global dimensions of this class:
        int idxD = offset_block + expand(j, Ns, R); // idxD=idxDestination
        for(int r=0; r<R; r++)
            data1[idxD + r*Ns] = v[r];
        */ // Replaced with:
        int b = get_global_id(1);
        int t = get_local_id(2);
        int idxD = (int)(t/Ns)*R + (t%Ns);
        exchange( v, idxD, Ns, t, T, shared);
        idxD = offset_block + b*R*T + t;
        for( int r=0; r<R; r++ )
            data1[idxD+r*T] = v[r];    
                   """,
                                      replacements={
                                          'fft_radix_R': f'fft_radix_{R}'
                                      })
        func_fft_radix_2 = Function('fft_radix_2',
                                    {'v': Private(data_t)},
                                    """
                               data_t v0;
                               v0= v[0];
                               v[0] = v0 + v[1];
                               v[1] = v0 - v[1];
                               """)
        # [1]: The expand() function can be thought of as inserting a dimension of length N2 after the first
        # dimension of length N1 in a linearized index.
        func_expand = Function('expand',
                               {'idxL': Scalar(Types.int),
                                'N1': Scalar(Types.int),
                                'N2': Scalar(Types.int)},
                               """
                                 return (int)(idxL/N1)*N1*N2 + (idxL%N1);
                                 """,
                               return_type=Types.int)
        # float2* v, int R, int idxD, int incD, int idxS, int incS
        func_exchange = Function('exchange',
                                 {'v': Private(data_t),
                                  'idxD': Scalar(Types.int),
                                  'incD': Scalar(Types.int),
                                  'idxS': Scalar(Types.int),
                                  'incS': Scalar(Types.int),
                                  'shared': Local(real_t),
                                  },
                                 """
                                __local float* sr = shared;
                                __local float* si = shared+T*R;
                                barrier(CLK_LOCAL_MEM_FENCE);
                                for( int r=0; r<R; r++ ) {
                                    int i = (idxD + r*incD)*STRIDE;
                                    sr[i] = v[r].s0;
                                    si[i] = v[r].s1;
                                }
                                barrier(CLK_LOCAL_MEM_FENCE);
                                for(int r=0; r<R; r++ ) {
                                    int i = (idxS + r*incS)*STRIDE;
                                    v[r] = (data_t)(sr[i], si[i]);
                                }
                                 """,
                                 return_type=Types.int,
                                 defines={'STRIDE': 1})

        self.cl_program = Program([func_exchange, func_expand, func_fft_radix_2, func_fft_iteration],
                                  [knl_gpu_fft], defines=defines, type_defs=typedefs
                                  ).compile(Thread.from_buffer(in_buffer), b_python=self.b_python)
        self.N = N
        self.R = R
        self.in_buffer = in_buffer
        self._data_in = data_in
        self._data_out = data_out


# todo :remove seed
np.random.seed(0)
shape = (1024, 16, 16, 16)
axes = (1, 2, 3)

data = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
data = data.astype(Types.cdouble)


@mark.parametrize("in_data_np", [
    np.random.random((2, 2 ** 3,)).astype(Types.cdouble),
    (_ := np.arange((2 ** 8))).reshape((1, _.shape[0])).astype(Types.cdouble),
    np.random.random((50, 2 ** 16,)).astype(Types.cdouble),
    # np.sin(np.linspace(0, 10 * 2 * np.pi, 2 ** 8)).astype(ClTypes.cdouble),
    # np.arange(2 ** 3).astype(ClTypes.cfloat),
])
def test_fft(in_data_np):
    import numpy as np
    np.random.seed(0)
    t_np = time.time()
    fft_in_data_np = np.fft.fft(in_data_np, axis=-1)
    t_np = time.time() - t_np

    thread = Thread(b_profiling_enable=False)
    in_data_cl = to_device(thread.queue, in_data_np)

    fft_cl = Fft(in_data_cl, b_python=False)

    attempts = 2
    ts = []
    for i in range(attempts):
        t1 = time.time()
        fft_in_data_cl = fft_cl()
        fft_in_data_cl.queue.finish()
        t2 = time.time()
        ts.append(t2 - t1)
    t_cl = min(ts)
    if in_data_np.size < 512:
        # Test against emulation (commented since it is slower)
        fft_cl_emulation = Fft(in_data_cl, b_python=True)
        fft_in_data_cl_emulation = fft_cl_emulation()
        assert np.allclose(fft_in_data_np, fft_in_data_cl_emulation.get().view(Types.double))
        assert np.allclose(fft_in_data_cl_emulation.get().view(Types.double),
                           fft_in_data_cl.get().view(Types.double))

    # import matplotlib.pyplot as plt
    # plt.plot(fft_in_data_np.flatten())
    # plt.plot(fft_in_data_cl_emulation.get().flatten())
    # plt.show()
    assert np.allclose(fft_in_data_np, fft_in_data_cl.get())

    # benchmark using reikna
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
    t = 0
