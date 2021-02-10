import time
from math import log

__author__ = "piveloper"
__copyright__ = "10.02.2021, piveloper"
__version__ = "1.0"
__email__ = "piveloper@gmail.com"
__doc__ = """This module contains the FFT operation"""

from pyopencl.array import zeros_like, Array, zeros
from pytest import mark

from pyopencl_extension import ClFunction, ClKernel, KnlArgScalar, ClTypes, KnlArgBuffer, to_device, ArgScalar, \
    ArgBuffer, ClProgram, ClInit, ArgPrivate, set_b_use_existing_file_for_emulation
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
        set_b_use_existing_file_for_emulation(False)
        self.b_python = b_python

        # fir internal data types to input data type
        if in_buffer.dtype == ClTypes.cfloat:
            data_t = ClTypes.float2
            real_t = ClTypes.double
        elif in_buffer.dtype == ClTypes.cdouble:
            data_t = ClTypes.double2
            real_t = ClTypes.double
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
        T = 4*ClInit.from_buffer(in_buffer).device.global_mem_cacheline_size
        defines = {'R': (R := 2),  # currently only Radix 2 fft is supported
                   'N': N,
                   'M': M,  # number of FFTs to perform simultaneously
                   'T': T  # work group size)
                   }
        knl_gpu_fft = ClKernel('gpu_fft',
                               {'Ns': KnlArgScalar(ClTypes.int),
                                'data_in': KnlArgBuffer(data_in, '__const'),
                                'data_out': KnlArgBuffer(data_out)},
                               """
                 int t = get_local_id(2);
                 int by = get_global_id(0); int bx = get_global_id(1); 
                 int idx_butterfly = bx*T + t;               
                 if(idx_butterfly<(int)(N/2)){ // in case N/2 is not multiple of Threads T, deactivate certain work items
                    int j = by*N + idx_butterfly; // as proposed in [1, p.3, text section A]
                    fft_iteration(j, Ns, data_in, data_out);
                 }
                 """,
                               global_size=(M, max(1, int(N / (R * T))), T),
                               local_size=(1, 1, T))
        func_fft_iteration = ClFunction('fft_iteration',
                                        {'j': ArgScalar(ClTypes.int),
                                         'Ns': ArgScalar(ClTypes.int),
                                         'data0': ArgBuffer(data_in.dtype),
                                         'data1': ArgBuffer(data_out.dtype)},
                                        """
        private data_t v[R];
        int idxS = j;
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
        
        // changed line 27 of [1, Figure 2] to work with global dimensions of this class:
        int idxD = (int)(j/N)*N + expand(j%N, Ns, R);
        for(int r=0; r<R; r++)
            data1[idxD + r*Ns] = v[r];        
                   """,
                                        replacements={
                                            'fft_radix_R': f'fft_radix_{R}'
                                        })
        func_fft_radix_2 = ClFunction('fft_radix_2',
                                      {'v': ArgPrivate(data_t)},
                                      """
                               data_t v0;
                                v0= v[0];
                               v[0] = v0 + v[1];
                               v[1] = v0 - v[1];
                               """)
        # [1]: The expand() function can be thought of as inserting a dimension of length N2 after the first
        # dimension of length N1 in a linearized index.
        func_expand = ClFunction('expand',
                                 {'idxL': ArgScalar(ClTypes.int),
                                  'N1': ArgScalar(ClTypes.int),
                                  'N2': ArgScalar(ClTypes.int)},
                                 """
                                 return (int)(idxL/N1)*N1*N2 + (idxL%N1);
                                 """,
                                 return_type=ClTypes.int)

        self.cl_program = ClProgram([func_expand, func_fft_radix_2, func_fft_iteration],
                                    [knl_gpu_fft], defines=defines, type_defs=typedefs
                                    ).compile(ClInit.from_buffer(in_buffer), b_python=self.b_python)
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
data = data.astype(ClTypes.cdouble)


@mark.parametrize("in_data_np", [
    # np.arange(2 ** 3).astype(ClTypes.cfloat),
    np.random.random((50, 2 ** 16,)).astype(ClTypes.cdouble),
    # np.sin(np.linspace(0, 10 * 2 * np.pi, 2 ** 8)).astype(ClTypes.cdouble),
    # np.arange(2 ** 3).astype(ClTypes.cfloat),
])
def test_fft(in_data_np):
    import numpy as np
    np.random.seed(0)
    t_np = time.time()
    fft_in_data_np = np.fft.fft(in_data_np, axis=-1)
    t_np = time.time() - t_np

    cl_init = ClInit(b_profiling_enable=True)
    in_data_cl = to_device(cl_init.queue, in_data_np)

    fft_cl = Fft(in_data_cl)

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
        assert np.allclose(fft_in_data_cl_emulation.get().view(ClTypes.double),
                           fft_in_data_cl.get().view(ClTypes.double))

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
    #numpy.fft.fftn(data[:, :, 0], axes=(1,))
    treikna_min = min(ts)
    assert np.allclose(fft_in_data_np, res_dev.get())
    t = 0