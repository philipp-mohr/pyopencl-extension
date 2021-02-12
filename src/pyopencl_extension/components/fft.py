import time
from math import log

__author__ = "piveloper"
__copyright__ = "10.02.2021, piveloper"
__version__ = "1.0"
__email__ = "piveloper@gmail.com"
__doc__ = """This module contains the FFT operation. Radix 2 and """

from pyopencl.array import zeros_like, Array, zeros
from pytest import mark

from pyopencl_extension import Function, Kernel, Scalar, Types, Global, to_device, Scalar, \
    Global, Program, Thread, Private, set_b_use_existing_file_for_emulation, Local
from pyopencl_extension.components.copy_array_region import CopyArrayRegion, Slice
import numpy as np


class Fft:
    """
    Implementation of reference algorithm [1, Fig. 2] (where only radix 2 kernel is provided)
    A reference for higher radix kernels can be found in [2]

    [1] N. K. Govindaraju, B. Lloyd, Y. Dotsenko, B. Smith, und J. Manferdelli, „High performance discrete Fourier
    transforms on graphics processors“, in 2008 SC - International Conference for High Performance Computing,
    Networking, Storage and Analysis, Austin, TX, Nov. 2008, S. 1–12, doi: 10.1109/SC.2008.5213922.

    [2] http://www.bealto.com/gpu-fft_opencl-2.html
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

    def __init__(self, in_buffer: Array, b_python=False, radix: int = 2):
        # just for debugging:
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
        R = radix
        T = min(int(N / R), Thread.from_buffer(in_buffer).device.global_mem_cacheline_size)
        defines = {'R': R,
                   'N': N,
                   'M': M,  # number of FFTs to perform simultaneously
                   'T': T,  # work group size)
                   'CW': (CW := Thread.from_buffer(in_buffer).device.global_mem_cacheline_size)  # Coalescing width
                   }
        knl_gpu_fft = Kernel('gpu_fft',
                             {'Ns': Scalar(Types.int),
                              # In each iteration, the algorithm can be thought of combining the radix R FFTs on
                              # subsequences of length Ns into the FFT of a new sequence of length RNs by
                              # performing an FFT of length R on the corresponding elements of the subsequences.
                              'data_in': Global(data_in, '__const'),
                              'data_out': Global(data_out)},
                             """
                 local real_t shared[(int)(2*T*R)];//2*Ns];//T*R];
                 int t = get_local_id(2);
                 int b = get_global_id(1); 
                 int idx_butterfly = b*T + t;               
                 if(idx_butterfly<(int)(N/R)){ // in case N/R is not multiple of Threads T, deactivate certain work items
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
        
        // According to [1, p.4] commented block is replaced below, such that global memory writes are coalesced
        if(Ns>=CW){
            // changed line 27 of [1, Figure 2] to work with global dimensions of this class:
            int idxD = offset_block + expand(j, Ns, R); // idxD=idxDestination
            for(int r=0; r<R; r++)
                data1[idxD + r*Ns] = v[r];
        }else{
            int b = get_global_id(1);
            int t = get_local_id(2);
            // !! mistake in [1] where *Ns is missing: idxD = (int)(t/Ns)*R + (t%Ns); !!
            int idxD = (int)(t/Ns)*Ns*R + (t%Ns);
            exchange( v, idxD, Ns, t, T, shared);
            idxD = offset_block + b*R*T + t;
            for( int r=0; r<R; r++ )
                data1[idxD+r*T] = v[r];  
        }
                   """,
                                      replacements={
                                          'fft_radix_R': f'fft_radix_{R}'
                                      })
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
                                __local real_t* sr = shared; // vector with real part
                                __local real_t* si = shared+T*R; // vector with imaginary part
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
        # ----------------------------------------
        # Definition of radix 2 - 16 functions:
        func_fft_radix_2 = Function('fft_radix_2',
                                    {'v': Private(data_t)},
                                    """
                               data_t v0= v[0];
                               v[0] = v0 + v[1];
                               v[1] = v0 - v[1];
                               """)
        # mul_pxqy(a) returns a*exp(-j * PI * p / q) where p=x and q=y
        mul_pxpy_dict = {'mul_p1q2': (mul_p1q2 := 'return (data_t)(a.y,-a.x);')}
        funcs_mulpxpy_4 = [Function(k, {'a': Scalar(data_t)}, v, return_type=data_t) for k, v in mul_pxpy_dict.items()]
        func_fft_radix_4 = Function('fft_radix_4',
                                    {'v': Private(data_t)},
                                    """
                // 2x DFT2 and twiddle
                data_t v0 = v[0] + v[2];
                data_t v1 = v[0] - v[2];
                data_t v2 = v[1] + v[3];
                data_t v3 = mul_p1q2(v[1] - v[3]); // twiddle
                
                // 2x DFT2 and store
                v[0] = v0 + v2;
                v[1] = v1 + v3;
                v[2] = v0 - v2;
                v[3] = v1 - v3;
                                    """)
        funcs_radix_4 = funcs_mulpxpy_4 + [func_fft_radix_4]

        defines['SQRT_1_2'] = np.cos(np.pi / 4)
        mul_pxpy_dict = {'mul_p0q2': (mul_p0q2 := 'return a;'),
                         'mul_p0q4': mul_p0q2,
                         'mul_p2q4': (mul_p2q4 := mul_p1q2),
                         'mul_p1q4': (mul_p1q4 := 'return (data_t)(SQRT_1_2)*(data_t)(a.x+a.y,-a.x+a.y);'),
                         'mul_p3q4': (mul_p3q4 := 'return (data_t)(SQRT_1_2)*(data_t)(-a.x+a.y,-a.x-a.y);'),
                         }
        funcs_mulpxpy_8 = [Function(k, {'a': Scalar(data_t)}, v, return_type=data_t) for k, v in mul_pxpy_dict.items()]
        func_fft_radix_8 = Function('fft_radix_8',
                                    {'v': Private(data_t)},
                                    """
                // 4x in-place DFT2
                data_t u0 = v[0];
                data_t u1 = v[1];
                data_t u2 = v[2];
                data_t u3 = v[3];
                data_t u4 = v[4];
                data_t u5 = v[5];
                data_t u6 = v[6];
                data_t u7 = v[7];
                
                data_t v0 = u0 + u4;
                data_t v4 = mul_p0q4(u0 - u4);
                data_t v1 = u1 + u5;
                data_t v5 = mul_p1q4(u1 - u5);
                data_t v2 = u2 + u6;
                data_t v6 = mul_p2q4(u2 - u6);
                data_t v3 = u3 + u7;
                data_t v7 = mul_p3q4(u3 - u7);
                
                // 4x in-place DFT2 and twiddle
                u0 = v0 + v2;
                u2 = mul_p0q2(v0 - v2);
                u1 = v1 + v3;
                u3 = mul_p1q2(v1 - v3);
                u4 = v4 + v6;
                u6 = mul_p0q2(v4 - v6);
                u5 = v5 + v7;
                u7 = mul_p1q2(v5 - v7);
                
                // 4x DFT2 and store (reverse binary permutation)
                v[0]   = u0 + u1;
                v[1]   = u4 + u5;
                v[2] = u2 + u3;
                v[3] = u6 + u7;
                v[4] = u0 - u1;
                v[5] = u4 - u5;
                v[6] = u2 - u3;
                v[7] = u6 - u7;
                                    """)
        funcs_radix_8 = funcs_mulpxpy_8 + [func_fft_radix_8]

        func_mul1 = Function('mul_1',
                             {'a': Scalar(data_t), 'b': Scalar(data_t)},
                             'data_t x; x.even = MUL_RE(a,b); x.odd = MUL_IM(a,b); return x;', return_type=data_t,
                             defines={'MUL_RE(a,b)': '(a.even*b.even - a.odd*b.odd)',
                                      'MUL_IM(a,b)': '(a.even*b.odd + a.odd*b.even)'})

        mul_pxpy_dict = {'mul_p0q8 ': mul_p0q2,
                         'mul_p1q8': 'return mul_1((data_t)(COS_8,-SIN_8),a);',
                         'mul_p2q8': mul_p1q4,
                         'mul_p3q8': 'return mul_1((data_t)(SIN_8,-COS_8),a);',
                         'mul_p4q8 ': mul_p2q4,
                         'mul_p5q8': 'return mul_1((data_t)(-SIN_8,-COS_8),a);',
                         'mul_p6q8': mul_p3q4,
                         'mul_p7q8': 'return mul_1((data_t)(-COS_8,-SIN_8),a);'}
        funcs_mulpxpy_16 = [Function(k, {'a': Scalar(data_t)}, v, return_type=data_t) for k, v in mul_pxpy_dict.items()]
        funcs_mulpxpy_16[0].defines = {'COS_8': np.cos(np.pi / 8), 'SIN_8': np.sin(np.pi / 8)}
        func_fft_radix_16 = Function('fft_radix_16',
                                     {'v': Private(data_t)},
                                     """
                data_t u[16];
                for (int m=0;m<16;m++) u[m] = v[m];
                // 8x in-place DFT2 and twiddle (1)
                DFT2_TWIDDLE(u[0],u[8],mul_p0q8);
                DFT2_TWIDDLE(u[1],u[9],mul_p1q8);
                DFT2_TWIDDLE(u[2],u[10],mul_p2q8);
                DFT2_TWIDDLE(u[3],u[11],mul_p3q8);
                DFT2_TWIDDLE(u[4],u[12],mul_p4q8);
                DFT2_TWIDDLE(u[5],u[13],mul_p5q8);
                DFT2_TWIDDLE(u[6],u[14],mul_p6q8);
                DFT2_TWIDDLE(u[7],u[15],mul_p7q8);
                
                // 8x in-place DFT2 and twiddle (2)
                DFT2_TWIDDLE(u[0],u[4],mul_p0q4);
                DFT2_TWIDDLE(u[1],u[5],mul_p1q4);
                DFT2_TWIDDLE(u[2],u[6],mul_p2q4);
                DFT2_TWIDDLE(u[3],u[7],mul_p3q4);
                DFT2_TWIDDLE(u[8],u[12],mul_p0q4);
                DFT2_TWIDDLE(u[9],u[13],mul_p1q4);
                DFT2_TWIDDLE(u[10],u[14],mul_p2q4);
                DFT2_TWIDDLE(u[11],u[15],mul_p3q4);
                
                // 8x in-place DFT2 and twiddle (3)
                DFT2_TWIDDLE(u[0],u[2],mul_p0q2);
                DFT2_TWIDDLE(u[1],u[3],mul_p1q2);
                DFT2_TWIDDLE(u[4],u[6],mul_p0q2);
                DFT2_TWIDDLE(u[5],u[7],mul_p1q2);
                DFT2_TWIDDLE(u[8],u[10],mul_p0q2);
                DFT2_TWIDDLE(u[9],u[11],mul_p1q2);
                DFT2_TWIDDLE(u[12],u[14],mul_p0q2);
                DFT2_TWIDDLE(u[13],u[15],mul_p1q2);
                
                // 8x DFT2 and store (reverse binary permutation)
                v[0]  = u[0]  + u[1];
                v[1]  = u[8]  + u[9];
                v[2]  = u[4]  + u[5];
                v[3]  = u[12] + u[13];
                v[4]  = u[2]  + u[3];
                v[5]  = u[10] + u[11];
                v[6]  = u[6]  + u[7];
                v[7]  = u[14] + u[15];
                v[8]  = u[0]  - u[1];
                v[9]  = u[8]  - u[9];
                v[10] = u[4]  - u[5];
                v[11] = u[12] - u[13];
                v[12] = u[2]  - u[3];
                v[13] = u[10] - u[11];
                v[14] = u[6]  - u[7];
                v[15] = u[14] - u[15];
                                    """,
                                     defines={'DFT2_TWIDDLE(a,b,t)': '{ data_t tmp = t(a-b); a += b; b = tmp; }'})
        functions_radix_16 = [func_mul1] + funcs_mulpxpy_16 + [func_fft_radix_16]

        funcs_radix = [func_fft_radix_2] + funcs_radix_4 + funcs_radix_8  + functions_radix_16
        self.cl_program = Program(funcs_radix + [func_exchange, func_expand, func_fft_iteration],
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


def in_data_complex_radix(radix=2, exponent=10):
    np.random.seed(0)
    in_data_np = np.random.random((50, 2 ** 16,)).astype(Types.cdouble)
    in_data_np = np.random.random((50, radix ** exponent,)).astype(Types.cdouble)
    return in_data_np, radix


@mark.parametrize("in_data", [
    in_data_complex_radix(radix=16, exponent=3),
    (np.random.random((2, 2 ** 3,)).astype(Types.cdouble), 2),
    in_data_complex_radix(radix=2, exponent=10),
    in_data_complex_radix(radix=4, exponent=5),
    in_data_complex_radix(radix=8, exponent=4),
    # ,
    # in_data_complex_radix(radix=4, exponent=5),
    # in_data_complex_radix(radix=8, exponent=4),
])
def test_fft(in_data):
    import numpy as np
    in_data_np = in_data[0]
    radix = in_data[1]
    attempts = 4

    for i in range(attempts):
        t_np = time.time()

    ts = []
    for i in range(attempts):
        t1 = time.time()
        fft_in_data_np = np.fft.fft(in_data_np, axis=-1)
        t2 = time.time()
        ts.append(t2 - t1)
    t_np = min(ts)

    thread = Thread(b_profiling_enable=False)
    in_data_cl = to_device(thread.queue, in_data_np)

    fft_cl = Fft(in_data_cl, b_python=False, radix=radix)

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
        set_b_use_existing_file_for_emulation(False)
        fft_cl_py = Fft(in_data_cl, b_python=True, radix=radix)

        fft_in_data_cl_py = fft_cl_py()
        a = fft_in_data_cl_py.get().view(Types.cdouble)
        b = fft_in_data_cl.get().view(Types.cdouble)
        assert np.allclose(a, b)
        assert np.allclose(fft_in_data_np.view(Types.double), fft_in_data_cl_py.get().view(Types.double))

    # import matplotlib.pyplot as plt
    # plt.plot(fft_in_data_np.flatten())
    # plt.plot(fft_in_data_cl_emulation.get().flatten())
    # plt.show()
    assert np.allclose(fft_in_data_np, fft_in_data_cl.get())

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
    t = 0
