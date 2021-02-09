from pyopencl.array import zeros_like, Array

from pyopencl_extension import ClFunction, ClKernel, KnlArgScalar, ClTypes, KnlArgBuffer, to_device, ArgScalar, \
    ArgBuffer, ClProgram, ClInit, ArgPrivate
from pyopencl_extension.components.copy_array_region import CopyArrayRegion, Slice


class Fft:
    def __call__(self, *args, **kwargs):
        self._copy_in_buffer()
        data_in = self._data_in
        data_out = self._data_out
        for Ns in range(1, self.N, self.R):
            self.cl_program.gpu_fft(Ns=Ns,
                                    data_in=data_in,
                                    data_out=data_out)
            data_in, data_out = data_out, data_in  # swap
        return data_in

    def __init__(self, in_buffer: Array):
        if in_buffer.dtype == ClTypes.cfloat:
            data_t = ClTypes.float2
            real_t = ClTypes.double
        elif in_buffer.dtype == ClTypes.cdouble:
            data_t = ClTypes.double2
            real_t = ClTypes.double
        else:
            raise ValueError()
        self._copy_in_buffer = CopyArrayRegion(in_buffer=in_buffer)
        data_in = self._copy_in_buffer.out_buffer.view(data_t)
        data_out = zeros_like(data_in)
        typedefs = {'data_t': data_t,
                    'real_t': real_t}
        defines = {'R': (R := 2),  # currently only Radix 2 fft is supported
                   'N': (N := in_buffer.shape[0])}
        knl_gpu_fft = ClKernel('gpu_fft',
                               {'Ns': KnlArgScalar(ClTypes.int),
                                'data_in': KnlArgBuffer(data_in),
                                'data_out': KnlArgBuffer(data_out)},
                               """
                 int b = get_group_id(0); int t = get_local_id(0);
                 
                 int j = b*N + t;
                 fft_iteration(j, Ns, data_in, data_out);
                 """,
                               global_size=(int(N / R),),  # The number of threads used for GPU_FFT() is N/R
                               local_size=(int(N / R),))
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
            v[r] = data0[idxS + r* N/R];
            v[r] *= (cos(r*angle), sin(r*angle)); //todo how to assign vector dtype?
        }
        ${fft_radix_R}(v);
        
        int idxD = expand(j, Ns, R);
        for(int r=0; r<R; r++)
            data1[idxD + r*Ns] = v[r];        
                   """,
                                        replacements={
                                            'fft_radix_R': f'fft_radix_{R}'
                                        })
        func_fft_radix_2 = ClFunction('fft_radix_2',
                                      {'v': ArgPrivate(data_t)},
                                      """
                               data_t v0 = v[0];
                               v[0] = v0 + v[1];
                               v[1] = v0 - v[1];
                               """)
        func_expand = ClFunction('expand',
                                 {'idxL': ArgScalar(ClTypes.int),
                                  'N1': ArgScalar(ClTypes.int),
                                  'N2': ArgScalar(ClTypes.int)},
                                 """
                                 return (idxL/N1)*N1*N2 + (idxL%N1);
                                 """,
                                 return_type=ClTypes.int)

        self.cl_program = ClProgram([func_expand, func_fft_radix_2, func_fft_iteration],
                                    [knl_gpu_fft], defines=defines, type_defs=typedefs
                                    ).compile(ClInit.from_buffer(in_buffer),True)
        self.N = N
        self.R = R
        self.in_buffer = in_buffer
        self._data_in = data_in
        self._data_out = data_out


def test_fft():
    import numpy as np
    in_data_np = np.random.random((2 ** 8,)).astype(ClTypes.cdouble)
    fft_in_data_np = np.fft.fft(in_data_np)

    cl_init = ClInit()
    in_data_cl = to_device(cl_init.queue, in_data_np)
    fft_cl = Fft(in_data_cl)
    fft_in_data_cl = fft_cl()

    assert np.allclose(fft_in_data_np, fft_in_data_cl.get())
