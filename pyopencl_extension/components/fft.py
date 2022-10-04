__author__ = "P.Mohr"
__copyright__ = "10.02.2021, P.Mohr"
__version__ = "1.0"
__email__ = "philipp.mohr@tuhh.de"
__doc__ = """This module contains the FFT operation. Radix 2 and """

from dataclasses import dataclass
from typing import List

import numpy as np

from pyopencl_extension import Function, Kernel, Types, Scalar, \
    Global, Program, Private, Local, zeros_like, Array, zeros


# Definition of radix 2 - 16 functions:

def get_funcs_radix2(data_t):
    func_fft_radix_2 = Function('fft_radix_2',
                                {'v': Private(data_t)},
                                """
                           data_t v0= v[0];
                           v[0] = v0 + v[1];
                           v[1] = v0 - v[1];
                           """)
    return [], func_fft_radix_2


mul_pxpy_dict4 = {'mul_p1q2': (mul_p1q2 := 'return (data_t)(a.y,-a.x);')}


def get_funcs_radix4(data_t):
    # mul_pxqy(a) returns a*exp(-j * PI * p / q) where p=x and q=y
    funcs_mulpxpy_4 = [Function(k, {'a': Scalar(data_t)}, v, returns=data_t) for k, v in mul_pxpy_dict4.items()]
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
    return funcs_mulpxpy_4, func_fft_radix_4


mul_pxpy_dict8 = {**{'mul_p0q2': (mul_p0q2 := 'return a;'),
                     'mul_p0q4': mul_p0q2,
                     'mul_p2q4': (mul_p2q4 := mul_p1q2),
                     'mul_p1q4': (mul_p1q4 := 'return (data_t)(SQRT_1_2)*(data_t)(a.x+a.y,-a.x+a.y);'),
                     'mul_p3q4': (mul_p3q4 := 'return (data_t)(SQRT_1_2)*(data_t)(-a.x+a.y,-a.x-a.y);')},
                  **mul_pxpy_dict4}


def get_funcs_radix8(data_t):
    funcs_mulpxpy_8 = [Function(k, {'a': Scalar(data_t)}, v, returns=data_t) for k, v in mul_pxpy_dict8.items()]
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
    return funcs_mulpxpy_8, func_fft_radix_8


mul_pxpy_dict16 = {**{'mul_p0q8 ': mul_p0q2,
                      'mul_p1q8': 'return mul_1((data_t)(COS_8,-SIN_8),a);',
                      'mul_p2q8': mul_p1q4,
                      'mul_p3q8': 'return mul_1((data_t)(SIN_8,-COS_8),a);',
                      'mul_p4q8 ': mul_p2q4,
                      'mul_p5q8': 'return mul_1((data_t)(-SIN_8,-COS_8),a);',
                      'mul_p6q8': mul_p3q4,
                      'mul_p7q8': 'return mul_1((data_t)(-COS_8,-SIN_8),a);'}, **mul_pxpy_dict8}


def get_funcs_radix16(data_t):
    func_mul1 = Function('mul_1',
                         {'a': Scalar(data_t), 'b': Scalar(data_t)},
                         'data_t x; x.even = MUL_RE(a,b); x.odd = MUL_IM(a,b); return x;', returns=data_t,
                         defines={'MUL_RE(a,b)': '(a.even*b.even - a.odd*b.odd)',
                                  'MUL_IM(a,b)': '(a.even*b.odd + a.odd*b.even)'})

    funcs_mulpxpy_16 = [Function(k, {'a': Scalar(data_t)}, v, returns=data_t) for k, v in mul_pxpy_dict16.items()]
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
    funcs_helpers = [func_mul1] + funcs_mulpxpy_16
    return funcs_helpers, func_fft_radix_16


dict_radix_funcs = {
    2: get_funcs_radix2,
    4: get_funcs_radix4,
    8: get_funcs_radix8,
    16: get_funcs_radix16,
}


def get_funcs_radixes(radixes, data_t: np.dtype):
    # makes sure radixes are in order, since higher radix functions use function loaded for lower radixes.
    radixes = sorted(radixes)
    defines_radices = {'SQRT_1_2': np.cos(np.pi / 4)}
    funcs = [dict_radix_funcs[r](data_t) for r in radixes]
    # filter out duplicate helpers
    helpers = list({helper.name: helper for helpers, _ in funcs for helper in helpers}.values())
    func_radices = [func for _, func in funcs]
    return helpers + func_radices, defines_radices


def test_get_funcs_radixes():
    res = get_funcs_radixes([8, 16], Types.float2)
    assert res[0][-2].name == 'fft_radix_8'
    assert res[0][-1].name == 'fft_radix_16'


@dataclass
class FftStageBuilder:
    radix: int
    data_t: np.dtype
    real_t: np.dtype
    fft_size: int
    global_mem_cacheline_size: int
    size_data_in_first_iteration: int
    iteration_max: int
    b_local_memory: bool = False

    def get_funcs_for_all_stages(self, radixes: List[int]):
        conf = self
        typedefs = {'data_t': (data_t := conf.data_t),
                    'real_t': (real_t := conf.real_t)}

        R = conf.radix
        defines = {'R': R,
                   'N': (N := conf.fft_size),
                   # work group size:
                   'T': (T := min(int(N / R), conf.global_mem_cacheline_size)),
                   'CW': (CW := conf.global_mem_cacheline_size),  # Coalescing width,
                   'N_INPUT': conf.size_data_in_first_iteration,  # in_buffer length which might not be power of two
                   'ITERATION_MAX': self.iteration_max,  # in_buffer length which might not be power of two
                   }

        funcs_radix, defines_radices = get_funcs_radixes(radixes, data_t)
        defines = {**defines, **defines_radices}

        # [1]: The expand() function can be thought of as inserting a dimension of length N2 after the first
        # dimension of length N1 in a linearized index.
        func_expand = Function('expand',
                               {'idxL': Scalar(Types.int),
                                'N1': Scalar(Types.int),
                                'N2': Scalar(Types.int)},
                               """
                                 return (int)(idxL/N1)*N1*N2 + (idxL%N1);
                                 """,
                               returns=Types.int)
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
                                 defines={'STRIDE': 1})
        func_v_from_data0 = Function('v_from_data0',
                                     {'v': Private(self.data_t),
                                      'data0': Local(self.data_t) if self.b_local_memory else Global(self.data_t,
                                                                                                     read_only=True),
                                      'idxS': Scalar(Types.int),
                                      'r': Scalar(Types.int),
                                      'direction': Scalar(Types.int),
                                      'iteration': Scalar(Types.int)},
                                     """
                 if(iteration ==0){
                     if(idxS<N_INPUT){
                        v[r]=data0[(int)(idxS)];
                     }else{
                        v[r]=(data_t)(0.0, 0.0);
                     }
                     if(direction == -1){
                        v[r] = (data_t)(v[r].s1, v[r].s0);
                     }
                 }else{
                    v[r] = data0[(int)(idxS)];
                 }
                 """)
        func_data_1_from_v = Function('data_1_from_v',
                                      {'v': Private(self.data_t),
                                       'data1': Local(self.data_t) if self.b_local_memory else Global(self.data_t),
                                       'idxD': Scalar(Types.int),
                                       'r': Scalar(Types.int),
                                       'direction': Scalar(Types.int),
                                       'iteration': Scalar(Types.int)},
                                      """
                 if(iteration==ITERATION_MAX && direction == -1){
                    data1[idxD] = (data_t)((float)(1.0/N)) * (data_t)(v[r].s1, v[r].s0);
                 }else{
                    data1[idxD] = v[r];
                 }
                 """)
        shared_memory_init = 'local real_t shared[(int)(2*T*R)];//2*Ns];//T*R];'
        return funcs_radix + [func_exchange, func_expand, func_v_from_data0, func_data_1_from_v], \
               typedefs, defines, shared_memory_init

    def get_fft_stage_func(self):
        replacements = {'fft_radix_R': f'fft_radix_{self.radix}'}
        func_fft_iteration = Function('fft_stage', {
            # thread index: T radix R fft ops are done in parallel
            't': Scalar(Types.int),
            # thread block index: identifies block of N/(R*T) blocks.
            'b': Scalar(Types.int),
            'Ns': Scalar(Types.int),
            'data0': Local(self.data_t) if self.b_local_memory else Global(self.data_t, read_only=True),
            'data1': Local(self.data_t) if self.b_local_memory else Global(self.data_t),
            'shared': Local(self.real_t),
            'direction': Scalar(Types.int),
            'iteration': Scalar(Types.int)},
                                      """
        int j = b*T + t; // as proposed in [1, p.3, text section A]
        private data_t v[R];        
        real_t angle = -2*PI*(j % Ns)/(Ns*R);
        for(int r=0; r<R; r++){
            int idxS = j + r* N/R; // idxS=idxSource
            v_from_data0(v, data0, idxS, r, direction,iteration);
            real_t cos_angle = cos(r*angle);
            real_t sin_angle = sin(r*angle);
            v[r] = (data_t)(v[r].s0*cos_angle-v[r].s1 *sin_angle,
                            v[r].s0*sin_angle+v[r].s1 *cos_angle);
        }
        ${fft_radix_R}(v);

        if(Ns>=CW){ // todo: remove condition and store as separate kernel
            // changed line 27 of [1, Figure 2] to work with global dimensions of this class:
            int offset = expand(j, Ns, R); 
            for(int r=0; r<R; r++){
                int idxD = offset + r*Ns; // idxD=idxDestination
                data_1_from_v(v, data1, idxD, r, direction, iteration);
            }
        }else{ // According to [1, p.4], such that global memory writes are coalesced
            // !! mistake in [1] where *Ns is missing: idxD = (int)(t/Ns)*R + (t%Ns); !!
            int idxD = (int)(t/Ns)*Ns*R + (t%Ns);
            exchange( v, idxD, Ns, t, T, shared);
            int offset = b*R*T+ t; 
            for( int r=0; r<R; r++ ){
                idxD = offset + r*T;
                data_1_from_v(v, data1, idxD, r, direction, iteration);
            }  
        }
                   """,
                                      replacements=replacements)
        return func_fft_iteration


class FftBase:
    """
    Implementation of reference algorithm [1, Fig. 2] (where only radix 2 kernel is provided)
    A reference for higher radix kernels can be found in [2]

    [1] N. K. Govindaraju, B. Lloyd, Y. Dotsenko, B. Smith, und J. Manferdelli, „High performance discrete Fourier
    transforms on graphics processors“, in 2008 SC - International Conference for High Performance Computing,
    Networking, Storage and Analysis, Austin, TX, Nov. 2008, S. 1–12, doi: 10.1109/SC.2008.5213922.

    [2] http://www.bealto.com/gpu-fft_opencl-2.html
    """

    def __call__(self, *args, **kwargs):
        data_in = self._in_buffer
        data_out = self._data_out
        Ns = 1
        for iteration, radix in enumerate(self.schedule):
            self.kernels_for_iteration[iteration](Ns=Ns,
                                                  data_in=data_in,
                                                  data_out=data_out,
                                                  direction=self.direction,
                                                  iteration=iteration)
            Ns = Ns * radix
            if iteration == 0:
                data_in = self._data_in  # map data_in to internal buffer with correct power of two size
            data_in, data_out = data_out, data_in  # swap
        return data_in.view(self.in_buffer.dtype).reshape(self._out_shape)

    @staticmethod
    def get_mixed_radix_schedule(N, radixes):
        _radixes = [r for r in radixes if r <= int(N / 2)]  # if N is very short do not use large radix ffts
        schedule = []
        n = N
        while n != 1:
            _radix = radixes[np.argmax([(n / i).is_integer() for i in radixes])]
            schedule.append(_radix)
            n /= _radix
            if not n.is_integer():
                raise ValueError('Something went wrong. n must be integer')
        return schedule  # [2 for _ in range(int(log(N, 2)))]

    @staticmethod
    def _get_kernel(stage_builder: FftStageBuilder, iteration, radix, data_in, data_out, emulate=False):
        if iteration == 0:  # data0 is in_buffer, whose length might not be power of two
            offset_data_in = 'get_global_id(0)*N_INPUT'
        else:
            offset_data_in = 'get_global_id(0)*N'
        stage_builder.radix = radix
        funcs, type_defs, d, shared_mem_fft_stage = stage_builder.get_funcs_for_all_stages([radix])
        funcs.append(stage_builder.get_fft_stage_func())
        d['M'] = (M := data_in.shape[0])  # number of FFTs to perform simultaneously
        knl_gpu_fft = Kernel('gpu_fft',
                             {'Ns': Scalar(Types.int),
                              # In each iteration, the algorithm can be thought of combining the radix R FFTs on
                              # subsequences of length Ns into the FFT of a new sequence of length RNs by
                              # performing an FFT of length R on the corresponding elements of the subsequences.
                              'data_in': Global(data_in),
                              'data_out': Global(data_out),
                              'direction': Scalar(Types.int),
                              'iteration': Scalar(Types.int)},
                             """
                 ${shared_mem_fft_stage}
                 int t = get_local_id(2); // thread index 
                 int b = get_global_id(1); // thread block index
                 fft_stage(t, b, Ns, 
                           data_in  +${offset_data_in},
                           data_out +get_global_id(0)*N,
                           shared,direction, iteration);""",
                             replacements={'offset_data_in': offset_data_in,
                                           'shared_mem_fft_stage': shared_mem_fft_stage},
                             global_size=(d['M'], max(1, int(d['N'] / (d['R'] * d['T']))), d['T']),
                             local_size=(1, 1, d['T']))
        program = Program(funcs,
                          [knl_gpu_fft], defines=d, type_defs=type_defs
                          ).compile(context=data_in.context, emulate=emulate,
                                    file=Program.get_default_dir_pycl_kernels().joinpath(
                                        f'fft_{iteration}_{stage_builder.radix}'))
        return program.gpu_fft

    def __init__(self, in_buffer: Array, b_inverse_fft=False, emulate=False):
        # just for debugging:
        self.emulate = emulate
        self.direction = -1 if b_inverse_fft else +1

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

        data_in = zeros((M, N), data_t)
        data_out = zeros_like(data_in)

        radixes = [16, 8, 4, 2]  # available radix ffts
        self.schedule = self.get_mixed_radix_schedule(N, radixes)
        fft_stage_builder = FftStageBuilder(
            radix=2,  # might change according to schedule
            data_t=data_t, real_t=real_t,
            fft_size=N,
            global_mem_cacheline_size=in_buffer.context.devices[0].global_mem_cacheline_size,
            size_data_in_first_iteration=in_buffer.shape[1], iteration_max=len(self.schedule) - 1)
        self.kernels_for_iteration = [
            self._get_kernel(fft_stage_builder, iteration=_iter, radix=_radix, data_in=data_in, data_out=data_out,
                             emulate=emulate) for _iter, _radix
            in enumerate(self.schedule)]
        self.in_buffer = in_buffer
        self._in_buffer = in_buffer.view(data_t)
        self._data_in = data_in
        self._data_out = data_out

        # optimize later: there may be some redundant kernels in self.kernels_for_iteration
        # max_iteration = len(self.schedule) - 1
        # # individual kernels for: iteration 0 and each different radix in other iterations
        # iter_unique_knl = [0] + (1 + np.sort(np.unique(self.schedule[1:], return_index=True)[1])).tolist() + \
        #                   [max_iteration + 1]
        # _parts = [[_iter, self.schedule[_iter]] for _iter in iter_unique_knl[:-1]]
        # _knl_ids = np.concatenate([np.ones(iter_unique_knl[i + 1] - iter_unique_knl[i], int) * i
        #                            for i, _ in enumerate(iter_unique_knl[:-1])])
        # if b_inverse_fft:  # final iteration has individual kernel
        #     pass


class Fft(FftBase):
    def __init__(self, in_buffer: Array, emulate=False):
        super().__init__(in_buffer, emulate=emulate)


class IFft(FftBase):
    def __init__(self, in_buffer: Array, emulate=False):
        super().__init__(in_buffer, b_inverse_fft=True, emulate=emulate)
