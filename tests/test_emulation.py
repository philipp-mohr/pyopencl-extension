from pathlib import Path

import numpy as np
from pyopencl import Program
from pyopencl.array import zeros, zeros_like, to_device
from pyopencl_extension import emulation
from pyopencl_extension import unparse_c_code_to_python, create_py_file_and_load_module, ClTypes, ClKernel, \
    KnlArgBuffer, ClFunction, ClProgram, ArgPrivate
from pytest import mark

path_py_cl = Path(__file__).parent.joinpath('py_cl_kernels')

rendered_template = """
#define PI 3.14159265358979323846
#define MUL(x,y) (x*y)
#define ADD(x,y) (x+y)
#define SUB(x,y) (x-y)
#define ABS(x) (fabs(x))
#define RMUL(x,y) (x*y)
#define NEW(x) (x)


#define SOME_CONSTANT 6

typedef short data_t;

float plus_one(__global  short *buffer,
               int idx)
{
    return buffer[idx]+5;
}

__kernel void some_operation(__global  short *buff,
                             short number)
{
    data_t factor = convert_short(1.3);
    buff[get_global_id(0)] = plus_one(buff, get_global_id(0)) + SOME_CONSTANT*factor;
}
"""


def test_debug_kernel(cl_init):
    buff_cl = zeros(cl_init.queue, (10, 1), ClTypes.short)
    # compute result with opencl
    program = Program(cl_init.context, str(rendered_template)).build()
    some_operation = program.all_kernels()[0]
    # some_operation(init.queue, buff_cl.shape, None,
    #                buff_cl.data, np.dtype(ClTypes.short).type(3))
    # alternative:
    some_operation.set_scalar_arg_dtypes([None, ClTypes.short])
    some_operation(cl_init.queue, buff_cl.shape, None,
                   buff_cl.data, 3)

    buff_py = zeros(cl_init.queue, (10, 1), ClTypes.short)
    code_py = unparse_c_code_to_python(rendered_template)
    module = create_py_file_and_load_module(code_py, str(path_py_cl.joinpath('debug_component')))

    module.some_operation(buff_py.shape, None,
                          buff_py,
                          np.dtype(ClTypes.short).type(3))

    assert np.all(buff_py.get() == 11)
    assert np.all(buff_py.get() == buff_cl.get())


rendered_template_cplx_float = """
#include <pyopencl-complex.h>
#define TP_ROOT cfloat
#define PI 3.14159265359f

#define MUL cfloat_mul
#define ADD cfloat_add
#define SUB cfloat_sub
#define ABS cfloat_abs
#define RMUL cfloat_rmul
#define NEW cfloat_new

__kernel void some_operation(__global  cfloat_t *buff,
                             cfloat_t number)
{
    buff[get_global_id(0)] = number;
}
"""


def test_debug_kernel_with_complex_numbers(cl_init):
    buff_cl = zeros(cl_init.queue, (10, 1), ClTypes.cfloat)
    # compute result with opencl
    program = Program(cl_init.context, str(rendered_template_cplx_float)).build()
    some_operation = program.all_kernels()[0]
    # some_operation.set_scalar_arg_dtypes([None, ClTypes.cfloat])
    some_operation(cl_init.queue, buff_cl.shape, None,
                   buff_cl.data, np.dtype(ClTypes.cfloat).type(3 + 1j))

    buff_py = zeros(cl_init.queue, (10, 1), ClTypes.cfloat)
    code_py = unparse_c_code_to_python(rendered_template_cplx_float)
    module = create_py_file_and_load_module(code_py, str(path_py_cl.joinpath('debug_component_complex_numbers')))

    module.some_operation(buff_py.shape, None,
                          buff_py, np.dtype(ClTypes.cfloat).type(3 + 1j))

    assert np.all(buff_py.get() == buff_cl.get())


rendered_template_cplx_double = """
#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define PYOPENCL_DEFINE_CDOUBLE
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#define PYOPENCL_DEFINE_CDOUBLE
#endif

#include <pyopencl-complex.h>
#define TP_ROOT cfloat
#define PI 3.14159265359f

__kernel void some_operation(__global  cdouble_t *buff,
                             cdouble_t number)
{
    buff[get_global_id(0)] = number;
}
"""


def test_debug_kernel_with_complex_numbers_double(cl_init):
    cplx_t = ClTypes.cdouble
    buff_cl = zeros(cl_init.queue, (10, 1), cplx_t)
    # compute result with opencl
    program = Program(cl_init.context, str(rendered_template_cplx_double)).build()
    some_operation = program.all_kernels()[0]
    # some_operation.set_scalar_arg_dtypes([None, ClTypes.cfloat])
    some_operation(cl_init.queue, buff_cl.shape, None,
                   buff_cl.data, np.dtype(cplx_t).type(3 + 1j))

    buff_py = zeros(cl_init.queue, (10, 1), cplx_t)
    code_py = unparse_c_code_to_python(rendered_template_cplx_float)
    module = create_py_file_and_load_module(code_py, str(path_py_cl.joinpath('debug_component_complex_numbers_double')))

    module.some_operation(buff_py.shape, None,
                          buff_py, np.dtype(cplx_t).type(3 + 1j))

    assert np.all(buff_py.get() == buff_cl.get())


def test_debug_try_python_conversion_if_c_compilation_fails():
    pass


# from here on all tests depend on framework
def test_debug_c_code_with_unary_increment_operation_inside_of_array(cl_init):
    buff_cl = zeros(cl_init.queue, (6, 1), ClTypes.short)
    knl = ClKernel('knl',
                   {'buff': KnlArgBuffer(buff_cl)},
                   """
        int number = -1;
        number++;
        buff[number++] = 1;
        buff[number] = 2;
        number = 0;
        buff[2+ number--] = 3;
        buff[3+ ++number] = 4;
        buff[5 + --number] = 5;
        int count = 0;
        for(int i=1; i<3; i++){
            count = count + i;
        }        
        buff[5] = count;
    """,
                   global_size=(1,))
    compiled_cl = knl.compile(cl_init, b_python=False)
    compiled_cl(buff=buff_cl)
    buff_py = zeros(cl_init.queue, (6, 1), ClTypes.short)
    compiled_py = knl.compile(cl_init, b_python=True, file=Path(__file__).parent.joinpath('py_cl_kernels/knl'))
    compiled_py(buff=buff_py)
    assert np.all(buff_py.get() == buff_cl.get())


def test_access_complex_variable(cl_init):
    buff = np.array([0.5]).astype(ClTypes.cfloat)
    buff_in = to_device(cl_init.queue, buff)
    buff_out = zeros_like(buff_in)
    knl = ClKernel('knl',
                   {'inp': KnlArgBuffer(buff_in),
                    'out': KnlArgBuffer(buff_out)},
                   """
        out[get_global_id(0)].real = inp[get_global_id(0)].real; 
    """,
                   global_size=(1,))
    compiled_cl = knl.compile(cl_init, b_python=False, file=Path(__file__).parent.joinpath('py_cl_kernels/knl'))
    compiled_cl()
    buff_out_py = zeros_like(buff_in)
    compiled_py = knl.compile(cl_init, b_python=True, file=Path(__file__).parent.joinpath('py_cl_kernels/knl'))
    # out[0] = complex64(inp[0].real+out[0].imag*1j) instead of out[0].real=inp[0].real
    compiled_py(out=buff_out_py)
    assert np.all(buff_out.get() == buff_out_py.get())


def test_debug_kernel_with_barriers(cl_init):
    buff = np.zeros(shape=(2, 4)).astype(ClTypes.int)
    mem_buffer = to_device(cl_init.queue, buff)
    knl = ClKernel('knl',
                   {'mem_glob': KnlArgBuffer(mem_buffer)},
                   """
        __local int mem[2];
        mem[0]=0;
        mem[1]=0;
        mem[get_local_id(1)] = get_local_id(1);
        barrier(CLK_LOCAL_MEM_FENCE);
        mem[get_local_id(1)] = mem[1];
        //barrier(CLK_GLOBAL_MEM_FENCE);
        mem_glob[get_global_id(0)*get_global_size(1)+get_global_id(1)] = mem[get_local_id(1)];
    """,
                   global_size=(2, 4),
                   local_size=(1, 2))
    compiled_cl = knl.compile(cl_init, b_python=False, file=Path(__file__).parent.joinpath('py_cl_kernels/knl'))
    compiled_cl()
    mem_buffer_py = zeros_like(mem_buffer)
    compiled_py = knl.compile(cl_init, b_python=True, file=Path(__file__).parent.joinpath('py_cl_kernels/knl'))
    # out[0] = complex64(inp[0].real+out[0].imag*1j) instead of out[0].real=inp[0].real
    compiled_py(mem_glob=mem_buffer_py)
    assert np.all(mem_buffer.get() == mem_buffer_py.get())


def test_number_overflow(cl_init):
    inp1 = np.array([127, 10, -128]).astype(ClTypes.char)
    inp2 = np.array([127, 10, -128]).astype(ClTypes.char)
    knl = ClKernel('knl',
                   {
                       'inp1': inp1,
                       'inp2': inp2,
                       'out1': np.zeros_like(inp1, dtype=ClTypes.char),
                       'out2': np.zeros_like(inp1, dtype=ClTypes.char)},
                   """
        char a = 0;
        a = add_sat(inp1[get_global_id(0)], inp2[get_global_id(0)]);
        char b = 0;
        b = inp1[get_global_id(0)] + inp2[get_global_id(0)];
        out1[get_global_id(0)] = a; 
        out2[get_global_id(0)] = b; 
    """,
                   global_size=inp1.shape)
    knl_cl = knl.compile(cl_init)
    knl_py = knl.compile(cl_init, b_python=True)
    knl_cl()
    res_cl = knl_cl.out1.get(), knl_cl.out2.get()
    knl_py()
    res_py = knl_cl.out1.get(), knl_cl.out2.get()
    assert np.all(res_cl[0] == res_py[0]) and np.all(res_cl[1] == res_py[1])


# implement pointer arithmetics with pointer wrapper class for every variable
def test_pointer_arithmetics(cl_init):
    # todo:
    # Problem: abstract syntax tree does not distiguish if an identifier is a pointer or a variable.
    # E.g. if incrementing the pointer to an array a (a=a+1) in Python this would increment all values in
    # the underlying array. However if
    data = np.array([0]).astype(ClTypes.char)
    knl = ClKernel('knl_pointer_arithmetics',
                   {'data': data},
                   """
        char a[5]={0};
        a[0] = a[0] + 1;
        a[1] = 1;
        char* b = a + 1;
        data[0] = b[0];
    """,
                   global_size=data.shape)
    knl_cl = knl.compile(cl_init)
    emulation.set_b_use_existing_file_for_emulation(False)
    knl_py = knl.compile(cl_init, b_python=True)
    knl_cl()
    res_cl = knl_cl.data.get()
    knl_py()
    res_py = knl_cl.data.get()
    assert np.all(res_cl[0] == res_py[0])


def test_pointer_increment(
        cl_init):  # todo use https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ctypes.html
    data = np.array([0]).astype(ClTypes.char)
    func = ClFunction('func',
                      {'data': ArgPrivate(data.dtype)},
                      """
        return data[0];
    """, return_type=data.dtype)
    knl = ClKernel('knl_pointer_arithmetics',
                   {'data': data},
                   """
        private char a[5] = {0};
        a[3] = 5;
        data[0] = func(a+3);
    """,
                   global_size=data.shape)
    prog = ClProgram(functions=[func], kernels=[knl])
    knl_cl = prog.compile(cl_init).knl_pointer_arithmetics
    knl_py = prog.compile(cl_init, b_python=True).knl_pointer_arithmetics
    knl_cl()
    res_cl = knl_cl.data.get()
    knl_py()
    res_py = knl_cl.data.get()
    assert np.all(res_cl[0] == res_py[0])


def test_bit_shift(cl_init):  # todo use https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ctypes.html
    data = np.array([0, 0, 0, 0]).astype(ClTypes.char)
    knl = ClKernel('knl_bit_packing',
                   {'data': data},
                   """
        uchar a = 5;
        uchar b = 3;
        uchar c = (a << 4) | b;
        data[0] = (c & 0xF0) >> 4;
        data[1] = c & (0x0F);
    """,
                   global_size=data.shape)
    prog = ClProgram(kernels=[knl])
    knl_cl = prog.compile(cl_init).knl_bit_packing
    knl_py = prog.compile(cl_init, b_python=True).knl_bit_packing
    knl_cl()
    cl_init.queue.finish()
    res_cl = knl_cl.data.get()
    knl_py()
    res_py = knl_cl.data.get()
    assert np.all(res_cl == res_py)


@mark.parametrize("header", ['for(int i=10; i >=0; i--)',
                             'for(int i=10; i !=0; i--)',
                             'for(int i=10; i >0; i--)',
                             'for(int i=0; i <=10; i++)',
                             'for(int i=0; i !=10; i++)',
                             'for(int i=0; i <10; i++)'])
def test_for_loop(cl_init, header):
    def eval_code(b_python=False):
        data = to_device(cl_init.queue, np.array([0]).astype(ClTypes.char))
        knl = ClKernel('knl_test_for_loop',
                       {'data': KnlArgBuffer(data)},
                       """
            ${header}{
                data[0]+=i;
            }
        """,
                       replacements={'header': header},
                       global_size=data.shape).compile(cl_init, b_python=b_python)
        knl()
        cl_init.queue.finish()
        res = knl.data.get()
        return res

    assert np.all(eval_code(False) == eval_code(True))


def test_vector_types(cl_init):  # todo use https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ctypes.html
    data = np.zeros((10,)).astype(ClTypes.char2)
    knl = ClKernel('knl_vector_types',
                   {'data': data},
                   """
        char2 a = (char2)(4,2);
        char2 b = (char2)(1,2);
        data[0] = a;
        data[1] = b;
        data[2] = a + b;
        data[3] = a * b;
        data[4] = a - b;
        data[5] = a / b;
    """,
                   global_size=data.shape)
    knl_cl = knl.compile(cl_init)
    knl_py = knl.compile(cl_init, b_python=True)
    knl_cl()
    cl_init.queue.finish()
    res_cl = knl_cl.data.get()
    knl_py()
    res_py = knl_py.data.get()
    assert np.all(res_cl == res_py)