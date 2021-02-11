from pathlib import Path

import numpy as np
from pyopencl import Program as pyopencl_program
from pyopencl.array import zeros, zeros_like, to_device
from pyopencl_extension import emulation, set_b_use_existing_file_for_emulation, Local, Global, Global, Local, Scalar
from pyopencl_extension import unparse_c_code_to_python, create_py_file_and_load_module, Types, Kernel, \
    Global, Function, Program, Private
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


def test_debug_kernel(thread):
    buff_cl = zeros(thread.queue, (10, 1), Types.short)
    # compute result with opencl
    program = pyopencl_program(thread.context, str(rendered_template)).build()
    some_operation = program.all_kernels()[0]
    # some_operation(init.queue, buff_cl.shape, None,
    #                buff_cl.data, np.dtype(ClTypes.short).type(3))
    # alternative:
    some_operation.set_scalar_arg_dtypes([None, Types.short])
    some_operation(thread.queue, buff_cl.shape, None,
                   buff_cl.data, 3)

    buff_py = zeros(thread.queue, (10, 1), Types.short)
    code_py = unparse_c_code_to_python(rendered_template)
    module = create_py_file_and_load_module(code_py, str(path_py_cl.joinpath('debug_component')))

    module.some_operation(buff_py.shape, None,
                          buff_py,
                          np.dtype(Types.short).type(3))

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


def test_debug_kernel_with_complex_numbers(thread):
    buff_cl = zeros(thread.queue, (10, 1), Types.cfloat)
    # compute result with opencl
    program = pyopencl_program(thread.context, str(rendered_template_cplx_float)).build()
    some_operation = program.all_kernels()[0]
    # some_operation.set_scalar_arg_dtypes([None, ClTypes.cfloat])
    some_operation(thread.queue, buff_cl.shape, None,
                   buff_cl.data, np.dtype(Types.cfloat).type(3 + 1j))

    buff_py = zeros(thread.queue, (10, 1), Types.cfloat)
    code_py = unparse_c_code_to_python(rendered_template_cplx_float)
    module = create_py_file_and_load_module(code_py, str(path_py_cl.joinpath('debug_component_complex_numbers')))

    module.some_operation(buff_py.shape, None,
                          buff_py, np.dtype(Types.cfloat).type(3 + 1j))

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


def test_debug_kernel_with_complex_numbers_double(thread):
    cplx_t = Types.cdouble
    buff_cl = zeros(thread.queue, (10, 1), cplx_t)
    # compute result with opencl
    program = pyopencl_program(thread.context, str(rendered_template_cplx_double)).build()
    some_operation = program.all_kernels()[0]
    # some_operation.set_scalar_arg_dtypes([None, ClTypes.cfloat])
    some_operation(thread.queue, buff_cl.shape, None,
                   buff_cl.data, np.dtype(cplx_t).type(3 + 1j))

    buff_py = zeros(thread.queue, (10, 1), cplx_t)
    code_py = unparse_c_code_to_python(rendered_template_cplx_float)
    module = create_py_file_and_load_module(code_py, str(path_py_cl.joinpath('debug_component_complex_numbers_double')))

    module.some_operation(buff_py.shape, None,
                          buff_py, np.dtype(cplx_t).type(3 + 1j))

    assert np.all(buff_py.get() == buff_cl.get())


def test_debug_try_python_conversion_if_c_compilation_fails():
    pass


# from here on all tests depend on framework
def test_debug_c_code_with_unary_increment_operation_inside_of_array(thread):
    buff_cl = zeros(thread.queue, (6, 1), Types.short)
    knl = Kernel('knl',
                 {'buff': Global(buff_cl)},
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
    compiled_cl = knl.compile(thread, b_python=False)
    compiled_cl(buff=buff_cl)
    buff_py = zeros(thread.queue, (6, 1), Types.short)
    compiled_py = knl.compile(thread, b_python=True, file=Path(__file__).parent.joinpath('py_cl_kernels/knl'))
    compiled_py(buff=buff_py)
    assert np.all(buff_py.get() == buff_cl.get())


def test_access_complex_variable(thread):
    buff = np.array([0.5]).astype(Types.cfloat)
    buff_in = to_device(thread.queue, buff)
    buff_out = zeros_like(buff_in)
    knl = Kernel('knl',
                 {'inp': Global(buff_in),
                  'out': Global(buff_out)},
                 """
        out[get_global_id(0)].real = inp[get_global_id(0)].real; 
    """,
                 global_size=(1,))
    compiled_cl = knl.compile(thread, b_python=False, file=Path(__file__).parent.joinpath('py_cl_kernels/knl'))
    compiled_cl()
    buff_out_py = zeros_like(buff_in)
    compiled_py = knl.compile(thread, b_python=True, file=Path(__file__).parent.joinpath('py_cl_kernels/knl'))
    # out[0] = complex64(inp[0].real+out[0].imag*1j) instead of out[0].real=inp[0].real
    compiled_py(out=buff_out_py)
    assert np.all(buff_out.get() == buff_out_py.get())


def test_debug_kernel_with_barriers(thread):
    buff = np.zeros(shape=(2, 4)).astype(Types.int)
    mem_buffer = to_device(thread.queue, buff)
    knl = Kernel('knl',
                 {'mem_glob': Global(mem_buffer)},
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
    compiled_cl = knl.compile(thread, b_python=False, file=Path(__file__).parent.joinpath('py_cl_kernels/knl'))
    compiled_cl()
    mem_buffer_py = zeros_like(mem_buffer)
    compiled_py = knl.compile(thread, b_python=True, file=Path(__file__).parent.joinpath('py_cl_kernels/knl'))
    # out[0] = complex64(inp[0].real+out[0].imag*1j) instead of out[0].real=inp[0].real
    compiled_py(mem_glob=mem_buffer_py)
    assert np.all(mem_buffer.get() == mem_buffer_py.get())


def test_number_overflow(thread):
    inp1 = np.array([127, 10, -128]).astype(Types.char)
    inp2 = np.array([127, 10, -128]).astype(Types.char)
    knl = Kernel('knl',
                 {
                     'inp1': inp1,
                     'inp2': inp2,
                     'out1': np.zeros_like(inp1, dtype=Types.char),
                     'out2': np.zeros_like(inp1, dtype=Types.char)},
                 """
        char a = 0;
        a = add_sat(inp1[get_global_id(0)], inp2[get_global_id(0)]);
        char b = 0;
        b = inp1[get_global_id(0)] + inp2[get_global_id(0)];
        out1[get_global_id(0)] = a; 
        out2[get_global_id(0)] = b; 
    """,
                 global_size=inp1.shape)
    knl_cl = knl.compile(thread)
    knl_py = knl.compile(thread, b_python=True)
    knl_cl()
    res_cl = knl_cl.out1.get(), knl_cl.out2.get()
    knl_py()
    res_py = knl_cl.out1.get(), knl_cl.out2.get()
    assert np.all(res_cl[0] == res_py[0]) and np.all(res_cl[1] == res_py[1])


# implement pointer arithmetics with pointer wrapper class for every variable
def test_pointer_arithmetics(thread):
    # todo:
    # Problem: abstract syntax tree does not distiguish if an identifier is a pointer or a variable.
    # E.g. if incrementing the pointer to an array a (a=a+1) in Python this would increment all values in
    # the underlying array. However if
    data = np.array([0]).astype(Types.char)
    knl = Kernel('knl_pointer_arithmetics',
                 {'data': data},
                 """
        char a[5]={0};
        a[0] = a[0] + 1;
        a[1] = 1;
        char* b = a + 1;
        data[0] = b[0];
    """,
                 global_size=data.shape)
    knl_cl = knl.compile(thread)
    emulation.set_b_use_existing_file_for_emulation(False)
    knl_py = knl.compile(thread, b_python=True)
    knl_cl()
    res_cl = knl_cl.data.get()
    knl_py()
    res_py = knl_cl.data.get()
    assert np.all(res_cl[0] == res_py[0])


def test_pointer_increment(
        thread):  # todo use https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ctypes.html
    data = np.array([0]).astype(Types.char)
    func = Function('func',
                    {'data': Private(data.dtype)},
                    """
        return data[0];
    """, return_type=data.dtype)
    knl = Kernel('knl_pointer_arithmetics',
                 {'data': data},
                 """
        private char a[5] = {0};
        a[3] = 5;
        data[0] = func(a+3);
    """,
                 global_size=data.shape)
    prog = Program(functions=[func], kernels=[knl])
    knl_cl = prog.compile(thread).knl_pointer_arithmetics
    knl_py = prog.compile(thread, b_python=True).knl_pointer_arithmetics
    knl_cl()
    res_cl = knl_cl.data.get()
    knl_py()
    res_py = knl_cl.data.get()
    assert np.all(res_cl[0] == res_py[0])


def test_bit_shift(thread):  # todo use https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ctypes.html
    data = np.array([0, 0, 0, 0]).astype(Types.char)
    knl = Kernel('knl_bit_packing',
                 {'data': data},
                 """
        uchar a = 5;
        uchar b = 3;
        uchar c = (a << 4) | b;
        data[0] = (c & 0xF0) >> 4;
        data[1] = c & (0x0F);
    """,
                 global_size=data.shape)
    prog = Program(kernels=[knl])
    knl_cl = prog.compile(thread).knl_bit_packing
    knl_py = prog.compile(thread, b_python=True).knl_bit_packing
    knl_cl()
    thread.queue.finish()
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
def test_for_loop(thread, header):
    def eval_code(b_python=False):
        data = to_device(thread.queue, np.array([0]).astype(Types.char))
        knl = Kernel('knl_test_for_loop',
                     {'data': Global(data)},
                     """
            ${header}{
                data[0]+=i;
            }
        """,
                     replacements={'header': header},
                     global_size=data.shape).compile(thread, b_python=b_python)
        knl()
        thread.queue.finish()
        res = knl.data.get()
        return res

    assert np.all(eval_code(False) == eval_code(True))


def test_vector_types(thread):  # todo use https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ctypes.html
    data = np.zeros((10,)).astype(Types.char2)
    knl = Kernel('knl_vector_types',
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
    knl_cl = knl.compile(thread)
    knl_py = knl.compile(thread, b_python=True)
    knl_cl()
    thread.queue.finish()
    res_cl = knl_cl.data.get()
    knl_py()
    res_py = knl_py.data.get()
    assert np.all(res_cl == res_py)


def test_nested_local_barrier_inside_function(thread):
    func_nested = Function('nested_func',
                           {
                               'ary': Global(Types.int),
                               'shared': Local(Types.int)},
                           """
               shared[get_global_id(0)] = ary[get_global_id(0)];
               barrier(CLK_LOCAL_MEM_FENCE);
               return shared[(get_global_id(0)+1)%2] ;
               """,
                           return_type=Types.int)
    func_parent = Function('parent',
                           func_nested.args,
                           """
               return nested_func(ary, shared);
               """,
                           return_type=Types.int)

    ary = to_device(thread.queue, (ary_np := np.array([1, 2]).astype(Types.int)))
    set_b_use_existing_file_for_emulation(False)
    knl = Kernel('some_knl',
                 {
                     'ary': Global(ary),
                 },
                 """
                __local int shared[2];
                ary[get_global_id(0)] = parent(ary, shared);
                   """, global_size=ary.shape)
    prog = Program([func_nested, func_parent], [knl])
    prog_py = prog.compile(thread, b_python=True)
    prog_cl = prog.compile(thread, b_python=False)
    prog_py.some_knl()

    ary_py = ary.get()
    ary.set(ary_np)
    prog_cl.some_knl()
    ary_cl = ary.get()
    thread.queue.finish()
    assert np.allclose(ary_py, np.array([2, 1]))
    assert np.allclose(ary_py, ary_cl)
