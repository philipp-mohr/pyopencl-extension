import logging
from pathlib import Path

import numpy as np
from pyopencl import Program as pyopencl_program
from pytest import mark

from pyopencl_extension import emulation, use_existing_file_for_emulation, Local, Scalar, \
    LocalArray, Types, Kernel, Global, Function, Program, Private, zeros, zeros_like, to_device, get_current_queue
from pyopencl_extension.emulation import unparse_c_code_to_python, create_py_file_and_load_module, compute_linear_idx, \
    compute_tuple_idx
from pyopencl_extension.framework import create_cl_files

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


def test_debug_kernel():
    buff_cl = zeros((10, 1), Types.short)
    # compute result with opencl
    queue = get_current_queue()
    program = pyopencl_program(queue.context, str(rendered_template)).build()
    some_operation = program.all_kernels()[0]
    # some_operation(init.queue, buff_cl.shape, None,
    #                buff_cl.data, np.dtype(ClTypes.short).type(3))
    # alternative:
    some_operation.set_scalar_arg_dtypes([None, Types.short])
    some_operation(queue, buff_cl.shape, None, buff_cl.data, 3)

    buff_py = zeros((10, 1), Types.short)
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


def test_debug_kernel_with_complex_numbers():
    buff_cl = zeros((10, 1), Types.cfloat)
    queue = get_current_queue()
    # compute result with opencl
    program = pyopencl_program(queue.context, str(rendered_template_cplx_float)).build()
    some_operation = program.all_kernels()[0]
    # some_operation.set_scalar_arg_dtypes([None, ClTypes.cfloat])
    some_operation(queue, buff_cl.shape, None,
                   buff_cl.data, np.dtype(Types.cfloat).type(3 + 1j))

    buff_py = zeros((10, 1), Types.cfloat)
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


def test_debug_kernel_with_complex_numbers_double():
    cplx_t = Types.cdouble
    buff_cl = zeros((10, 1), cplx_t)
    queue = get_current_queue()
    # compute result with opencl
    program = pyopencl_program(queue.context, str(rendered_template_cplx_double)).build()
    some_operation = program.all_kernels()[0]
    # some_operation.set_scalar_arg_dtypes([None, ClTypes.cfloat])
    some_operation(queue, buff_cl.shape, None,
                   buff_cl.data, np.dtype(cplx_t).type(3 + 1j))

    buff_py = zeros((10, 1), cplx_t)
    code_py = unparse_c_code_to_python(rendered_template_cplx_float)
    module = create_py_file_and_load_module(code_py, str(path_py_cl.joinpath('debug_component_complex_numbers_double')))

    module.some_operation(buff_py.shape, None,
                          buff_py, np.dtype(cplx_t).type(3 + 1j))

    assert np.all(buff_py.get() == buff_cl.get())


def test_debug_try_python_conversion_if_c_compilation_fails():
    pass


# from here on all tests depend on framework
def test_debug_c_code_with_unary_increment_operation_inside_of_array():
    buff_cl = zeros((6, 1), Types.short)
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
    compiled_cl = knl.compile(emulate=False)
    compiled_cl(buff=buff_cl)
    buff_py = zeros((6, 1), Types.short)
    compiled_py = knl.compile(emulate=True, file=Path(__file__).parent.joinpath('py_cl_kernels/knl'))
    compiled_py(buff=buff_py)
    assert np.all(buff_py.get() == buff_cl.get())


def test_access_complex_variable():
    buff = np.array([0.5]).astype(Types.cfloat)
    buff_in = to_device(buff)
    buff_out = zeros_like(buff_in)
    knl = Kernel('knl',
                 {'inp': Global(buff_in),
                  'out': Global(buff_out)},
                 """
        out[get_global_id(0)].real = inp[get_global_id(0)].real; 
    """,
                 global_size=(1,))
    compiled_cl = knl.compile(emulate=False, file=Path(__file__).parent.joinpath('py_cl_kernels/knl'))
    compiled_cl()
    buff_out_py = zeros_like(buff_in)
    compiled_py = knl.compile(emulate=True, file=Path(__file__).parent.joinpath('py_cl_kernels/knl'))
    # out[0] = complex64(inp[0].real+out[0].imag*1j) instead of out[0].real=inp[0].real
    compiled_py(out=buff_out_py)
    assert np.all(buff_out.get() == buff_out_py.get())


def test_debug_kernel_with_barriers():
    buff = np.zeros(shape=(2, 4)).astype(Types.int)
    mem_buffer = to_device(buff)
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
    compiled_cl = knl.compile(emulate=False, file=Path(__file__).parent.joinpath('py_cl_kernels/knl'))
    compiled_cl()
    mem_buffer_py = zeros_like(mem_buffer)
    compiled_py = knl.compile(emulate=True, file=Path(__file__).parent.joinpath('py_cl_kernels/knl'))
    # out[0] = complex64(inp[0].real+out[0].imag*1j) instead of out[0].real=inp[0].real
    compiled_py(mem_glob=mem_buffer_py)
    assert np.all(mem_buffer.get() == mem_buffer_py.get())


def test_number_overflow():
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
    knl_cl = knl.compile()
    knl_py = knl.compile(emulate=True)
    knl_cl()
    res_cl = knl_cl.out1.get(), knl_cl.out2.get()
    knl_py()
    res_py = knl_cl.out1.get(), knl_cl.out2.get()
    assert np.all(res_cl[0] == res_py[0]) and np.all(res_cl[1] == res_py[1])


# implement pointer arithmetics with pointer wrapper class for every variable
def test_pointer_arithmetics():
    # Problem: abstract syntax tree does not distinguish if an identifier is a pointer or a variable.
    # E.g. if incrementing the pointer to an array a (a=a+1) in Python this would increment all values in
    # the underlying array. However if
    data = np.array([0, 0]).astype(Types.char)
    knl = Kernel('knl_pointer_arithmetics',
                 {'data': data},
                 """
        char a[5]={0};
        a[0] = a[0] + 1;
        a[1] = 1;
        char* b = a + 1;
        b -= 1; b += 1;
        data[0] = b[0];
        
        char* c = a;
        c += 1;
        a[1] = 3;
        data[1] = c[0];
    """, global_size=data.shape)
    emulation.use_existing_file_for_emulation(False)
    knl_py = knl.compile(emulate=True)
    knl_cl = knl.compile()
    knl_cl()
    res_cl = knl_cl.data.get()
    knl_py()
    res_py = knl_cl.data.get()
    assert np.all(res_cl[0] == res_py[0])


@mark.parametrize('dtype',
                  [Types.char,
                   Types.char4],
                  ids=['scalar type',
                       'vector type'])
def test_pointer_increment(dtype):
    # todo use https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ctypes.html
    data = np.array([0]).astype(dtype)
    func = Function('func',
                    {'data': Private(data.dtype)},
                    """
        return data[0];
    """, returns=data.dtype)
    # Assigning to array pointers does not work in c (e.g. b=a does not compile):
    # https://stackoverflow.com/questions/744536/c-array-declaration-and-assignment
    # Below this can be solved by creating pointers p1 and p2 where their address can be exchange by assignment
    knl = Kernel('knl_pointer_arithmetics',
                 {'data': data},
                 """ private dtype a[5] = {0}; private dtype b[5] = {0};
                     dtype *p1 = a; dtype *p2 = b;
                     a[3] = (dtype)(5);                     
                     p2 = a;
                     data[0] = func(p2+3); """, global_size=data.shape, type_defs={'dtype': dtype})
    prog = Program(functions=[func], kernels=[knl])
    knl_cl = prog.compile().knl_pointer_arithmetics
    knl_cl()
    res_cl = knl_cl.data.get()
    knl_py = prog.compile(emulate=True).knl_pointer_arithmetics
    knl_py()
    res_py = knl_cl.data.get()
    assert np.all(res_cl[0] == res_py[0])


def test_bit_shift():  # todo use https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ctypes.html
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
    knl_cl = prog.compile().knl_bit_packing
    knl_py = prog.compile(emulate=True).knl_bit_packing
    knl_cl()
    get_current_queue().finish()
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
def test_for_loop(header):
    def eval_code(emulate=False):
        data = to_device(np.array([0]).astype(Types.char))
        knl = Kernel('knl_test_for_loop',
                     {'data': Global(data)},
                     """
            ${header}{
                data[0]+=i;
            }
        """,
                     replacements={'header': header},
                     global_size=data.shape).compile(emulate=emulate)
        knl()
        get_current_queue().finish()
        res = knl.data.get()
        return res

    assert np.all(eval_code(False) == eval_code(True))


def test_vector_types():  # todo use https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ctypes.html
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
    knl_cl = knl.compile()
    knl_py = knl.compile(emulate=True)
    knl_cl()
    get_current_queue().finish()
    res_cl = knl_cl.data.get()
    knl_py()
    res_py = knl_py.data.get()
    assert np.all(res_cl == res_py)


def test_nested_local_barrier_inside_function():
    func_nested = Function('nested_func',
                           {
                               'ary': Global(Types.int),
                               'shared': Local(Types.int)},
                           """
               shared[get_global_id(0)] = ary[get_global_id(0)];
               barrier(CLK_LOCAL_MEM_FENCE);
               return shared[(get_global_id(0)+1)%2] ;
               """,
                           returns=Types.int)
    func_parent = Function('parent',
                           func_nested.args,
                           """
               return nested_func(ary, shared);
               """,
                           returns=Types.int)

    ary = to_device((ary_np := np.array([1, 2]).astype(Types.int)))
    use_existing_file_for_emulation(False)
    knl = Kernel('some_knl',
                 {
                     'ary': Global(ary),
                 },
                 """
                __local int shared[2];
                ary[get_global_id(0)] = parent(ary, shared);
                   """, global_size=ary.shape)
    prog = Program([func_nested, func_parent], [knl])
    prog_py = prog.compile(emulate=True)
    prog_cl = prog.compile(emulate=False)
    prog_py.some_knl()

    ary_py = ary.get()
    ary.set(ary_np)
    prog_cl.some_knl()
    ary_cl = ary.get()
    get_current_queue().finish()
    assert np.allclose(ary_py, np.array([2, 1]))
    assert np.allclose(ary_py, ary_cl)


def test_macro_with_arguments():
    defines = {'FUNC(a,b,c)': '{ int tmp = c(a-b); a += b + tmp; }'}  # this is a macro with arguments
    ary = zeros((2,), Types.int)
    func_add_two = Function('add_two',
                            {'a': Scalar(Types.int)},
                            'return a + 2;', returns=Types.int)
    knl = Kernel('knl_macro_func',
                 {'ary': Global(ary)},
                 """
               int a = 1;
               int b = 2;
               FUNC(a, b, add_two)
               ary[get_global_id(0)] = a;
               """,
                 defines=defines,
                 global_size=ary.shape)
    Program([func_add_two], [knl]).compile().knl_macro_func()
    assert np.allclose(ary.get(), np.array([4, 4]).astype(ary.dtype))


# this test acts as an integration test, testing multiple different c operations:
# -ternary operator (e.g. x=cond ? a:b;)
# - ...
# Testing with one large integration test reduces test time compared to multiple individual tests.
def test_different_c_operations_at_once():
    ary = zeros((2,), Types.int)
    knl = Kernel('knl_multiple_c_operations',
                 {'ary': Global(ary)},
                 """int a = 1;
                    int b = 2;
                    dtype val; // test variable definition without assignment
                    dtype *ptr1; // test pointer definition 
                    global dtype *ptr2; // test global pointer definition 
                    ary[get_global_id(0)] = a>get_global_id(0) ? a : b;
                 """, global_size=ary.shape, type_defs={'dtype': ary.dtype}).compile(emulate=True)
    knl()
    assert np.allclose(ary.get(), np.array([1, 2]).astype(ary.dtype))


# https://community.khronos.org/t/allocate-array-in-a-kernel-of-length-known-only-at-runtime/3655/3
# http://developer.amd.com/wordpress/media/2013/12/AMD_OpenCL_Programming_User_Guide2.pdf
# http://developer.amd.com/wordpress/media/2013/01/Introduction_to_OpenCL_Programming-Training_Guide-201005.pdf
@mark.parametrize(argnames='data_t', argvalues=[Types.short, Types.double])
def test_local_memory_as_kernel_argument(data_t):
    def run(emulate=False):
        ary = to_device(np.ones(10).astype(data_t))
        local_mem = LocalArray(dtype=data_t, shape=5)  # 5 is to to test that local array argument is changed
        knl = Kernel('knl_local_arg',
                     {'ary': Global(ary), 'local_mem': local_mem},
                     """
               int offset = get_group_id(0)*get_local_size(0);
               for(int i=0; i<5; i++) local_mem[i] = ary[offset + i];
               barrier(CLK_LOCAL_MEM_FENCE);
               data_t sum = (data_t)(0);
               for(int i=0; i<5; i++) sum+=local_mem[i];
               ary[get_global_id(0)] = sum;
                     """,
                     type_defs={'data_t': data_t},
                     global_size=ary.shape,
                     local_size=(5,))
        local_mem = LocalArray(dtype=data_t, shape=5)
        knl.compile(emulate=emulate)(local_mem=local_mem)
        return ary.get()

    ary_cl = run(emulate=False)
    ary_py = run(emulate=True)
    assert np.allclose(ary_cl, ary_py) and np.allclose(ary_cl, 5 * np.ones(10).astype(ary_py.dtype))


# constants in global scope cannot be set from host.
# https://stackoverflow.com/questions/7140820/opencl-initializing-program-scope-variables-from-the-host
# def test_program_scope_variable(thread):
#     pass


def test_barrier_global_local_mem_fence():
    pass


@mark.parametrize(['name', 'dtype'], [('fmax', Types.float),
                                      ('abs_diff', Types.int),
                                      ('abs_diff', Types.int),
                                      ('max', Types.int),
                                      ])
def test_two_input_integer_functions(name, dtype):
    a_cl = to_device(np.ones((10,), dtype))
    a_emulation = to_device(np.ones((10,), dtype))
    knl = Kernel(f'knl_{name}',
                 {'a': Global(a_cl), 'num': Scalar(dtype(0))},
                 f'a[get_global_id(0)]={name}(a[get_global_id(0)], num);', global_size=a_cl.shape)
    knl.compile()(a=a_cl)
    knl.compile(emulate=True)(a=a_emulation)
    assert np.all(a_cl.get() == a_emulation.get())



import pyopencl_extension as cl


def test_kernel_compile_on_call_and_abbreviations_work_item_builtin_fncs():
    a = cl.zeros((10,), dtype=cl.int)
    knl = cl.Kernel('my_knl', {'a': a},
                    """a[gid0]+=1.0; a[gid0] = a[gid0];""",
                    global_size=a.shape)
    knl_py = knl.compile(emulate=True)
    # dfs = {}
    # knl = cl.Kernel('my_knl', {'a': a}, 'a[get_global_id(0)]+=1.0;', global_size=a.shape, defines=dfs).compile(emulate=True)
    knl()  # compile on execution if not compiled yet
    # knl(emulate=True)
    assert a.get()[0] == 1.0
    knl_py()
    assert a.get()[0] == 2.0


@mark.parametrize('idx_tuple,dimensions, idx_lin_ref', [((1, 2, 0), (2, 4, 2), 12),
                                                        ((1, 2), (2, 4), 6),
                                                        ((1,), (2,), 1)])
def test_compute_tuple_from_idx_linear(idx_tuple, dimensions, idx_lin_ref):
    idx_lin = compute_linear_idx(idx_tuple, dimensions)
    assert idx_lin == idx_lin_ref
    idx_tuple2 = compute_tuple_idx(idx_lin, dimensions)
    assert idx_tuple == idx_tuple2


def test_kernel_pointer_decl():
    a = cl.zeros((10,), dtype=cl.int)
    knl = cl.Kernel('my_knl', {'a': a},
                    """
                    global int* c_glob = a;
                    private int ary[5];
                    int* c;
                    int* b=ary;
                    c = b; // swap pointers
                    c[1] = 5;
                    a[1] = ary[1];
                    c_glob[2] = 10;""",
                    global_size=a.shape)
    knl_py = knl.compile(emulate=True)
    knl_py()
    assert a.get()[1] == 5   # verify private pointer behavior
    assert a.get()[2] == 10  # verify global pointer behavior global int* c_glob = a;
    a2 = cl.zeros((10,), dtype=cl.int)
    knl(a=a2)
    assert a2.get()[1] == 5
    assert a2.get()[2] == 10
