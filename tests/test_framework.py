import logging
import time

import numpy as np
from pyopencl.array import zeros, Array, to_device

from pyopencl_extension import ClHelpers, ClInit, ClTypes, ClProgram, ClKernel, KnlArgBuffer, \
    KnlArgScalar, \
    ClFunction, ArgScalar, ArgBuffer, c_name_from_dtype


class MyComponentAutomaticArgs:
    def __init__(self, cl_init: ClInit):
        self.buff = zeros(cl_init.queue, (10,), ClTypes.short)
        self.knl = ClKernel('some_operation',
                            {'buff': KnlArgBuffer(self.buff),
                             'number': KnlArgScalar(ClTypes.short, 3)},
                            ["""
                                buff[get_global_id(0)] = number;
                                """],
                            global_size=self.buff.shape).compile(cl_init)

    def __call__(self, *args, **kwargs):
        self.knl()


def test_automatic_kernel_arguments(cl_init):
    component = MyComponentAutomaticArgs(cl_init)

    component()

    buff_cl = component.buff.get()

    assert np.all(buff_cl == 3.0)


class MyComponentManualArgs:
    def __init__(self, cl_init: ClInit, b_create_kernel_file: bool = True):
        self.buff = zeros(cl_init.queue, (10,), ClTypes.short)
        self.knl = ClKernel('some_operation',
                            {'buff': KnlArgBuffer(self.buff),
                             'number': KnlArgScalar(ClTypes.short, 1)},
                            ["""
                                buff[get_global_id(0)] = number;
                                """],
                            global_size=self.buff.shape).compile(cl_init)

    def __call__(self, number: float = 0, **kwargs) -> Array:
        self.knl(number=number, **kwargs)
        return self.buff


def test_manual_kernel_arguments(cl_init):
    component = MyComponentManualArgs(cl_init)

    component(number=3)

    buff_cl = component.buff.get()

    assert np.all(buff_cl == 3)


class MyComponentManualNoSuperCall(MyComponentManualArgs):
    def __call__(self, number: float = 0, **kwargs) -> Array:
        self.knl(global_size=self.buff.shape,
                 buff=self.buff,
                 number=number)
        return self.buff


def test_manual_kernel_arguments_no_super_call(cl_init):
    component = MyComponentManualNoSuperCall(cl_init)

    component(number=3)

    buff_cl = component.buff.get()

    assert np.all(buff_cl == 3.0)


class MyComponentComplexExample:
    def __call__(self, *args, **kwargs):
        self.program.some_operation()
        return self.buff

    def __init__(self, cl_init: ClInit, b_create_kernel_file: bool = True):
        self.buff = zeros(cl_init.queue, (10,), ClTypes.short)
        self.data_t = self.buff.dtype
        func = ClFunction('plus_one',
                          {'buffer': ArgBuffer(self.buff.dtype),
                           'idx': ArgScalar(ClTypes.int)},
                          ["""
                                  return buffer[idx]+${some_integer};
                                  """],
                          {'some_integer': 5},
                          ClTypes.float)
        knl = ClKernel('some_operation',
                       {'buff': KnlArgBuffer(self.buff, '', True),
                        'number': KnlArgScalar(ClTypes.short, 3.0)},
                       ["""
                                data_t factor = convert_${data_t}(1.3);
                                buff[get_global_id(0)] = plus_one(buff, get_global_id(0)) + SOME_CONSTANT*factor;
                                """],
                       replacements={'data_t': c_name_from_dtype(self.data_t)},
                       global_size=self.buff.shape)
        defines = {'SOME_CONSTANT': 6}
        type_defs = {'data_t': self.data_t}
        self.program = ClProgram(defines=defines, type_defs=type_defs, functions=[func],
                                 kernels=[knl]).compile(cl_init)


def test_complex_example(cl_init):
    component = MyComponentComplexExample(cl_init)
    res = component().get()

    assert np.all(res == 11)


def test_complex_example_conversion_python(cl_init):
    component = MyComponentComplexExample(cl_init)
    res_cl = component().get()
    res_py = component(b_python=True).get()

    assert np.all(res_cl == res_py - res_cl)


def test_non_existing_argument_raises_warning(cl_init):
    component = MyComponentComplexExample(cl_init)
    try:
        res_cl = component(buffer2='something').get()
    except ValueError as err:
        assert str(
            err) == 'keyword argument [\'buffer2\'] does not exist in kernel argument list [\'buff\', \'number\']'


def test_memoize_kernel(cl_init):
    ary_a = np.ones(100)
    ary_b = np.zeros(100)
    ary_a_buffer = to_device(cl_init.queue, ary_a)
    ary_b_buffer = to_device(cl_init.queue, ary_b)
    n_recompilations = 100
    for i in range(n_recompilations + 1):
        some_knl = ClKernel('some_knl',
                            {'ary_a': KnlArgBuffer(ary_a_buffer),
                             'ary_b': KnlArgBuffer(ary_b_buffer)},
                            """
                 ary_b[get_global_id(0)] = ary_a[get_global_id(0)];
                 """).compile(cl_init)
        some_knl(global_size=ary_a.shape)
        if i == 1:
            t = time.time()
    time_per_recompile = (time.time() - t) / n_recompilations
    assert time_per_recompile < 0.004  # less than 4 ms overhead per recompilation


def test_get_refreshed_argument_of_memoized_kernel(cl_init):
    for i in range(10):
        ary_a = np.ones(100) + i
        ary_b = np.zeros(100)
        some_knl = ClKernel('some_knl',
                            {'ary_a': KnlArgBuffer(to_device(cl_init.queue, ary_a)),
                             'ary_b': KnlArgBuffer(to_device(cl_init.queue, ary_b))},
                            """
                 ary_b[get_global_id(0)] = ary_a[get_global_id(0)];
                 """).compile(cl_init)
        some_knl(global_size=ary_a.shape)
    assert np.all(some_knl.ary_b.get() == ary_a)


def test_automatic_numpy_array_and_scalar_arg_to_device_conversion(cl_init):
    for i in range(10):
        ary_a = np.ones(100)
        some_knl = ClKernel('some_knl',
                            {'ary_a': ary_a,
                             'offset': float(i)},
                            'ary_a[get_global_id(0)] = ary_a[get_global_id(0)] + offset;',
                            global_size=ary_a.shape).compile(cl_init)
        some_knl()
    assert np.all(some_knl.ary_a.get() == ary_a + 9)


logging.basicConfig(level=logging.INFO)


def test_local_from_global_dimenstions(cl_init):
    local_size = ClHelpers._get_local_size_coalesced_last_dim(global_size=(10000, 128), desired_wg_size=64)
    assert local_size == (1, 64)
    local_size = ClHelpers._get_local_size_coalesced_last_dim(global_size=(13, 13), desired_wg_size=64)
    assert local_size == (1, 13)
    local_size = ClHelpers._get_local_size_coalesced_last_dim(global_size=(10000,), desired_wg_size=64)
    assert local_size == (50,)


def test_multiple_command_queues():
    cl_init = ClInit()
    cl_init2 = ClInit(cl_init.context)
    ary_a = to_device(cl_init.queue, np.ones(100000) + 1)
    ary_b = to_device(cl_init.queue, np.zeros(100000))
    some_knl = ClKernel('some_knl',
                        {'ary_a': KnlArgBuffer(ary_a),
                         'ary_b': KnlArgBuffer(ary_b)},
                        """
             ary_b[get_global_id(0)] += ary_a[get_global_id(0)];
             """, global_size=ary_a.shape).compile(cl_init2)
    some_knl(queue=cl_init2.queue)
    # cl_init2.queue.finish()
    some_knl(queue=cl_init.queue)
    test = 0

# to much work to make overloading functionality working from outside of C compiler, because that requires tracking
# types of variables. If complex support is required just make second implementation.
# def test_real_complex_support():
#     cl_init = ClInit()
#     ary_a = to_device(cl_init.queue, np.ones(100))
#     ary_b = to_device(cl_init.queue, np.ones(100))
#     knl = ClKernel('sum_and_multiply',
#                    {'a': KnlArgBuffer(),
#                     'b': KnlArgBuffer()},
#                    """
#              int i = get_global_id(0);
#              a[i] = (a[i]+3) *b[i];
#              """).compile()
