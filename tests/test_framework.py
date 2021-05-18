import logging
import time

import numpy as np
from pyopencl.array import zeros, Array, to_device

from pyopencl_extension import Helpers, Thread, Program, Kernel, Function, Scalar, Global, HashArray
from pyopencl_extension.types.utilities_np_cl import c_name_from_dtype, Types


class MyComponentAutomaticArgs:
    def __init__(self, thread: Thread):
        self.buff = zeros(thread.queue, (10,), Types.short)
        self.knl = Kernel('some_operation',
                          {'buff': Global(self.buff),
                           'number': Scalar(Types.short(3))},
                          ["""
                                buff[get_global_id(0)] = number;
                                """],
                          global_size=self.buff.shape).compile(thread)

    def __call__(self, *args, **kwargs):
        self.knl()


def test_automatic_kernel_arguments(thread):
    component = MyComponentAutomaticArgs(thread)

    component()

    buff_cl = component.buff.get()

    assert np.all(buff_cl == 3.0)


class MyComponentManualArgs:
    def __init__(self, thread: Thread, b_create_kernel_file: bool = True):
        self.buff = zeros(thread.queue, (10,), Types.short)
        self.knl = Kernel('some_operation',
                          {'buff': Global(self.buff),
                           'number': Scalar(Types.short(1))},
                          ["""
                                buff[get_global_id(0)] = number;
                                """],
                          global_size=self.buff.shape).compile(thread)

    def __call__(self, number: float = 0, **kwargs) -> Array:
        self.knl(number=number, **kwargs)
        return self.buff


def test_manual_kernel_arguments(thread):
    component = MyComponentManualArgs(thread)

    component(number=3)

    buff_cl = component.buff.get()

    assert np.all(buff_cl == 3)


class MyComponentManualNoSuperCall(MyComponentManualArgs):
    def __call__(self, number: float = 0, **kwargs) -> Array:
        self.knl(global_size=self.buff.shape,
                 buff=self.buff,
                 number=number)
        return self.buff


def test_manual_kernel_arguments_no_super_call(thread):
    component = MyComponentManualNoSuperCall(thread)

    component(number=3)

    buff_cl = component.buff.get()

    assert np.all(buff_cl == 3.0)


class MyComponentComplexExample:
    def __call__(self, *args, **kwargs):
        self.program.some_operation()
        return self.buff

    def __init__(self, thread: Thread, b_create_kernel_file: bool = True):
        self.buff = zeros(thread.queue, (10,), Types.short)
        self.data_t = self.buff.dtype
        func = Function('plus_one',
                        {'buffer': Global(self.buff.dtype),
                         'idx': Scalar(Types.int)},
                        ["""
                                  return buffer[idx]+${some_integer};
                                  """],
                        {'some_integer': 5},
                        returns=Types.float)
        knl = Kernel('some_operation',
                     {'buff': Global(self.buff),
                      'number': Scalar(Types.short(3.0))},
                     ["""
                                data_t factor = convert_${data_t}(1.3);
                                buff[get_global_id(0)] = plus_one(buff, get_global_id(0)) + SOME_CONSTANT*factor;
                                """],
                     replacements={'data_t': c_name_from_dtype(self.data_t)},
                     global_size=self.buff.shape)
        defines = {'SOME_CONSTANT': 6}
        type_defs = {'data_t': self.data_t}
        self.program = Program(defines=defines, type_defs=type_defs, functions=[func],
                               kernels=[knl]).compile(thread)


def test_complex_example(thread):
    component = MyComponentComplexExample(thread)
    res = component().get()

    assert np.all(res == 11)


def test_complex_example_conversion_python(thread):
    component = MyComponentComplexExample(thread)
    res_cl = component().get()
    res_py = component(emulate=True).get()

    assert np.all(res_cl == res_py - res_cl)


def test_non_existing_argument_raises_warning(thread):
    component = MyComponentComplexExample(thread)
    try:
        res_cl = component(buffer2='something').get()
    except ValueError as err:
        assert str(
            err) == 'keyword argument [\'buffer2\'] does not exist in kernel argument list [\'buff\', \'number\']'


def test_memoize_kernel(thread):
    ary_a = np.ones(100)
    ary_b = np.zeros(100)
    ary_a_buffer = to_device(thread.queue, ary_a)
    ary_b_buffer = to_device(thread.queue, ary_b)
    n_recompilations = 100
    for i in range(n_recompilations + 1):
        some_knl = Kernel('some_knl',
                          {'ary_a': Global(ary_a_buffer),
                           'ary_b': Global(ary_b_buffer)},
                          """
                 ary_b[get_global_id(0)] = ary_a[get_global_id(0)];
                 """).compile(thread)
        some_knl(global_size=ary_a.shape)
        if i == 1:
            t = time.time()
    time_per_recompile = (time.time() - t) / n_recompilations
    assert time_per_recompile < 0.004  # less than 4 ms overhead per recompilation


def test_get_refreshed_argument_of_memoized_kernel(thread):
    for i in range(10):
        ary_a = np.ones(100) + i
        ary_b = np.zeros(100)
        some_knl = Kernel('some_knl',
                          {'ary_a': Global(to_device(thread.queue, ary_a)),
                           'ary_b': Global(to_device(thread.queue, ary_b))},
                          """
                 ary_b[get_global_id(0)] = ary_a[get_global_id(0)];
                 """).compile(thread)
        some_knl(global_size=ary_a.shape)
    assert np.all(some_knl.ary_b.get() == ary_a)


def test_automatic_numpy_array_and_scalar_arg_to_device_conversion(thread):
    for i in range(10):
        ary_a = np.ones(100)
        some_knl = Kernel('some_knl',
                          {'ary_a': ary_a,
                           'offset': float(i)},
                          'ary_a[get_global_id(0)] = ary_a[get_global_id(0)] + offset;',
                          global_size=ary_a.shape).compile(thread)
        some_knl()
    assert np.all(some_knl.ary_a.get() == ary_a + 9)


logging.basicConfig(level=logging.INFO)


def test_local_from_global_dimenstions(thread):
    local_size = Helpers._get_local_size_coalesced_last_dim(global_size=(10000, 128), desired_wg_size=64)
    assert local_size == (1, 64)
    local_size = Helpers._get_local_size_coalesced_last_dim(global_size=(13, 13), desired_wg_size=64)
    assert local_size == (1, 13)
    local_size = Helpers._get_local_size_coalesced_last_dim(global_size=(10000,), desired_wg_size=64)
    assert local_size == (50,)


def test_multiple_command_queues():
    thread = Thread()
    thread2 = Thread(thread.context)
    ary_a = to_device(thread.queue, np.ones(100000) + 1)
    ary_b = to_device(thread.queue, np.zeros(100000))
    some_knl = Kernel('some_knl',
                      {'ary_a': Global(ary_a),
                       'ary_b': Global(ary_b)},
                      """
             ary_b[get_global_id(0)] += ary_a[get_global_id(0)];
             """, global_size=ary_a.shape).compile(thread2)
    some_knl(queue=thread2.queue)
    # thread2.queue.finish()
    some_knl(queue=thread.queue)
    test = 0


def test_hash_array(thread):
    ary = zeros(thread.queue, shape=(100,), dtype=Types.float)
    hash_ary = HashArray(ary)
    a_hash = hash_ary.hash
    hash_ary.set(np.ones(hash_ary.shape).astype(hash_ary.dtype))
    b_hash = hash_ary.hash
    assert a_hash != b_hash
    hash_ary[0] = 5
    c_hash = hash_ary.hash
    assert c_hash != b_hash

# to much work to make overloading functionality working from outside of C compiler, because that requires tracking
# types of variables. If complex support is required just make second implementation.
# def test_real_complex_support():
#     thread = ClInit()
#     ary_a = to_device(thread.queue, np.ones(100))
#     ary_b = to_device(thread.queue, np.ones(100))
#     knl = ClKernel('sum_and_multiply',
#                    {'a': KnlArgBuffer(),
#                     'b': KnlArgBuffer()},
#                    """
#              int i = get_global_id(0);
#              a[i] = (a[i]+3) *b[i];
#              """).compile()
