import logging
import time

import numpy as np

from pyopencl_extension import Helpers, Program, Kernel, Function, Scalar, Global, HashArray, Array, zeros, \
    to_device
from pyopencl_extension import get_current_queue, set_current_queue, create_queue
from pyopencl_extension.modifications_pyopencl.command_queue import QueueProperties
from pyopencl_extension.types.utilities_np_cl import c_name_from_dtype, Types
import pytest


def test_current_queue_feature_1():
    # whenever create_thread() is called this is set as the current thread. Be careful, that Thread() leads to a new
    # context/queue.
    queue = create_queue()
    assert hash(get_current_queue()) == hash(queue)


def test_current_queue_feature_2():
    set_current_queue(None)
    queue1 = get_current_queue()
    # use get_devices() to get list with available_devices where index corresponds to device id
    set_current_queue(queue1)
    thread1_reused = get_current_queue()
    assert hash(queue1) == hash(thread1_reused)
    set_current_queue(create_queue(device_id=0))
    thread2 = get_current_queue()
    assert hash(queue1) != hash(thread2)
    set_current_queue(queue1)
    thread1_reused = get_current_queue()
    assert hash(queue1) == hash(thread1_reused)


class MyComponentAutomaticArgs:
    def __init__(self):
        self.buff = zeros((10,), Types.short)
        self.knl = Kernel('some_operation',
                          {'buff': Global(self.buff),
                           'number': Scalar(Types.short(3))},
                          ["""
                                buff[get_global_id(0)] = number;
                                """],
                          global_size=self.buff.shape).compile()

    def __call__(self, *args, **kwargs):
        self.knl()


def test_automatic_kernel_arguments():
    component = MyComponentAutomaticArgs()

    component()

    buff_cl = component.buff.get()

    assert np.all(buff_cl == 3.0)


class MyComponentManualArgs:
    def __init__(self, b_create_kernel_file: bool = True):
        self.buff = zeros((10,), Types.short)
        self.knl = Kernel('some_operation',
                          {'buff': Global(self.buff),
                           'number': Scalar(Types.short(1))},
                          ["""
                                buff[get_global_id(0)] = number;
                                """],
                          global_size=self.buff.shape).compile()

    def __call__(self, number: float = 0, **kwargs) -> Array:
        self.knl(number=number, **kwargs)
        return self.buff


def test_manual_kernel_arguments():
    component = MyComponentManualArgs()

    component(number=3)

    buff_cl = component.buff.get()

    assert np.all(buff_cl == 3)


class MyComponentManualNoSuperCall(MyComponentManualArgs):
    def __call__(self, number: float = 0, **kwargs) -> Array:
        self.knl(global_size=self.buff.shape,
                 buff=self.buff,
                 number=number)
        return self.buff


def test_manual_kernel_arguments_no_super_call():
    component = MyComponentManualNoSuperCall()

    component(number=3)

    buff_cl = component.buff.get()

    assert np.all(buff_cl == 3.0)


class MyComponentComplexExample:
    def __call__(self, *args, **kwargs):
        if self.mode == 'a':
            self.program.some_operation(number=1)
        elif self.mode == 'b':
            self.knl(number=1)
        return self.buff

    def __init__(self, mode='a', b_create_kernel_file: bool = True):
        self.mode = mode
        self.buff = zeros((10,), Types.short)
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
                               kernels=[knl]).compile()
        self.knl = knl


def test_complex_example():
    component = MyComponentComplexExample()
    component_b = MyComponentComplexExample(mode='b')
    res = component().get()
    res_b = component_b().get()

    assert np.all(res == 11)


def test_complex_example_conversion_python():
    component = MyComponentComplexExample()
    res_cl = component().get()
    res_py = component(emulate=True).get()

    assert np.all(res_cl == res_py - res_cl)


def test_non_existing_argument_raises_warning():
    component = MyComponentComplexExample()
    try:
        res_cl = component(buffer2='something').get()
    except ValueError as err:
        assert str(
            err) == 'keyword argument [\'buffer2\'] does not exist in kernel argument list [\'buff\', \'number\']'


def test_memoize_kernel():
    # thread = Thread(profile=True)
    ary_a = np.ones(int(1e3))
    ary_b = np.zeros(ary_a.shape)
    ary_a_buffer = to_device(ary_a)
    ary_b_buffer = to_device(ary_b)
    n_recompilations = 100
    for i in range(n_recompilations + 1):
        kernels = []
        for j in range(10):
            some_knl = Kernel(f'some_knl_{j}',
                              {'ary_a': Global(ary_a_buffer),
                               'ary_b': Global(ary_b_buffer)},
                              """
                     ary_b[get_global_id(0)] = ary_a[get_global_id(0)];
                     """)
            kernels.append(some_knl)
        Program(kernels=kernels).compile()
        some_knl(global_size=ary_a.shape)
        if i == 1:
            t = time.time()
    time_per_recompile = (time.time() - t) / n_recompilations
    # thread.queue.get_profiler().show_histogram_cumulative_kernel_times()
    print(time_per_recompile)
    assert time_per_recompile < 0.001  # less than 1 ms overhead per recompilation achieved through caching


def test_get_refreshed_argument_of_memoized_kernel():
    for i in range(10):
        ary_a = np.ones(100) + i
        ary_b = np.zeros(100)
        some_knl = Kernel('some_knl',
                          {'ary_a': Global(to_device(ary_a)),
                           'ary_b': Global(to_device(ary_b))},
                          """
                 ary_b[get_global_id(0)] = ary_a[get_global_id(0)];
                 """).compile()
        some_knl(global_size=ary_a.shape)
    assert np.all(some_knl.ary_b.get() == ary_a)


def test_kernel_arg_type_conversion():
    mem = {'ary_b': zeros(shape=(100,), dtype=Types.int)}
    for i in range(5):
        ary_a = np.ones(100, Types.int)
        some_knl = Kernel('some_knl',
                          mem | {'ary_a': ary_a,
                                 'offset': float(i),  # checks if float is accepted
                                 'val': Types.ushort(5.0)  # just a dummy value to test if ushort is accepted
                                 },
                          'ary_a[get_global_id(0)] = ary_a[get_global_id(0)] + offset;' + \
                          'ary_b[get_global_id(0)] = ary_b[get_global_id(0)] + offset;',
                          global_size=ary_a.shape).compile()
        some_knl()
    assert np.all(some_knl.ary_a.get() == ary_a + 4)  # every kernel call the numpy array is send to device
    assert np.all(10 == mem['ary_b'].get())


logging.basicConfig(level=logging.INFO)


def test_local_from_global_dimenstions():
    local_size = Helpers._get_local_size_coalesced_last_dim(global_size=(10000, 128), desired_wg_size=64)
    assert local_size == (1, 64)
    local_size = Helpers._get_local_size_coalesced_last_dim(global_size=(13, 13), desired_wg_size=64)
    assert local_size == (1, 13)
    local_size = Helpers._get_local_size_coalesced_last_dim(global_size=(10000,), desired_wg_size=64)
    assert local_size == (50,)


def test_multiple_command_queues():
    queue1 = create_queue(device_id=0)
    queue2 = create_queue(context=queue1.context)
    ary_a = to_device(np.ones(100000) + 1, queue1)
    ary_b = to_device(np.zeros(100000), queue1)
    some_knl = Kernel('some_knl',
                      {'ary_a': Global(ary_a),
                       'ary_b': Global(ary_b)},
                      """
             ary_b[get_global_id(0)] += ary_a[get_global_id(0)];
             """, global_size=ary_a.shape).compile(queue2.context)
    some_knl(queue=queue2)
    # thread2.queue.finish()
    some_knl(queue=queue1)
    test = 0


def test_hash_array():
    ary = zeros(shape=(100,), dtype=Types.float)
    hash_ary = HashArray(ary)
    a_hash = hash_ary.hash.copy()
    hash_ary.set(np.ones(hash_ary.shape).astype(hash_ary.dtype))
    b_hash = hash_ary.hash.copy()
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

@pytest.mark.skip()
def test_profiling():
    # todo: python (net) time seems not correct
    # - How to deal with multiple events created in e.g. zeros()
    queue = create_queue(queue_properties=QueueProperties.PROFILING_ENABLE)
    size = int(1e8)
    for i in range(10):
        _ = zeros((size,), dtype=Types.int)
    get_current_queue().get_profiler().show_histogram_cumulative_kernel_times()
    queue.finish()


# def test_kernel_timeit():
#     # todo: python (net) time seems not correct
#     # - How to deal with multiple events created in e.g. zeros()
#     thread = Thread(profile=True)
#     ary_a = np.ones(100) + 5
#     ary_b = np.zeros(100)
#     some_knl = Kernel('some_knl',
#                       {'ary_a': Global(to_device(thread.queue, ary_a)),
#                        'ary_b': Global(to_device(thread.queue, ary_b))},
#                       """
#              ary_b[get_global_id(0)] = ary_a[get_global_id(0)];
#              """, global_size=ary_a.shape).compile(thread)
#     some_knl.timeit(reps=1, number=10)


def test_add_functions_inside_function_or_kernel_definition():
    ary_a = to_device(np.ones(100))
    fnc_add3 = Function('add_three',
                        {'a': Scalar(Types.int)},
                        'return a + 3;', returns=Types.int)
    fnc_add5 = Function('add_five',
                        {'a': Scalar(Types.int)},
                        """
             return add_three(a)+2;
             """,
                        functions=[fnc_add3], returns=Types.int)
    some_knl = Kernel('some_knl',
                      {'ary_a': Global(ary_a)},
                      """
             ary_a[get_global_id(0)] = add_five(ary_a[get_global_id(0)]);
             """, global_size=ary_a.shape,
                      functions=[fnc_add5])
    functions = []  # funcitons defined here have higher proiority in case of name conflicts
    Program(functions=functions, kernels=[some_knl]).compile()
    some_knl()
    assert ary_a.get()[0] == 6


def test_conversion_knl_fnc_args_with_no_pointer_format():
    # import pyopencl_extension as cl
    # cl.activate_profiling()
    a_np = np.array([0.1, 0.2], dtype=Types.float)
    b_cl = zeros(shape=(2,), dtype=Types.float)
    fnc = Function('copy_fnc',
                   {'a': a_np, 'b': b_cl, 'idx': Scalar(Types.int)},
                   """
                   b[idx] = a[idx];
                   """)
    knl = Kernel('some_knl',
                 {'a': a_np, 'b': b_cl},
                 """
                 copy_fnc(a, b, get_global_id(0));
                 """, functions=[fnc], global_size=b_cl.shape)
    knl.compile()
    knl()
    # cl.evaluate_profiling()
    assert np.all(a_np == b_cl.get())
