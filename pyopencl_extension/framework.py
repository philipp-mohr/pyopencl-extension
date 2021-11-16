import os
import re
import time
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Union, Tuple, List, Dict, Callable

import numpy as np
import pyastyle
from mako import exceptions
from mako.template import Template
import pyopencl as cl
from pyopencl._cl import Device
from pyopencl.array import Array as ClArray

from pyopencl_extension import CommandQueue, Array, to_device
from pyopencl_extension.helpers.general import write_string_to_file
from pyopencl_extension.modifications_pyopencl.command_queue import QueueProperties, get_current_queue
from pyopencl_extension.modifications_pyopencl.context import Context, get_devices
from pyopencl_extension.types.auto_gen.cl_types import ClTypesScalar
from pyopencl_extension.types.utilities_np_cl import c_name_from_dtype, scalar_type_from_vec_type, \
    get_vec_size, Types, number_vec_elements_of_cl_type, VEC_INDICES
from pyopencl_extension.emulation import create_py_file_and_load_module, unparse_c_code_to_python


@dataclass
class LocalArray:
    shape: int
    dtype: np.dtype
    cl_local_memory: cl.LocalMemory = field(init=False, default=None)

    def __post_init__(self):
        self.cl_local_memory = cl.LocalMemory(int(self.shape * np.dtype(self.dtype).itemsize))


TypesClArray = Union[Array, ClArray]
TypesDefines = Union[str, float, int, bool]
TypesReplacement = Union[str, float, int, bool]
TypesArgArrays = Union[np.ndarray, Array, ClArray, LocalArray]
_ = ClTypesScalar
TypesArgScalar = Union[int, float,
                       _.char, _.short, _.int, _.long, _.uchar, _.ushort, _.uint, _.ulong, _.half, _.float, _.double]
# TypesKernelArg = Union[Array, TypesDefines] # todo: remove?

__author__ = "piveloper"
__copyright__ = "26.03.2020, piveloper"
__version__ = "1.0"
__email__ = "piveloper@gmail.com"
__doc__ = """This script includes helpful functions to extended PyOpenCl functionality."""

preamble_activate_double = """
#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define PYOPENCL_DEFINE_CDOUBLE
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#define PYOPENCL_DEFINE_CDOUBLE
#endif
"""

preamble_activate_complex_numbers = """
#include <pyopencl-complex.h>
#define TP_ROOT ${cplx_type}
"""


def preamble_precision(precision: str = 'single'):
    """
    This function generates preamble to support either single or double precision floating point numbers.
    :param precision:
    :return:
    """
    if precision == 'single':
        return """
#define PI 3.14159265359f
            """
    elif precision == 'double':
        return preamble_activate_double + """\n
#define PI 3.14159265358979323846
            """
    else:
        raise NotImplementedError()


def preamble_generic_type_operations(number_format: str = 'real', precision: str = 'single'):
    """
    This function returns a preamble which defines how generic operations are executed on device.

    This becomes especially important when dealing with complex types which  OpenCl does not support out of the box.
    As a solution, pyopencl-complex.h includes several basic function for complex operations.

    E.g. consider a kernel which adds two numbers, however the input type can be real or complex valued.
    Typically one would implement c = a + b. However, OpenCl does not support + operation when a and b are complex
    valued. Therefore using this preamble one can write c = ADD(a,b). ADD acts a a generic operation which supports
    real and complex input depending on selection for number_format.


    :param number_format: 'real' or 'complex
    :param precision: 'single' or 'double'
    :return: preamble to support generic operations
    """

    if number_format == 'complex':
        cplx_type_pyopencl = {'single': 'cfloat',
                              'double': 'cdouble'}[precision]
        return preamble_activate_complex_numbers + """
#define MUL ${cplx_type}_mul
#define ADD ${cplx_type}_add
#define SUB ${cplx_type}_sub
#define ABS ${cplx_type}_abs
#define RMUL ${cplx_type}_rmul
#define NEW ${cplx_type}_new
#define CONJ ${cplx_type}_conj
#define REAL(x) x.real
#define IMAG(x) x.imag
    """.replace('${cplx_type}', cplx_type_pyopencl)
    elif number_format == 'real':
        return """
#define MUL(x,y) (x*y)
#define ADD(x,y) (x+y)
#define SUB(x,y) (x-y)
#define ABS(x) (fabs(x))
#define RMUL(x,y) (x*y)
#define NEW(x,y) (x)
#define CONJ(x) (x)
#define REAL(x) (x)
#define IMAG(x) (0)
        """
    else:
        raise NotImplementedError()


def catch_invalid_argument_name(name: str):
    """
    E.g. when using certain argument names like 'channel' the opencl compiler throws a compilation error, probably
    because channel is an reserved opencl command. There we replace those names by appending '_' character.

    :param name:
    :return:
    """
    invalid_names = ['channel']
    if name in invalid_names:
        raise ValueError('Invalid opencl name: \'{}\' used.'.format(name))
    else:
        return name


class OrderInMemory(Enum):
    C_CONTIGUOUS: str = 'c_contiguous'
    F_CONTIGUOUS: str = 'f_contiguous'


@dataclass
class ArgBase(ABC):
    # too much restriction, shape of array might change during runtime
    # shape: Tuple[int, ...] = (1,)  # default: argument is scalar

    @property
    @abstractmethod
    def address_space_qualifier(self) -> str:
        # __global, __local, __private, __constant
        pass

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        # __global, __local, __private, __constant
        pass

    def to_string(self, name):
        new_name = catch_invalid_argument_name(name)
        if type(self) in [Scalar]:  # scalar
            return '{} {} {}'.format(self.address_space_qualifier, c_name_from_dtype(self.dtype), new_name)
        else:  # array
            return '{} {} *{}'.format(self.address_space_qualifier, c_name_from_dtype(self.dtype), new_name)


@dataclass
class Scalar(ArgBase):
    dtype: np.dtype = field(default=Types.int)
    address_space_qualifier: str = field(default='')
    default: np.dtype = field(init=False, default=None)

    def __post_init__(self):
        if np.isscalar(self.dtype):
            self.default = self.dtype
            if type(self.default) == float:
                self.dtype = Types.double
            elif type(self.default) == int:
                self.dtype = Types.int
            else:
                self.dtype = type(self.dtype)


@dataclass
class Pointer(ArgBase, ABC):
    dtype: np.dtype = field(default=Types.int)
    address_space_qualifier: str = field(init=False, default='__global')


@dataclass
class Private(Pointer):
    address_space_qualifier: str = field(init=False, default='__private')


@dataclass
class Local(Pointer):
    dtype: Union[np.dtype, LocalArray] = field(default=Types.int)
    address_space_qualifier: str = field(init=False, default='__local')
    order_in_memory: OrderInMemory = OrderInMemory.C_CONTIGUOUS
    default: cl.LocalMemory = field(init=False, default=None)

    def __post_init__(self):
        if isinstance(self.dtype, LocalArray):
            self.default = self.dtype.cl_local_memory
            self.dtype = self.dtype.dtype


@dataclass
class Global(Pointer):
    dtype: Union[np.dtype, TypesClArray] = field(default=Types.int)
    read_only: bool = False  # adds 'const' qualifier to let compiler know that global array is never written
    order_in_memory: OrderInMemory = OrderInMemory.C_CONTIGUOUS
    address_space_qualifier: str = field(init=False, default='__global')
    default: TypesClArray = field(init=False, default='')

    def __post_init__(self):
        if isinstance(self.dtype, TypesClArray.__args__):
            self.default = self.dtype
            self.dtype = self.dtype.dtype

            if self.read_only:
                self.address_space_qualifier = 'const __global'


@dataclass
class Constant(Pointer):
    """
    const is only a hint for the compiler that the data does not change
    __constant leads to usage of very fast constant cache memory which is shared among
    multiple compute units. From AMD optimization guide, e.g. we can read 4 bytes/cycles.
    Local memory can be ready twice as fast with 8bytes/cycle, however local memory is a even more scarce resource.

    https://stackoverflow.com/questions/17991714/opencl-difference-between-constant-memory-and-const-global-memory/50931783
    """
    dtype: Union[np.dtype, TypesClArray] = field(default=Types.int)
    order_in_memory: str = OrderInMemory.C_CONTIGUOUS
    address_space_qualifier: str = field(init=False, default='__constant')
    default: TypesClArray = field(init=False, default='')

    def __post_init__(self):
        if isinstance(self.dtype, TypesClArray.__args__):
            self.default = self.dtype
            self.dtype = self.dtype.dtype


def template(func: Union['Kernel', 'Function']) -> str:
    body = ''.join(func.body)
    tpl = func.header + '\n{' + body + '}\n'
    args = [value.to_string(key) + ',' for key, value in func.args.items()]
    args = '{}'.format('\n'.join(args))
    args = args[:-1]  # remove last comma
    replacements = {'name': func.name,
                    'args': args,
                    'returns': c_name_from_dtype(func.returns)}
    for key, value in func.replacements.items():
        replacements[key] = str(value)
    try:  # todo: e.g. if replacement has been forgotten, still save template as file
        tpl = Template(tpl).render(**replacements)
    except:
        raise ValueError(exceptions.text_error_template().render())

    defines = '\n'.join(['#define {} {}'.format(key, str(value)) for key, value in func.defines.items()])
    tpl_formatted = pyastyle.format('{}\n\n{}'.format(defines, tpl), '--style=allman --indent=spaces=4')
    return tpl_formatted


@dataclass
class FunctionBase(ABC):
    name: str = 'func'
    args: Dict[str, Union[TypesArgArrays, TypesArgScalar, Scalar, Global, Local, Private, Constant]] = \
        field(default_factory=lambda: [])
    body: Union[List[str], str] = field(default_factory=lambda: [])
    replacements: Dict[str, TypesReplacement] = field(default_factory=lambda: {})
    type_defs: Dict[str, np.dtype] = field(default_factory=lambda: {})  # todo
    defines: Dict[str, TypesDefines] = field(default_factory=lambda: {})
    functions: List['Function'] = field(default_factory=lambda: [])

    def __post_init__(self):
        if isinstance(self.body, str):
            self.body = [self.body]
        self._prepares_args()

    @property
    def header(self):
        return '${returns} ${name}(${args})'

    @property
    @abstractmethod
    def template(self) -> str:
        pass

    @staticmethod
    def _prepare_arg(v):
        """
        This is a convenience feature. Arguments might be provided only as numpy array, python integer or float etc.
        Therefore, this function adds appropriate pointer type.
        """
        if isinstance(v, np.ndarray):
            g_arg = Global(v.dtype)
            g_arg.default = v
            return g_arg
        elif isinstance(v, TypesClArray.__args__):
            return Global(v)
        elif isinstance(v, LocalArray):
            return Local(v)
        elif isinstance(v, TypesArgScalar.__args__):
            return Scalar(v)
        else:
            return v

    def _prepares_args(self):
        self.args = {k: self._prepare_arg(v) for k, v in self.args.items()}


@dataclass
class Function(FunctionBase):
    @property
    def template(self) -> str:
        return template(self)

    returns: np.dtype = field(default_factory=lambda: np.dtype(np.void))

    def __str__(self) -> str:
        return super().__str__() + str(self.returns)


KernelGridType = Union[Tuple[int], Tuple[int, int], Tuple[int, int, int]]


class Compilable:
    @abstractmethod
    def compile(self, context: Context = None, emulate: bool = False):
        pass

    @staticmethod
    def get_default_dir_pycl_kernels():
        return Path(os.getcwd()).joinpath('cl_py_modules')


@dataclass
class Kernel(FunctionBase, Compilable):
    def compile(self, context: Context = None, emulate: bool = False, file='$default_path'):
        Program(kernels=[self]).compile(context=context, emulate=emulate, file=file)
        return self.callable_kernel

    global_size: KernelGridType = None
    local_size: KernelGridType = None
    returns: np.dtype = field(default_factory=lambda: np.dtype(np.void), init=False)
    callable_kernel: 'CallableKernel' = field(default_factory=lambda: None, init=False)

    def __str__(self) -> str:
        return super().__str__()

    @property
    def template(self) -> str:
        return template(self)

    @property
    def header(self):
        return '__kernel ${returns} ${name}(${args})'

    def __call__(self, global_size: KernelGridType = None, local_size: KernelGridType = None, **kwargs):
        if self.callable_kernel is not None:
            return self.callable_kernel(global_size=global_size, local_size=local_size, **kwargs)
        else:
            raise ValueError('Kernel has not been compiled yet.')


def _get_all_funcs(f: FunctionBase, flat_list=None) -> List[FunctionBase]:
    if flat_list is None:
        flat_list = []
        for sub_f in f.functions:
            _get_all_funcs(sub_f, flat_list)
        flat_list.append(f)
        return flat_list
    else:
        flat_list.extend([_get_all_funcs(sub_f, flat_list) for sub_f in f.functions])
        flat_list.append(f)


def _get_list_with_unique_functions(functions, kernels):
    functions_in_kernels = [f for k in kernels for f in k.functions]
    all_funcs = [_f for f in functions + functions_in_kernels for _f in _get_all_funcs(f)]
    all_funcs_unique, _func_names = [], []
    for f in all_funcs:
        if len(_func_names) == 0 or f.name not in _func_names:
            all_funcs_unique.append(f)
            _func_names.append(f.name)
    return all_funcs_unique


@dataclass
class Program(Compilable):
    """
    Models an OpenCl Program containing functions or kernels.
    """

    def compile(self, context: Context = None, emulate: bool = False,
                file: str = '$default_path') -> 'ProgramContainer':

        return compile_cl_program(self, context, emulate, file)

    functions: List[Function] = field(default_factory=lambda: [])
    kernels: List[Kernel] = field(default_factory=lambda: [])
    defines: Dict[str, TypesDefines] = field(default_factory=lambda: {})
    type_defs: Dict[str, np.dtype] = field(default_factory=lambda: {})

    @staticmethod
    def _arg_to_str_for_hash(name, arg: ArgBase):
        return name + str(type(arg)) + str(hash(arg.dtype)) + arg.address_space_qualifier

    @staticmethod
    def _func_to_str_for_hash(func: FunctionBase):
        str_args = ''.join(Program._arg_to_str_for_hash(k, v) for k, v in func.args.items())
        str_repl = ''.join(k + str(v) for k, v in func.replacements.items())
        str_type_defs = ''.join(k + str(v) for k, v in func.type_defs.items())
        str_defines = ''.join(k + str(v) for k, v in func.defines.items())
        str_body = ''.join(func.body)
        return str_type_defs + str_defines + str_repl + func.header + func.name + str_args + str_body

    def __hash__(self) -> int:
        str_funcs = ''.join(self._func_to_str_for_hash(knl) for knl in self.functions)
        str_kernels = ''.join(self._func_to_str_for_hash(knl) for knl in self.kernels)
        str_type_defs = ''.join(k + str(v) for k, v in self.type_defs.items())
        str_defines = ''.join(k + str(v) for k, v in self.defines.items())
        prog_str = str_defines + str_type_defs + str_funcs + str_kernels
        return int(str(hash(prog_str)) + str(abs(hash(prog_str + 'something'))))  # double hash to avoid collisions

    @property
    def rendered_template(self):
        all_funcs = _get_list_with_unique_functions(self.functions, self.kernels)

        functions = [f.template for f in all_funcs] + [k.template for k in self.kernels]
        functions = '\n'.join(functions)

        if 'double' in functions:
            _preamble_precision = preamble_precision('double')
        else:
            _preamble_precision = preamble_precision('single')

        if 'cfloat_t' in functions:
            _preamble_generic_operations = preamble_generic_type_operations('complex', 'single')
        elif 'cdouble_t' in functions:
            _preamble_generic_operations = preamble_generic_type_operations('complex', 'double')
        else:
            _preamble_generic_operations = preamble_generic_type_operations('real')
        preamble_buff_t = f'{_preamble_precision}\n{_preamble_generic_operations}'

        # join program typedefs with typedefs from kernels and functions
        # todo: consider replacing type strings directly to avoid name conflicts
        def update_and_checks_for_duplicates_same_type(items: dict, dic: dict):
            for key, value in items.items():
                if key in dic:
                    if not dic[key] == value:
                        raise ValueError('Same type def name for different types')
                else:
                    dic[key] = value

        [update_and_checks_for_duplicates_same_type(func.type_defs, self.type_defs) for func in self.functions]
        [update_and_checks_for_duplicates_same_type(func.type_defs, self.type_defs) for func in self.kernels]

        # remove since defines are inserted before function/kernels
        # [update_and_checks_for_duplicates_same_type(func.defines, self.defines) for func in self.functions]
        # [update_and_checks_for_duplicates_same_type(func.defines, self.defines) for func in self.kernels]

        defines = '\n'.join(['#define {} {}'.format(key, str(value)) for key, value in self.defines.items()])

        type_defs = '\n'.join(
            ['typedef {c_name} {new_name};\n#define convert_{new_name}(x) convert_{c_name}(x)'.format(
                c_name=c_name_from_dtype(value),
                new_name=str(key))
                for key, value in self.type_defs.items()])

        tpl_all = self._get_tpl(preamble_buff_t, defines, type_defs, functions)

        tpl_formatted = pyastyle.format(tpl_all, '--style=allman --indent=spaces=4')
        return tpl_formatted

    def _get_tpl(self, preamble_complex, defines, type_defs, functions):
        return '{}\n\n{}\n\n{}\n\n{}\n\n'.format(preamble_complex, defines, type_defs, functions)


def build_for_device(context: Context, template_to_be_compiled: str, file: str = None) -> cl.Program:
    if file is not None:
        write_string_to_file(template_to_be_compiled, file + '.cl', b_logging=False)
    try:
        program = cl.Program(context, str(template_to_be_compiled)).build()
    except Exception as error:
        tpl_line_numbers = '\n'.join(
            ['{:<4}{}'.format(i + 1, line) for i, line in enumerate(template_to_be_compiled.split('\n'))])
        raise ValueError('\n{}\n\n{}'.format(tpl_line_numbers, str(error)))
    return program


# Todo: Find good structure for modeling cl and python kernels
@dataclass
class CallableKernel(ABC):
    kernel_model: Kernel

    def __getattr__(self, name):
        if name in self.kernel_model.args.keys():
            return self.kernel_model.args[name].default
        return super().__getattribute__(name)

    @abstractmethod
    def __call__(self, global_size: KernelGridType = None,
                 local_size: KernelGridType = None,
                 **kwargs):
        pass

    @staticmethod
    def _typing_scalar_argument(arg_model: Union[Scalar, Scalar],
                                scalar_value_provided: TypesArgScalar):
        if get_vec_size(arg_model.dtype) == 1:
            return np.dtype(arg_model.dtype).type(scalar_value_provided)
        else:
            dtype_scalar = scalar_type_from_vec_type(arg_model.dtype)
            scalar = np.dtype(dtype_scalar).type(scalar_value_provided)  # converts to bytes like object
            return scalar.astype(arg_model.dtype)  # converts to vector type

    @staticmethod
    def _prepare_arguments(queue: CommandQueue, knl: Kernel, **kwargs):
        global_size = kwargs.pop('global_size', None)
        local_size = kwargs.pop('local_size', None)
        global_size = knl.global_size if global_size is None else global_size
        local_size = knl.local_size if local_size is None else local_size

        supported_kws = [k for k in knl.args.keys()]
        kw_not_in_kernel_arguments = [kw for kw in kwargs if kw not in supported_kws]
        if len(kw_not_in_kernel_arguments) > 0:
            raise ValueError(
                f'keyword argument {kw_not_in_kernel_arguments} does not exist in kernel argument list {supported_kws}')

        # If kernel arguments are of type np.ndarray they are converted to cl arrays here
        # This is done here, since thq queue is available at this point for sure.
        # todo: deal with case if kwarg is numpy argument
        def deal_with_np_arrays(v):
            if isinstance(v, Global) and isinstance(v.default, np.ndarray):
                v.default = to_device(ary=v.default, queue=queue)
                return v
            else:
                return v

        knl.args = {k: deal_with_np_arrays(v) for k, v in knl.args.items()}

        # set default arguments. Looping over kernel model forces correct order of arguments
        args_call = [kwargs.pop(key, value.default if isinstance(value, (Constant, Global, Scalar, Local)) else None)
                     for key, value in knl.args.items()]
        if any(arg is None for arg in args_call):
            raise ValueError('Argument equal to None can lead to system crash')

        if global_size is None:
            raise ValueError('global_size not provided!')
        if global_size == ():
            raise ValueError('Global size is empty')
        if 0 in global_size:
            raise ValueError(f'Parameter in global size {global_size} equal to zero')
        if local_size is not None and 0 in local_size:
            raise ValueError(f'Parameter in local size {local_size} equal to zero')
        # convert scalar argument to correct type. E.g. argument can be python int and is converted to char
        args_model = list(knl.args.values())
        args_call = [CallableKernel._typing_scalar_argument(args_model[i], arg)
                     if type(args_model[i]) in [Scalar, Scalar] else arg
                     for i, arg in enumerate(args_call)]
        # if argument of type LocalArray extract cl.LocalMemory instance to be passed as argument
        args_call = [arg.cl_local_memory if isinstance(arg, LocalArray) else arg for arg in args_call]
        # check if buffer have same type as defined in the kernel function header
        b_types_equal = [args_call[i].dtype == v.dtype for i, v in enumerate(args_model) if isinstance(v, Global)]
        if not np.all(b_types_equal):
            idx_buffer_list = int(np.argmin(b_types_equal))
            idx = [i for i, kv in enumerate(knl.args.items()) if isinstance(kv[1], Global)][idx_buffer_list]
            buffer_name = [k for k, v in knl.args.items()][idx]
            buffer_type_expected = args_model[idx].dtype
            buffer_type_call = args_call[idx].dtype
            raise ValueError(f'Expected buffer argument \'{buffer_name}\' with type {buffer_type_expected} '
                             f'but got buffer with type {buffer_type_call}')

        # check if buffer elements of array arguments have memory order as expected (c or f contiguous)
        def b_array_memory_order_as_expected(ary_model: Global, ary_call: TypesClArray):
            if ary_model.order_in_memory == OrderInMemory.C_CONTIGUOUS:
                return ary_call.flags.c_contiguous
            else:  # f_contiguous
                return ary_call.flags.f_contiguous

        knl_args_invalid_memory_order = [(k, v) for idx, (k, v) in enumerate(knl.args.items())
                                         if isinstance(v, Global) and
                                         not b_array_memory_order_as_expected(v, args_call[idx])]
        if len(knl_args_invalid_memory_order) > 0:
            msg = '\n'.join([f'Array argument \'{arg[0]}\' is not {arg[1].order_in_memory} (as defined in Kernel)'
                             for arg in knl_args_invalid_memory_order])
            raise ValueError(msg)
        non_supported_types = [np.ndarray]
        if any(_ := [type(arg) in non_supported_types for arg in args_call]):
            raise ValueError(f'Type of argument \'{list(knl.args.items())[np.where(_)[0][0]][0]}\' is not supported in '
                             f'kernel call')
        return global_size, local_size, args_call


@dataclass
class CallableKernelEmulation(CallableKernel):
    function: Callable

    def __call__(self,
                 global_size: KernelGridType = None,
                 local_size: KernelGridType = None,
                 **kwargs: Union[TypesClArray, object]) -> cl.Event:
        # e.g. if two kernels of a program shall run concurrently, this can be enable by passing another queue here
        queue = kwargs.pop('queue', get_current_queue())
        global_size, local_size, args = self._prepare_arguments(queue=queue, knl=self.kernel_model,
                                                                global_size=global_size,
                                                                local_size=local_size, **kwargs)
        self.function(global_size, local_size, *args)
        # create user event with context retrieved from first arg of type Array
        event = cl.UserEvent([_ for _ in args if isinstance(_, TypesClArray.__args__)][0].context)
        event.set_status(cl.command_execution_status.COMPLETE)
        return event


@dataclass
class CallableKernelDevice(CallableKernel):
    compiled: cl.Kernel

    @staticmethod
    def check_local_size_not_exceeding_device_limits(device: Device, local_size):
        # E.g. on nvidia the local size might be individually limited to be (1024,1024,64).
        # This shall trigger an exception, when wrong local size is provided.
        if local_size is not None and any([desired_local_size > device.max_work_item_sizes[dim]
                                           for dim, desired_local_size in enumerate(local_size)]):
            raise ValueError(f'Requested local dimensions {local_size} exceed {device.max_work_item_sizes=}')

    def __call__(self,
                 global_size: KernelGridType = None,
                 local_size: KernelGridType = None,
                 **kwargs) -> cl.Event:
        # e.g. if two kernels of a program shall run concurrently, this can be enable by passing another queue here
        queue = kwargs.pop('queue', get_current_queue())
        assert self.compiled.context.int_ptr == queue.context.int_ptr
        global_size, local_size, args = self._prepare_arguments(queue=queue, knl=self.kernel_model,
                                                                global_size=global_size,
                                                                local_size=local_size, **kwargs)
        self.check_local_size_not_exceeding_device_limits(queue.device, local_size)
        # extract buffer from cl arrays separate, since in emulation we need cl arrays
        args_cl = [arg.data if isinstance(arg, TypesClArray.__args__) else arg for i, arg in enumerate(args)]
        event = self.compiled(queue, global_size, local_size, *args_cl)
        queue.add_event(event, self.kernel_model.name)
        return event


@dataclass
class ProgramContainer:
    """
    Responsibility:
    A callable kernel is returned with program.kernel_name. Depending on value of b_run_python_emulation a call of this
    kernel is executed on device or in emulation.
    """
    program_model: Program
    file: str
    init: CommandQueue
    callable_kernels: Dict[str, Union[CallableKernelEmulation, CallableKernelDevice]] = None

    def __getattr__(self, name) -> CallableKernel:
        if name in self.callable_kernels:
            return self.callable_kernels[name]
        else:
            return super().__getattribute__(name)


# https://stackoverflow.com/questions/1988804/what-is-memoization-and-how-can-i-use-it-in-python
class MemoizeKernelFunctions:
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, program_model: Program, context: Context, file: str = None):
        # body = ''.join(program_model.rendered_template)
        _id = hash(f'{hash(context)}{hash(program_model)}')
        if _id not in self.memo:
            self.memo[_id] = self.f(program_model, context, file)
        return self.memo[_id]


@MemoizeKernelFunctions
def compile_cl_program_device(program_model: Program, context: Context = None, file: str = None) -> Dict[str, Kernel]:
    code_cl = program_model.rendered_template
    program = build_for_device(context, code_cl, file)
    kernels_model = program_model.kernels
    # set scalar arguments for each kernel from kernel model
    callable_kernels = {knl.function_name: knl
                        for i, knl in enumerate(program.all_kernels())}
    for i, knl in enumerate(kernels_model):
        arg_types = [arg.dtype if type(arg) in [Scalar, Scalar] else None
                     for _, arg in kernels_model[i].args.items()]
        callable_kernels[knl.name].set_scalar_arg_dtypes(arg_types)
    return callable_kernels


@MemoizeKernelFunctions
def compile_cl_program_emulation(program_model: Program, context: Context, file: str = None,
                                 *args, **kwargs) -> Dict[str, Callable]:
    code_py = unparse_c_code_to_python(code_c=program_model.rendered_template)
    module = create_py_file_and_load_module(code_py, file)
    kernels_model = program_model.kernels
    callable_kernels = {knl.name: module.__getattribute__(knl.name) for knl in kernels_model}
    return callable_kernels


def compile_cl_program(program_model: Program, context: Context = None, emulate: bool = False,
                       file: str = '$default_path') -> ProgramContainer:
    t_ns_start = time.perf_counter_ns()
    # deal with file name
    if isinstance(file, Path):
        file = str(file)
    if file is None and emulate:
        raise ValueError('You intended to create no file by setting file=None. '
                         'However, a file must be created for debugging.')  # todo can python debugging run without file?
    elif file == '$default_path':
        file = str(program_model.get_default_dir_pycl_kernels().joinpath(program_model.kernels[0].name))

    if context is None:
        context = get_current_queue().context

    dict_kernels_program_model = {knl.name: knl for knl in program_model.kernels}
    if emulate:
        dict_emulation_kernel_functions = compile_cl_program_emulation(program_model, context, file)
        callable_kernels = {k: CallableKernelEmulation(kernel_model=dict_kernels_program_model[k], function=v)
                            for k, v in dict_emulation_kernel_functions.items()}
    else:
        dict_device_kernel_functions = compile_cl_program_device(program_model, context, file)
        callable_kernels = {k: CallableKernelDevice(kernel_model=dict_kernels_program_model[k], compiled=v)
                            for k, v in dict_device_kernel_functions.items()}
    # make callable kernel available in knl model instance
    for knl in program_model.kernels:
        knl.callable_kernel = callable_kernels[knl.name]
    context.add_time_compilation(time.perf_counter_ns() - t_ns_start)
    return ProgramContainer(program_model=program_model,
                            file=file,
                            init=context,
                            callable_kernels=callable_kernels)


def int_safe(val: float):
    if val.is_integer():
        return int(val)
    else:
        raise ValueError(f'val={val} is no integer')


class HashArray(Array):
    def __init__(self, *args, **kwargs):
        if isinstance(args[0], TypesClArray.__args__):
            a = args[0]
            super().__init__(a.queue, a.shape, a.dtype, order="C", allocator=a.allocator,
                             data=a.data, offset=a.offset, strides=a.strides, events=a.events)
        else:
            super().__init__(*args, **kwargs)
        self.hash = hash(self.get().tobytes())

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.update_hash()

    def set(self, ary, queue=None, async_=None, **kwargs):
        res = super().set(ary, queue, async_, **kwargs)
        self.update_hash()
        return res

    def update_hash(self):
        self.hash = hash(self.get().tobytes())


class Helpers:
    # helper methods which can be useful in interplay with this framwork
    @staticmethod
    def _camel_to_snake(name):
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    @staticmethod
    def command_compute_address(n_dim: int) -> str:
        command = '0'
        for i in range(n_dim):
            offset = '1'
            for j in range(i + 1, n_dim):
                offset += '*get_global_size({})'.format(j)
            command += '+get_global_id({})*{}'.format(i, offset)
        return command

    # helpers for using vector types

    @staticmethod
    def get_vec_dtype(dtype_vec: np.dtype, dtype_scalar: np.dtype) -> np.dtype:
        if number_vec_elements_of_cl_type(dtype_vec) == 1:
            return dtype_scalar
        else:
            c_name = '{}{}'.format(c_name_from_dtype(dtype_scalar), number_vec_elements_of_cl_type(dtype_vec))
            return getattr(Types, c_name)

    @staticmethod
    def array_indexing_for_vec_type(array: str, index: str, dtype: np.dtype):
        """
        https://stackoverflow.com/questions/24746221/using-a-vector-as-an-array-index
        e.g.
        uchar4 keys = (uchar4)(5, 0, 2, 6);
        uint4 results = (uint4)(data[keys.s0], data[keys.s1], data[keys.s2], data[keys.s3]);

        :param dtype:
        :param array:
        :param index:
        :return:
        """
        if number_vec_elements_of_cl_type(dtype) == 1:
            return '{array_name}[{index_name}]'.format(array_name=array, index_name=index)
        else:
            return '({c_type_name})({vector})'.format(c_type_name=c_name_from_dtype(dtype),
                                                      vector=', '.join(
                                                          ['{array_name}[{index_name}.s{i_vec_element}]'.format(
                                                              array_name=array,
                                                              index_name=index,
                                                              i_vec_element=VEC_INDICES[i])
                                                              for i in range(number_vec_elements_of_cl_type(dtype))]))

    @staticmethod
    def command_const_vec_type(param: Union[str, float, int], dtype: np.dtype) -> str:
        """
        param = 1.5, dtype=ClTypes.float -> 'convert_float(1.5)'
        param = 1.5, dtype=ClTypes.float2 -> '(float2)(convert_float(1.5), convert_float(1.5))

        :param param:
        :param dtype:
        :return:
        """
        if number_vec_elements_of_cl_type(dtype) == 1:
            return 'convert_{}({})'.format(c_name_from_dtype(dtype), param)
        else:
            dtype_c_name = c_name_from_dtype(scalar_type_from_vec_type(dtype))
            return '({})(({}))'.format(c_name_from_dtype(dtype),
                                       ', '.join(['convert_{}({})'.format(dtype_c_name,
                                                                          param)] * get_vec_size(dtype)))

    @staticmethod
    def command_vec_sum(var_name: str, dtype: np.dtype) -> str:
        """
        Cases:
        float var_name -> return 'var_name'
        float4 var_name -> return 'var_name.s0 + var_name.s1 + var_name.s2 + var_name.s3'
        :param var_name:
        :return:
        """
        if get_vec_size(dtype) == 1:
            return var_name
        else:
            return ' + '.join(
                ['{}.s{}'.format(var_name, VEC_INDICES[i]) for i in range(get_vec_size(dtype))])

    # todo: use splay method of pyopencl library instead
    # from pyopencl.array import splay
    # splay
    @staticmethod
    def _get_local_size_coalesced_last_dim(global_size, desired_wg_size):
        """
        E.g. global_size = (1000, 25) and desired_wg_size=64
        Then a local_size=(2,25) is returned for multiple reasons:

        The work group size must be equal or smaller than the desired work group size.
        We make the last local dimension is large as possible (cannot exceed global size of last dimension).
        If possible the second last dimension is set to a value larger than 1, such that we get close to our desired
        work group size.

        :param global_size:
        :param desired_wg_size:
        :return:
        """

        local_size = [1] * len(global_size)
        for i_dim in range(1, len(global_size) + 1):
            if global_size[-i_dim] * local_size[-i_dim + 1] < desired_wg_size:
                local_size[-i_dim] = global_size[-i_dim]
            else:
                local_size[-i_dim] = np.max([i for i in range(1, desired_wg_size + 1)
                                             if (global_size[-i_dim] / i).is_integer() and
                                             i * local_size[-i_dim + 1] <= desired_wg_size])
        if np.product(local_size) < desired_wg_size:
            pass
            # res = inspect.stack()
            # logging.info(f'Local size {local_size} is suboptimal for desired work group size of {desired_wg_size}. '
            #              f'For best performance increase the global size of the most inner dimension, until it is '
            #              f'divisible by {desired_wg_size}. \n'
            #              f'More information: '
            #              f'https://stackoverflow.com/questions/3957125/questions-about-global-and-local-work-size')
        return tuple(local_size)
        # return None

    @staticmethod
    def get_local_size_coalesced_last_dim(global_size, context: Context):
        """
        If global size is no multiple of the local size, according to following link it should not work.
        https://community.khronos.org/t/opencl-ndrange-global-size-local-size/4167

        However (only for AMD GPU), simple tests have shown that it still works. Therefore this class gives a local size, where the global
        size is not necessarily a multiple.

        :param global_size:
        :param context:
        :return:
        """
        desired_wg_size = 4 * context.device.global_mem_cacheline_size
        return Helpers._get_local_size_coalesced_last_dim(global_size, desired_wg_size)
