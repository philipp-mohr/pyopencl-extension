import re as re
from typing import Union, Callable

import numpy as np

__author__ = "piveloper"
__copyright__ = "26.03.2020, piveloper"
__version__ = "1.0"
__email__ = "piveloper@gmail.com"
__doc__ = """This script includes helpful functions to extended PyOpenCl functionality."""

from pyopencl_extension.types.auto_gen.cl_types import ClTypesVector
from pyopencl_extension.types.auto_gen.cl_types import _ClTypes

np_to_c_type_name = {
    'float16': 'half',
    'float32': 'float',
    'float64': 'double',
    'int64': 'long',
    'uint64': 'ulong',
    'int32': 'int',
    'uint32': 'uint',
    'int16': 'short',
    'uint16': 'ushort',
    'int8': 'char',
    'uint8': 'uchar',
    'complex64': 'cfloat_t',
    'complex128': 'cdouble_t',
    'void': 'void'
}

c_to_np_type_name = {v: k for k, v in np_to_c_type_name.items()}


def np_to_c_type_name_catch(np_name):
    try:
        return np_to_c_type_name[np_name]
    except:
        return np_name


def c_to_np_type_name_catch(c_name):
    try:
        return c_to_np_type_name[c_name]
    except:
        return c_name


class Types(_ClTypes):
    cfloat: Union[np.dtype, Callable] = np.dtype(np.complex64)
    cdouble: Union[np.dtype, Callable] = np.dtype(np.complex128)


VEC_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'B', 'C', 'D', 'E', 'F']


def get_vec_size(dtype: np.dtype) -> int:
    # if dtype is scalar type return 1
    if not hasattr(dtype, 'fields') or dtype.fields is None:
        return 1
    else:
        return len(dtype.names)


def c_name_from_dtype(dtype: np.dtype) -> str:
    """
    :param dtype: A numpy data type class
    :return: The C-Type corresponding to the numpy dtype
    """
    if not hasattr(dtype, 'fields') or dtype.fields is None:  # scalar type
        try:
            return np_to_c_type_name[dtype.name]
        except:
            return np_to_c_type_name[dtype.__name__]
    else:  # vector type
        n_dim = get_vec_size(dtype)
        name = np_to_c_type_name[dict(dtype.fields)[list(dtype.fields)[0]][0].name]
        return '{}{}'.format(name, n_dim)


def dtype_from_c_name(c_name: str):
    return getattr(Types, c_name)


def dtype_to_c_name(dtype):
    return np_to_c_type_name_catch(dict(dtype.fields)['s0'][0].name) + str(len(dtype.names))


def scalar_type_from_vec_type(dtype: np.dtype) -> np.dtype:
    if number_vec_elements_of_cl_type(dtype) == 1:
        return dtype
    else:
        c_vec_name = c_name_from_dtype(dtype)
        return getattr(Types, re.search(r'([a-z]+)([\d]+)', c_vec_name).group(1))


def number_vec_elements_of_cl_type(dtype: np.dtype) -> int:
    if dtype.names is None:
        return 1
    else:
        return len(dtype.names)
    # if 'double' in c_name_from_dtype(dtype):
    #     return int(dtype.itemsize / 8)
    # else:
    #     return dtype.itemsize


def match_vec_size(desired_type: np.dtype, match_vec_type: np.dtype) -> np.dtype:
    """

    :param desired_type:
    :param match_vec_type:
    :return: e.g. desired_type=ClTypes.float, match_vec_type=ClTypes.int4 -> return ClType.float4
    """
    desired_vec_size = get_vec_size(match_vec_type)
    desired_type_scalar = scalar_type_from_vec_type(desired_type)
    if desired_vec_size == 1:
        return desired_type_scalar
    else:
        c_name = '{}{}'.format(c_name_from_dtype(desired_type_scalar), desired_vec_size)
        return dtype_from_c_name(c_name)


def b_is_whole_number(t: np.dtype) -> bool:
    return scalar_type_from_vec_type(t) not in [Types.float, Types.double]


def match_integer_type_for_select(dtype: np.dtype):
    """
    When using overloadable select(a,b,c), the correct function is found by the types.
    E.g. for certain types like half or double this type can be either short or ushort.
    We must provide the type with a cast in order to avoid compilation error.
    select(half16 a, half16 b, short16 c);
    select(half16 a, half16 b, ushort16 c);
    match_integer_type_for_select(dtype=ClTypes.half16)->Return short16
    """
    dtype_scalar = scalar_type_from_vec_type(dtype)
    name = c_name_from_dtype(dtype_scalar)
    if name.startswith('u'):
        name = name[1:]
    conversions = {
        'char': 'char',
        'short': 'short',
        'int': 'int',
        'long': 'long',
        'float': 'int',
        'half': 'short',
        'double': 'long',
    }
    c_name_condition_type = conversions[name]
    # unsigned variant sometimes does not work, signed always works
    # if b_prefer_unsigned:
    #     c_name_condition_type ='u{}'.format(c_name_condition_type)
    dtype_condition = match_vec_size(dtype_from_c_name(c_name_condition_type), dtype)
    return dtype_condition


def scalar_to_vector_type_array(ary: np.ndarray, vec_size: int = 1):
    pass


def is_signed_integer_type(dtype: np.dtype) -> bool:
    scalar_dtype = scalar_type_from_vec_type(dtype)
    if scalar_dtype in [Types.char, Types.short, Types.int, Types.long]:
        return True
    else:
        return False


def is_complex_type(dtype: np.dtype):
    if dtype in [Types.cfloat, Types.cdouble]:
        return True
    else:
        return False


def is_vector_type(dtype: np.dtype):
    return dtype in list(ClTypesVector().__dict__.values())


def defines_generic_operations(cl_code, generic_type: np.dtype):
    if is_complex_type(generic_type):
        preample_cplx_operations = """
        #define MUL c${cplx_type}_mul
        #define ADD c${cplx_type}_add
        #define SUB c${cplx_type}_sub
        #define ABS c${cplx_type}_abs
        #define RMUL c${cplx_type}_rmul
        #define NEW c${cplx_type}_new
        #define CONJ c${cplx_type}_conj
        """.replace('${cplx_type}', 'float' if generic_type is Types.cfloat else 'double')
    else:
        preample_real_operations = """
        #define MUL c${cplx_type}_mul
        #define ADD c${cplx_type}_add
        #define SUB c${cplx_type}_sub
        #define ABS c${cplx_type}_abs
        #define RMUL c${cplx_type}_rmul
        #define NEW c${cplx_type}_new
        #define CONJ c${cplx_type}_conj
        """.replace('${cplx_type}', 'float' if generic_type is Types.cfloat else 'double')

    return cl_code
