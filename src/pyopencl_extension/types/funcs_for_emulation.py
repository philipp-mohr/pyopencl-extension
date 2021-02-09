import numpy as np
import math as math

from pyopencl_extension.types.utilities_np_cl import c_to_np_type_name_catch, ClTypes, is_vector_type
from pyopencl_extension.types.auto_gen.types_for_emulation import *

cfloat_t = np.dtype('complex64').type
cdouble_t = np.dtype('complex128').type
void = np.dtype('void').type


class CArray(np.ndarray):
    # https://numpy.org/doc/stable/user/basics.subclassing.html

    def __new__(cls, shape, dtype, *args, **kwargs):
        if is_vector_type(dtype):
            return CArrayVec(shape, dtype, *args, **kwargs)
        else:
            return super(CArray, cls).__new__(cls, shape, dtype, *args, **kwargs)
        # here, attributes can be added to numpy class instance_

    def __add__(self, other):
        ary = self[other:]
        if hasattr(self, 'org'):  # keeps pointer to original allocated memory space
            ary.org = self.org
        else:
            ary.org = self
        return ary

    @property
    def np(self):
        return np.array(self)

    # todo: decrement a pointer
    # def __sub__(self, other):
    #     return self[other:]

    def __set__(self, instance, value):
        pass

    def __setitem__(self, instance, value):
        super().__setitem__(instance, value)


class CArrayVec(np.ndarray):
    def __new__(cls, *args, **kwargs):
        instance_ = super(CArrayVec, cls).__new__(cls, *args, **kwargs)
        instance_.vec_size = len(instance_.dtype.descr)
        return instance_

    def __add__(self, other):
        pass

    def __setitem__(self, instance, value):
        for i in range(self.vec_size):
            field = f's{i}'
            self[instance].val[field] = value.val[field]

    def __getitem__(self, item):
        res = super(CArrayVec, self).__getitem__(item)
        return VecVal(res)


sign = lambda x: np.sign(x)
fabs = lambda x: np.abs(x)
log2 = lambda x: np.log2(x)
min = lambda x, y: np.min([x, y])
sqrt = lambda x: x ** 0.5
exp = lambda x: np.exp(x)
log = lambda x: np.log(x)
pow = lambda x, y: x ** y
barrier = lambda x: x
CLK_GLOBAL_MEM_FENCE = None
select = lambda a, b, c: b if c else a
any = lambda x: np.any(x)
init_array = lambda size, type_c: np.ones((size,), dtype=c_to_np_type_name_catch(type_c)).view(CArray)


# init_array = lambda size, type_c: np.empty((size,), dtype=c_to_np_type_name_catch(type_c))


def add_sat(x, y):
    res = np.add(x, y, dtype=int)
    if res > np.iinfo(x.dtype).max:
        return x.dtype.type(np.iinfo(x.dtype).max)
    elif res < np.iinfo(x.dtype).min:
        return x.dtype.type(np.iinfo(x.dtype).min)
    else:
        return res


def set_real(ary, idx, value):
    ary[idx] = value + 1j * ary[idx].imag


def set_imag(ary, idx, value):
    ary[idx] = ary[idx].real + 1j * value


# https://www.askpython.com/python/python-modulo-operator-math-fmod
def c_modulo(dividend, divisor):
    if type(dividend).__module__ == 'numpy':
        return dividend.dtype.type(math.fmod(dividend, divisor))
    elif type(dividend) == int:
        return int(math.fmod(dividend, divisor))
    else:
        raise NotImplementedError(f'modulo division not implemented for divendend of type {type(dividend)}')
