import math as math

from pyopencl_extension.types.auto_gen.types_for_emulation import *
from pyopencl_extension.types.utilities_np_cl import c_to_np_type_name_catch, is_vector_type, get_unsigned_dtype

cfloat_t = np.dtype('complex64').type
cdouble_t = np.dtype('complex128').type
void = np.dtype('void').type


class CArrayBase(np.ndarray):
    @property
    def np(self):
        pass


class CArray(CArrayBase):
    # https://numpy.org/doc/stable/user/basics.subclassing.html
    # not needed since we convert np.array a with a.view(CArrayVec) and call to __new__ is omitted
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


class CArrayVec(CArrayBase):
    # not needed since we convert np.array a with a.view(CArrayVec) and call to __new__ is omitted
    # def __new__(cls, *args, **kwargs):
    #     instance_ = super(CArrayVec, cls).__new__(cls, *args, **kwargs)
    #     instance_.vec_size = len(instance_.dtype.descr)
    #     return instance_

    def __add__(self, other):
        ary = self[other:]
        if hasattr(self, 'org'):  # keeps pointer to original allocated memory space
            ary.org = self.org
        else:
            ary.org = self
        return ary

    def __setitem__(self, instance, value):
        vec_size = len(self.dtype.descr)
        # get reference
        element = super(CArrayVec, self).__getitem__(instance)
        # element =self[instance] this returns a copy of the the element in memory and therefore an assignment has no effect
        # on origional
        for i in range(vec_size):
            field = f's{i}'
            element[field] = value.val[field]
            # self[instance].val[field] = value.val[field]

    def __getitem__(self, item):
        if isinstance(item, slice):
            return super(CArrayVec, self).__getitem__(item)
        else:
            res = super(CArrayVec, self).__getitem__(item)
            return VecVal(res.copy())

    @property
    def np(self):
        vector_element_dtype = dict(self.dtype.fields)['s0'][0]
        vector_len = len(self.dtype.descr)
        res = np.array(self.view(vector_element_dtype)).reshape(self.shape[0], vector_len)
        return res


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
INFINITY = np.inf


def init_array(size, type_c):
    if isinstance(type_c, TypeHandlerScalar):
        ary = np.empty((size,), dtype=c_to_np_type_name_catch(type_c.dtype))
        ary[:] = 99999  # arbitrary number to indicate that element of array contains intial value
        return ary.view(CArray)
    elif isinstance(type_c, TypeHandlerVec):
        ary = np.empty((size,), dtype=type_c.dtype)
        ary[:] = 99999  # arbitrary number to indicate that element of array contains intial value
        return ary.view(CArrayVec)
    else:
        raise ValueError(f'Array initialized with {str(type_c)} not supported')


# init_array = lambda size, type_c: np.empty((size,), dtype=c_to_np_type_name_catch(type_c))


def add_sat(x, y):
    res = np.add(x, y, dtype=int)
    if res > np.iinfo(x.dtype).max:
        return x.dtype.type(np.iinfo(x.dtype).max)
    elif res < np.iinfo(x.dtype).min:
        return x.dtype.type(np.iinfo(x.dtype).min)
    else:
        return res


def abs(x):
    return get_unsigned_dtype(x.dtype.type)(np.abs(x))


def abs_diff(x, y):
    return get_unsigned_dtype(x.dtype.type)(np.abs(x - y))


def set_real(ary, idx, value):
    ary[idx] = value + 1j * ary[idx].imag


def set_imag(ary, idx, value):
    ary[idx] = ary[idx].real + 1j * value


# https://www.askpython.com/python/python-modulo-operator-math-fmod
def c_modulo(dividend, divisor):
    if type(dividend).__module__ == 'numpy':
        return dividend.dtype.type(math.fmod(dividend, divisor))
    elif type(dividend) == int or type(dividend) == int_:
        return int(math.fmod(dividend, divisor))
    else:
        raise NotImplementedError(f'modulo division not implemented for divendend of type {type(dividend)}')


def max(x, y):
    return np.max(x, y)


def fmax(x, y):
    return np.fmax(x, y)
