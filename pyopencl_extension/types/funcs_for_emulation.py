import math as math
from abc import abstractmethod

from pyopencl_extension.types.auto_gen.types_for_emulation import *
from pyopencl_extension.types.utilities_np_cl import c_to_np_type_name_catch, is_vector_type, get_unsigned_dtype

cfloat_t = np.dtype('complex64').type
cdouble_t = np.dtype('complex128').type
void = np.dtype('void').type


class CPointerBase:
    @property
    @abstractmethod
    def np(self) -> tuple[np.ndarray, np.ndarray]:
        pass

    def __str__(self):
        return str(self.np[0])

    def __init__(self, size, type_c, mem=None, pos=0):
        self.type_c = type_c
        self.size = size
        self.memory = self._init_memory() if mem is None else mem
        self._validate_pos(pos)
        self.pos = pos

    @classmethod
    def from_np(cls, ary):
        return cls(size=ary.size, type_c=ary.dtype, mem=ary)

    def view(self, type_c):
        mem = self.memory.view(type_c)
        return self.__class__(type_c=type_c, size=mem.shape[0], mem=mem, pos=self.pos)

    def _validate_pos(self, pos):
        if not 0 <= pos < self.size:
            raise MemoryError('Out of memory bounds positioning of pointer')
        else:
            return pos

    def fill(self, val):
        self.memory.fill(val)

    def __add__(self, other):
        return self.__class__(size=self.size, type_c=self.type_c, mem=self.memory, pos=self.pos + other)

    def __sub__(self, other):
        return self.__class__(size=self.size, type_c=self.type_c, mem=self.memory, pos=self.pos - other)

    def __getitem__(self, item):
        return self.memory[self.pos + item]

    @abstractmethod
    def __setitem__(self, instance, value):
        pass

    def _init_memory(self):
        mem = np.empty((self.size,), dtype=self.type_c)
        mem[:] = 99999  # arbitrary number to indicate that element of array contains intial value
        return mem


class CPointer(CPointerBase):
    @property
    def np(self) -> tuple[np.ndarray, np.ndarray]:
        return self.memory[self.pos:], self.memory

    def __setitem__(self, instance, value):
        self.memory[self.pos + instance] = value


class CPointerVec(CPointerBase):
    @property
    def np(self):
        vector_element_dtype = dict(self.memory.dtype.fields)['s0'][0]
        vector_len = len(self.memory.dtype.descr)
        res = np.array(self.memory.view(vector_element_dtype)).reshape(self.memory.shape[0], vector_len)
        return res[self.pos:, :], res[:, :]

    def __setitem__(self, instance, value):
        vec_size = len(self.memory.dtype.descr)
        # get reference
        element = self.memory[instance]
        # element =self[instance] this returns a copy of the the element in memory and therefore an assignment has no effect
        # on original
        for i in range(vec_size):
            field = f's{i}'
            element[field] = value.val[field]
            # self[instance].val[field] = value.val[field]

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.memory[self.pos]
        else:
            res = self.memory[item + self.pos]
            return VecVal(res.copy())


sign = lambda x: np.sign(x)
fabs = lambda x: np.abs(x)
log2 = lambda x: np.log2(x)
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
        return CPointer(size, type_c)
    elif isinstance(type_c, TypeHandlerVec):
        return CPointerVec(size, type_c)
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
    return np.max([x, y])


def min(x, y):
    return np.min([x, y])


def fmax(x, y):
    return np.fmax(x, y)


def clamp(x, minval, maxval):
    return min(max(x, minval), maxval)
