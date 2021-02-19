from typing import Tuple, Any

import numpy as np

from pyopencl_extension.types.auto_gen.cl_types import ClTypesScalar
from pyopencl_extension.types.utilities_np_cl import c_name_from_dtype, dtype_from_c_name, VEC_INDICES, \
    VEC_INDICES_XYZW, Types, VEC_ADDRESSING, scalar_type_from_vec_type
import pyopencl_extension.modifications_pyopencl.cltypes as tp


class VecVal:
    """
    This class is a container for simulating the behavior of OpenCl vector types when using Python and Numpy.

    E.g. when adding two number a and b of type float2, the addition operation is not implemented by default.
    THe implementation of addition operation is included in this class: Each element of the all vector elements of
    a and b are added separately to a new number which is then return as another instance of VecVal.

    Likewise other operation are implemented in this class.
    """

    def __init__(self, val):
        self.val = val
        self.type_handler = TypeHandlerVec(c_name_from_dtype(val.dtype))
        self.vec_size = len(val.dtype.descr)

    def _perform_on_each_element(self, other, operation):
        # e.g. 1) vec[0] *= (a, b); instead of 2) vec[0] *= (vectype)(a, b);
        # case 1) leads to erroneous behavior in opencl, therefore therefore error
        if not isinstance(other, VecVal):
            # other = self.type_handler(*other)
            raise ValueError('Input type is invalid')

        res = tuple([getattr(self.val[f's{i}'], operation)(other.val[f's{i}'])
                     for i in range(self.vec_size)])
        return self.type_handler(*res)

    vec_indices = [f's{idx}' for idx in VEC_INDICES] + VEC_INDICES_XYZW
    vec_special = VEC_ADDRESSING
    scalar_types_builtins = [int, float]
    scalar_types_cl = list(ClTypesScalar().__dict__.values())
    scalar_types = scalar_types_builtins + scalar_types_cl

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.vec_indices:
            self.val[name] = value
        elif name in self.vec_special and type(value) in self.scalar_types:
            for i in self.vec_special[name][str(len(self.val.dtype.names))]:
                self.val[f's{i}'] = value
        elif name in self.vec_special and type(value) == VecVal:
            raise NotImplementedError()
        else:
            super().__setattr__(name, value)

    def __getattr__(self, item):
        # https://stackoverflow.com/questions/2405590/how-do-i-override-getattr-in-python-without-breaking-the-default-behavior
        if item in self.vec_indices:
            return self.val[f'{item}']
        elif item in VEC_ADDRESSING.keys():
            # get type
            items = [self.val[f's{i}'] for i in VEC_ADDRESSING[item][f'{self.vec_size}']]
            typename = c_name_from_dtype(scalar_type_from_vec_type(self.val.dtype))
            if len(items) > 1:
                typename += str(len(items))
                return TypeHandlerVec(typename)(*items)
            else:
                return TypeHandlerScalar(typename)(*items)
        else:
            raise AttributeError

    # Alternative to intercept all magic methods:
    # https://stackoverflow.com/questions/9057669/how-can-i-intercept-calls-to-pythons-magic-methods-in-new-style-classes
    def __add__(self, other):
        return self._perform_on_each_element(other, '__add__')

    def __mul__(self, other):
        return self._perform_on_each_element(other, '__mul__')

    def __sub__(self, other):
        return self._perform_on_each_element(other, '__sub__')

    def __truediv__(self, other):
        return self._perform_on_each_element(other, '__truediv__')

    # todo: implement other build in when required
    def __abs__(self, *args, **kwargs):  # real signature unknown
        """ abs(self) """
        raise NotImplementedError()

    def __and__(self, *args, **kwargs):  # real signature unknown
        """ Return self&value. """
        raise NotImplementedError()

    def __bool__(self, *args, **kwargs):  # real signature unknown
        """ self != 0 """
        raise NotImplementedError()

    def __ceil__(self, *args, **kwargs):  # real signature unknown
        """ Ceiling of an Integral returns itself. """
        raise NotImplementedError()

    def __divmod__(self, *args, **kwargs):  # real signature unknown
        """ Return divmod(self, value). """
        raise NotImplementedError()

    def __eq__(self, *args, **kwargs):  # real signature unknown
        """ Return self==value. """
        raise NotImplementedError()

    def __ge__(self, *args, **kwargs):  # real signature unknown
        """ Return self>=value. """
        raise NotImplementedError()

    def __gt__(self, *args, **kwargs):  # real signature unknown
        """ Return self>value. """
        raise NotImplementedError()

    def __hash__(self, *args, **kwargs):  # real signature unknown
        """ Return hash(self). """
        raise NotImplementedError()

    def __index__(self, *args, **kwargs):  # real signature unknown
        """ Return self converted to an integer, if self is suitable for use as an index into a list. """
        raise NotImplementedError()

    def __int__(self, *args, **kwargs):  # real signature unknown
        """ int(self) """
        raise NotImplementedError()

    def __invert__(self, *args, **kwargs):  # real signature unknown
        """ ~self """
        raise NotImplementedError()

    def __le__(self, *args, **kwargs):  # real signature unknown
        """ Return self<=value. """
        raise NotImplementedError()

    def __lshift__(self, *args, **kwargs):  # real signature unknown
        """ Return self<<value. """
        raise NotImplementedError()

    def __lt__(self, *args, **kwargs):  # real signature unknown
        """ Return self<value. """
        raise NotImplementedError()

    def __mod__(self, *args, **kwargs):  # real signature unknown
        """ Return self%value. """
        raise NotImplementedError()

    def __neg__(self, *args, **kwargs):  # real signature unknown
        """ -self """
        raise NotImplementedError()

    def __ne__(self, *args, **kwargs):  # real signature unknown
        """ Return self!=value. """
        raise NotImplementedError()

    def __or__(self, *args, **kwargs):  # real signature unknown
        """ Return self|value. """
        raise NotImplementedError()

    def __pos__(self, *args, **kwargs):  # real signature unknown
        """ +self """
        raise NotImplementedError()

    def __pow__(self, *args, **kwargs):  # real signature unknown
        """ Return pow(self, value, mod). """
        raise NotImplementedError()

    def __radd__(self, *args, **kwargs):  # real signature unknown
        """ Return value+self. """
        raise NotImplementedError()

    def __rand__(self, *args, **kwargs):  # real signature unknown
        """ Return value&self. """
        raise NotImplementedError()

    def __rdivmod__(self, *args, **kwargs):  # real signature unknown
        """ Return divmod(value, self). """
        raise NotImplementedError()

    def __repr__(self, *args, **kwargs):  # real signature unknown
        """ Return repr(self). """
        raise NotImplementedError()

    def __rfloordiv__(self, *args, **kwargs):  # real signature unknown
        """ Return value//self. """
        raise NotImplementedError()

    def __rlshift__(self, *args, **kwargs):  # real signature unknown
        """ Return value<<self. """
        raise NotImplementedError()

    def __rmod__(self, *args, **kwargs):  # real signature unknown
        """ Return value%self. """
        raise NotImplementedError()

    def __rmul__(self, *args, **kwargs):  # real signature unknown
        """ Return value*self. """
        raise NotImplementedError()

    def __ror__(self, *args, **kwargs):  # real signature unknown
        """ Return value|self. """
        raise NotImplementedError()

    def __round__(self, *args, **kwargs):  # real signature unknown
        """
        Rounding an Integral returns itself.
        Rounding with an ndigits argument also returns an integer.
        """
        raise NotImplementedError()

    def __rpow__(self, *args, **kwargs):  # real signature unknown
        """ Return pow(value, self, mod). """
        raise NotImplementedError()

    def __rrshift__(self, *args, **kwargs):  # real signature unknown
        """ Return value>>self. """
        raise NotImplementedError()

    def __rshift__(self, *args, **kwargs):  # real signature unknown
        """ Return self>>value. """
        raise NotImplementedError()

    def __rsub__(self, *args, **kwargs):  # real signature unknown
        """ Return value-self. """
        raise NotImplementedError()

    def __rtruediv__(self, *args, **kwargs):  # real signature unknown
        """ Return value/self. """
        raise NotImplementedError()

    def __rxor__(self, *args, **kwargs):  # real signature unknown
        """ Return value^self. """
        raise NotImplementedError()

    def __sizeof__(self, *args, **kwargs):  # real signature unknown
        """ Returns size in memory, in bytes. """
        raise NotImplementedError()

    def __trunc__(self, *args, **kwargs):  # real signature unknown
        """ Truncating an Integral returns itself. """
        raise NotImplementedError()

    def __xor__(self, *args, **kwargs):  # real signature unknown
        """ Return self^value. """
        raise NotImplementedError()


class TypeHandlerScalar:
    """
    This class types scalar OpenCl types.
    """

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, arg):
        from pyopencl_extension.types.funcs_for_emulation import CArray
        if isinstance(arg, CArray):
            return arg.view(self.dtype)
        else:
            return np.dtype(self.dtype).type(arg)


class TypeHandlerVec:
    """
    This class types vector OpenCl types.
    """

    def __init__(self, name):
        self.dtype = dtype_from_c_name(name)
        self._type = eval(f'tp.make_{name}')

    def __call__(self, *args):
        from pyopencl_extension import CArray, CArrayVec
        if isinstance(args[0], CArray) or isinstance(args[0], CArrayVec):
            return args[0].view(self.dtype)
        elif isinstance(args[0], VecVal):
            return args[0]
        elif len(args) == 1:
            a = self._type(*(args * len(self.dtype.names)))
            return VecVal(a)
        elif isinstance(args, Tuple):
            a = self._type(*args)
            return VecVal(a)
        else:
            raise ValueError(f'Input not supported: {args=}')


def test_vec_val():
    import auto_gen.types_for_emulation as tp
    a = tp.long2(2, 2)
    a.even = 1
    assert a.val == tp.long2(1, 2).val
    a.odd = 3
    assert a.val == tp.long2(1, 3).val
    a.s0 = 3
    assert a.val == tp.long2(3, 3).val

    a = tp.long2(1, 2)
    b = tp.long2(1, 2)
    assert (a + b).val == tp.long2(2, 4).val
    assert (a - b).val == tp.long2(0, 0).val
    assert (a * b).val == tp.long2(1, 4).val
    assert tp.long2(1).val == tp.long2(1, 1).val


def test_vec_val_and_carray():
    from pyopencl_extension.types.funcs_for_emulation import CArray
    from pyopencl_extension import Types
    import pyopencl_extension.types.auto_gen.types_for_emulation as tp_emulation
    a = tp_emulation.long2(1, 2)
    b = tp_emulation.long2(1, 2)

    ary = CArray((10,), Types.long2)
    ary[0] = a + b
    assert ary[0].val == tp_emulation.long2(2, 4).val

    ary = CArray((10,), Types.long2)
    ary_typed = tp_emulation.long4(ary)
    assert ary_typed.dtype == Types.long4
