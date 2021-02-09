from typing import Tuple, Any

import numpy as np
from pyopencl_extension.types.utilities_np_cl import c_name_from_dtype
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
        res = tuple([getattr(self.val[f's{i}'], operation)(other.val[f's{i}'])
                     for i in range(self.vec_size)])
        return self.type_handler(*res)

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
        pass

    def __and__(self, *args, **kwargs):  # real signature unknown
        """ Return self&value. """
        pass

    def __bool__(self, *args, **kwargs):  # real signature unknown
        """ self != 0 """
        pass

    def __ceil__(self, *args, **kwargs):  # real signature unknown
        """ Ceiling of an Integral returns itself. """
        pass

    def __divmod__(self, *args, **kwargs):  # real signature unknown
        """ Return divmod(self, value). """
        pass

    def __eq__(self, *args, **kwargs):  # real signature unknown
        """ Return self==value. """
        pass

    def __ge__(self, *args, **kwargs):  # real signature unknown
        """ Return self>=value. """
        pass

    def __gt__(self, *args, **kwargs):  # real signature unknown
        """ Return self>value. """
        pass

    def __hash__(self, *args, **kwargs):  # real signature unknown
        """ Return hash(self). """
        pass

    def __index__(self, *args, **kwargs):  # real signature unknown
        """ Return self converted to an integer, if self is suitable for use as an index into a list. """
        pass

    def __int__(self, *args, **kwargs):  # real signature unknown
        """ int(self) """
        pass

    def __invert__(self, *args, **kwargs):  # real signature unknown
        """ ~self """
        pass

    def __le__(self, *args, **kwargs):  # real signature unknown
        """ Return self<=value. """
        pass

    def __lshift__(self, *args, **kwargs):  # real signature unknown
        """ Return self<<value. """
        pass

    def __lt__(self, *args, **kwargs):  # real signature unknown
        """ Return self<value. """
        pass

    def __mod__(self, *args, **kwargs):  # real signature unknown
        """ Return self%value. """
        pass

    def __neg__(self, *args, **kwargs):  # real signature unknown
        """ -self """
        pass

    def __ne__(self, *args, **kwargs):  # real signature unknown
        """ Return self!=value. """
        pass

    def __or__(self, *args, **kwargs):  # real signature unknown
        """ Return self|value. """
        pass

    def __pos__(self, *args, **kwargs):  # real signature unknown
        """ +self """
        pass

    def __pow__(self, *args, **kwargs):  # real signature unknown
        """ Return pow(self, value, mod). """
        pass

    def __radd__(self, *args, **kwargs):  # real signature unknown
        """ Return value+self. """
        pass

    def __rand__(self, *args, **kwargs):  # real signature unknown
        """ Return value&self. """
        pass

    def __rdivmod__(self, *args, **kwargs):  # real signature unknown
        """ Return divmod(value, self). """
        pass

    def __repr__(self, *args, **kwargs):  # real signature unknown
        """ Return repr(self). """
        pass

    def __rfloordiv__(self, *args, **kwargs):  # real signature unknown
        """ Return value//self. """
        pass

    def __rlshift__(self, *args, **kwargs):  # real signature unknown
        """ Return value<<self. """
        pass

    def __rmod__(self, *args, **kwargs):  # real signature unknown
        """ Return value%self. """
        pass

    def __rmul__(self, *args, **kwargs):  # real signature unknown
        """ Return value*self. """
        pass

    def __ror__(self, *args, **kwargs):  # real signature unknown
        """ Return value|self. """
        pass

    def __round__(self, *args, **kwargs):  # real signature unknown
        """
        Rounding an Integral returns itself.
        Rounding with an ndigits argument also returns an integer.
        """
        pass

    def __rpow__(self, *args, **kwargs):  # real signature unknown
        """ Return pow(value, self, mod). """
        pass

    def __rrshift__(self, *args, **kwargs):  # real signature unknown
        """ Return value>>self. """
        pass

    def __rshift__(self, *args, **kwargs):  # real signature unknown
        """ Return self>>value. """
        pass

    def __rsub__(self, *args, **kwargs):  # real signature unknown
        """ Return value-self. """
        pass

    def __rtruediv__(self, *args, **kwargs):  # real signature unknown
        """ Return value/self. """
        pass

    def __rxor__(self, *args, **kwargs):  # real signature unknown
        """ Return value^self. """
        pass

    def __sizeof__(self, *args, **kwargs):  # real signature unknown
        """ Returns size in memory, in bytes. """
        pass

    def __trunc__(self, *args, **kwargs):  # real signature unknown
        """ Truncating an Integral returns itself. """
        pass

    def __xor__(self, *args, **kwargs):  # real signature unknown
        """ Return self^value. """
        pass


class TypeHandlerScalar:
    """
    This class types scalar OpenCl types.
    """

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, arg):
        return np.dtype(self.dtype).type(arg)


class TypeHandlerVec:
    """
    This class types vector OpenCl types.
    """

    def __init__(self, name):
        self._type = eval(f'tp.make_{name}')

    def __call__(self, *args):
        if isinstance(args[0], VecVal):
            return args[0]
        elif isinstance(args, Tuple):
            a = self._type(*args)
            return VecVal(a)
        else:
            raise ValueError(f'Input not supported: {args=}')


def test_vec_val():
    import cltypes_emulation as tp
    a = tp.long2(1, 2)
    b = tp.long2(1, 2)
    assert (a + b).val == tp.long2(2, 4).val
    assert (a - b).val == tp.long2(0, 0).val
    assert (a * b).val == tp.long2(1, 4).val


def test_vec_val_and_carray():
    from pyopencl_extension.funcs_for_cl_emulation import CArray
    from pyopencl_extension import ClTypes
    import cltypes_emulation as tp_emulation
    a = tp_emulation.long2(1, 2)
    b = tp_emulation.long2(1, 2)

    ary = CArray((10,), ClTypes.long2)
    ary[0] = a + b
    assert ary[0].val == tp_emulation.long2(2, 4).val
