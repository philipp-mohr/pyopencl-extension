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

    def __add__(self, other):
        return self._perform_on_each_element(other, '__add__')

    def __mul__(self, other):
        return self._perform_on_each_element(other, '__mul__')

    def __sub__(self, other):
        return self._perform_on_each_element(other, '__sub__')


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
        a = self._type(*args)
        return VecVal(a)


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
