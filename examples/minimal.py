from pyopencl_extension import Kernel, Global, Scalar, Types, zeros
import numpy as np

ary = zeros((10,), Types.short)

knl = Kernel('some_operation',
             {'ary': Global(ary),  # notice that ary is set as default argument
              'number': Scalar(Types.short)},
             """
                ary[get_global_id(0)] = get_global_id(0) + number;
             """,
             global_size=ary.shape).compile(emulate=True)
knl(number=3)
assert np.allclose(ary.get(), np.arange(10) + 3)
