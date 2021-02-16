from pyopencl.array import zeros
from pyopencl_extension import Kernel, Global, Scalar, Thread, Types
import numpy as np

thread = Thread()
ary = zeros(thread.queue, (10,), Types.short)

knl = Kernel('some_operation',
             {'buff': Global(ary),
              'number': Scalar(Types.short(3))},
             """
                buff[get_global_id(0)] = get_global_id(0) + number;
               """,
             global_size=ary.shape).compile(thread, b_python=True)
knl()
assert np.allclose(ary.get(), np.arange(10) + 3)
