from pyopencl.array import zeros
from pyopencl_extension import ClKernel, KnlArgBuffer, KnlArgScalar, ClInit, ClTypes
import numpy as np

cl_init = ClInit()
ary = zeros(cl_init.queue, (10,), ClTypes.short)

knl = ClKernel('some_operation',
               {'buff': KnlArgBuffer(ary),
                'number': KnlArgScalar(ClTypes.short, 3)},
               """
                buff[get_global_id(0)] = get_global_id(0) + 3;
               """,
               global_size=ary.shape).compile(cl_init)
knl()
assert np.allclose(ary.get(), np.arange(10)+3)
