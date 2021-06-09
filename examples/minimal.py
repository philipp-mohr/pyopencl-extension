from pyopencl_extension import Kernel, Global, Scalar, Thread, Types, zeros
import numpy as np

thread = Thread()
ary = zeros(thread.queue, (10,), Types.short)

knl = Kernel('some_operation',
             {'ary': Global(ary),  # notice that ary is set as default argument
              'number': Scalar(Types.short)},
             """
                ary[get_global_id(0)] = get_global_id(0) + number;
             """,
             global_size=ary.shape).compile(thread, emulate=True)
knl(number=3)
assert np.allclose(ary.get(), np.arange(10) + 3)
