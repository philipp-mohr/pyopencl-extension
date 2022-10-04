
pyopencl-extension
==========================

This package extends PyOpenCl by providing an object-oriented kernel programming framework and debugging capabilities.

What makes pyopencl-extension special?
   * Build on top of `PyOpenCl <https://pypi.org/project/pyopencl/>`_ which can increase performance significantly.
   * Usage of this framework forces consistent code when programming for GPU.
   * Allows debugging of OpenCl-Programs through kernel emulation in Python using a visual debugger (tested with Pycharm).
   * OpenCl emulation allows to find out-of-bounds array indexing more easily.
   * Integrated profiling features give quick overview over performance bottlenecks.

The project is in an early development stage and actively maintained.
For any feature requests/feedback/etc. you can get in touch via
`Github <https://github.com/philipp-mohr/pyopencl-extension/issues>`_ or by E-Mail (philipp.mohr@tuhh.de).

Installation
------------
Install this library with :code:`pip install pyopencl-extension`.

When automatic installation of PyOpenCl fails (happens when using Windows):

    1. Download appropriate .whl binary from `PyOpenCl binaries for Windows <https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl>`_

    2. Make sure to have `proper <https://streamhpc.com/blog/2015-03-16/how-to-install-opencl-on-windows/>`_ OpenCl driver for your graphics card installed.

    3. Run :code:`pip install "pyopencl-X-cpX-cpX-X.whl"` in terminal, where X must be replaced with corresponding version.

Usage
-----
One very simple example is given below.


.. code-block:: python

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

By setting the argument 'emulate=True' the kernel will be compiled in emulation mode. This mode creates a
file 'some_operation.py', which can be inspected using a visual debugger:

.. image:: https://i.imgur.com/Gfg9AtZ.png
    :width: 600

More advanced examples can by found in the `tests hosted on  Github <https://github.com/philipp-mohr/pyopencl-extension/tree/main/tests>`_.