
pyopencl-extension
==========================

This package builds on top of pyopencl by providing an elegant programming framework and debugging capabilities.

What makes pyopencl-extension special?
   * Build on top of pyopencl which can increase performance significantly.
   * Usage of this framework forces consistent code when programming for GPU.
   * Allows debugging of OpenCl-Programs through kernel emulation in Python using a visual debugger (tested with Pycharm).
   * OpenCl emulation allows to find out-of-bounds array indexing easily.
   * Integrated profiling feature give quick overview over performance bottlenecks.
   * ...


Installation
=============

When automatic installation of pyopencl fails (happen using Windows):

    1. Download appropriate .whl binary from `PyOpenCl binaries for Windows <https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl>`_

    2. Make sure to have `proper <https://streamhpc.com/blog/2015-03-16/how-to-install-opencl-on-windows/>`_ OpenCl driver for your graphics card installed.

    3. Run :code:`pip install "pyopencl-X-cpX-cpX-X.whl"` in terminal, where X must be replaced with corresponding version.

Usage
-----
One very simple example is given below. More advanced and useful scenarios will be added in the future.


.. code-block:: python

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