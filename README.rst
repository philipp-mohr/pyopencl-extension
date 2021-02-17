
pyopencl-extension
==========================

This package extends PyOpenCl by providing an elegant programming framework and debugging capabilities.

What makes pyopencl-extension special?
   * Build on top of `PyOpenCl <https://pypi.org/project/pyopencl/>`_ which can increase performance significantly.
   * Usage of this framework forces consistent code when programming for GPU.
   * Allows debugging of OpenCl-Programs through kernel emulation in Python using a visual debugger (tested with Pycharm).
   * OpenCl emulation allows to find out-of-bounds array indexing easily.
   * Integrated profiling features give quick overview over performance bottlenecks.
   * ...

The project is in an early development stage and actively maintained.
For any feature requests/feedback/etc. you can get in touch via
`Github <https://github.com/piveloper/pyopencl-extension/issues>`_ or by E-Mail (piveloper@gmail.com).

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
                 global_size=ary.shape).compile(thread, emulate=False)
    knl()
    assert np.allclose(ary.get(), np.arange(10) + 3)

By setting the argument 'emulate=True' the kernel will be compiled in emulation mode. This mode creates a
file 'some_operation.py', which can be inspected using a visual debugger:

.. image:: https://i.imgur.com/1ftgLLV.png
    :width: 600

More advanced and useful example scenarios will be added in the future here.