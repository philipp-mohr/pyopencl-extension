from pytest import fixture

from pyopencl_extension.framework import ClInit


@fixture(scope="session")
def cl_init():
    return ClInit()
