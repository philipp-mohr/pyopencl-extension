from pytest import fixture

from pyopencl_extension import ClInit


@fixture(scope="session")
def cl_init():
    return ClInit()
