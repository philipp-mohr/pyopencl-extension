from pytest import fixture

from pyopencl_extension import Thread


@fixture(scope="session")
def thread():
    return Thread()
