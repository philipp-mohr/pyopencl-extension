import os
from typing import Union
from pyopencl_extension.types.utilities_np_cl import *
from pyopencl_extension import Thread, Path, Array
import re

__author__ = "piveloper"
__copyright__ = "05.02.2021, piveloper"
__version__ = "1.0"
__email__ = "piveloper@gmail.com"
__doc__ = """This module contains useful function when interacting with pyopencl_extension"""


class HashArray(Array):
    def __init__(self, *args, **kwargs):
        if isinstance(args[0], Array):
            a = args[0]
            super().__init__(a.queue, a.shape, a.dtype, order="C", allocator=a.allocator,
                             data=a.data, offset=a.offset, strides=a.strides, events=a.events, _flags=a.flags)
        else:
            super().__init__(*args, **kwargs)
        self.hash = hash(self.get().tobytes())

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.update_hash()

    def set(self, ary, queue=None, async_=None, **kwargs):
        res = super().set(ary, queue, async_, **kwargs)
        self.update_hash()
        return res

    def update_hash(self):
        self.hash = hash(self.get().tobytes())


class ClHelpers:
    # helper methods which can be useful in interplay with this framwork
    @staticmethod
    def _camel_to_snake(name):
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    @staticmethod
    def command_compute_address(n_dim: int) -> str:
        command = '0'
        for i in range(n_dim):
            offset = '1'
            for j in range(i + 1, n_dim):
                offset += '*get_global_size({})'.format(j)
            command += '+get_global_id({})*{}'.format(i, offset)
        return command

    # helpers for using vector types

    @staticmethod
    def get_vec_dtype(dtype_vec: np.dtype, dtype_scalar: np.dtype) -> np.dtype:
        if number_vec_elements_of_cl_type(dtype_vec) == 1:
            return dtype_scalar
        else:
            c_name = '{}{}'.format(c_name_from_dtype(dtype_scalar), number_vec_elements_of_cl_type(dtype_vec))
            return getattr(Types, c_name)

    @staticmethod
    def array_indexing_for_vec_type(array: str, index: str, dtype: np.dtype):
        """
        https://stackoverflow.com/questions/24746221/using-a-vector-as-an-array-index
        e.g.
        uchar4 keys = (uchar4)(5, 0, 2, 6);
        uint4 results = (uint4)(data[keys.s0], data[keys.s1], data[keys.s2], data[keys.s3]);

        :param dtype:
        :param array:
        :param index:
        :return:
        """
        if number_vec_elements_of_cl_type(dtype) == 1:
            return '{array_name}[{index_name}]'.format(array_name=array, index_name=index)
        else:
            return '({c_type_name})({vector})'.format(c_type_name=c_name_from_dtype(dtype),
                                                      vector=', '.join(
                                                          ['{array_name}[{index_name}.s{i_vec_element}]'.format(
                                                              array_name=array,
                                                              index_name=index,
                                                              i_vec_element=VEC_INDICES[i])
                                                              for i in range(number_vec_elements_of_cl_type(dtype))]))

    @staticmethod
    def command_const_vec_type(param: Union[str, float, int], dtype: np.dtype) -> str:
        """
        param = 1.5, dtype=ClTypes.float -> 'convert_float(1.5)'
        param = 1.5, dtype=ClTypes.float2 -> '(float2)(convert_float(1.5), convert_float(1.5))

        :param param:
        :param dtype:
        :return:
        """
        if number_vec_elements_of_cl_type(dtype) == 1:
            return 'convert_{}({})'.format(c_name_from_dtype(dtype), param)
        else:
            dtype_c_name = c_name_from_dtype(scalar_type_from_vec_type(dtype))
            return '({})(({}))'.format(c_name_from_dtype(dtype),
                                       ', '.join(['convert_{}({})'.format(dtype_c_name,
                                                                          param)] * get_vec_size(dtype)))

    @staticmethod
    def command_vec_sum(var_name: str, dtype: np.dtype) -> str:
        """
        Cases:
        float var_name -> return 'var_name'
        float4 var_name -> return 'var_name.s0 + var_name.s1 + var_name.s2 + var_name.s3'
        :param var_name:
        :return:
        """
        if get_vec_size(dtype) == 1:
            return var_name
        else:
            return ' + '.join(
                ['{}.s{}'.format(var_name, VEC_INDICES[i]) for i in range(get_vec_size(dtype))])

    # todo: use splay method of pyopencl library instead
    # from pyopencl.array import splay
    # splay
    @staticmethod
    def _get_local_size_coalesced_last_dim(global_size, desired_wg_size):
        """
        E.g. global_size = (1000, 25) and desired_wg_size=64
        Then a local_size=(2,25) is returned for multiple reasons:

        The work group size must be equal or smaller than the desired work group size.
        We make the last local dimension is large as possible (cannot exceed global size of last dimension).
        If possible the second last dimension is set to a value larger than 1, such that we get close to our desired
        work group size.

        :param global_size:
        :param desired_wg_size:
        :return:
        """

        local_size = [1] * len(global_size)
        for i_dim in range(1, len(global_size) + 1):
            if global_size[-i_dim] * local_size[-i_dim + 1] < desired_wg_size:
                local_size[-i_dim] = global_size[-i_dim]
            else:
                local_size[-i_dim] = np.max([i for i in range(1, desired_wg_size + 1)
                                             if (global_size[-i_dim] / i).is_integer() and
                                             i * local_size[-i_dim + 1] <= desired_wg_size])
        if np.product(local_size) < desired_wg_size:
            pass
            # res = inspect.stack()
            # logging.info(f'Local size {local_size} is suboptimal for desired work group size of {desired_wg_size}. '
            #              f'For best performance increase the global size of the most inner dimension, until it is '
            #              f'divisible by {desired_wg_size}. \n'
            #              f'More information: '
            #              f'https://stackoverflow.com/questions/3957125/questions-about-global-and-local-work-size')
        return tuple(local_size)
        # return None

    @staticmethod
    def get_local_size_coalesced_last_dim(global_size, thread: Thread):
        """
        If global size is no multiple of the local size, according to following link it should not work.
        https://community.khronos.org/t/opencl-ndrange-global-size-local-size/4167

        However (only for AMD GPU), simple tests have shown that it still works. Therefore this class gives a local size, where the global
        size is not necessarily a multiple.

        :param global_size:
        :param thread:
        :return:
        """
        desired_wg_size = 4 * thread.device.global_mem_cacheline_size
        return ClHelpers._get_local_size_coalesced_last_dim(global_size, desired_wg_size)
