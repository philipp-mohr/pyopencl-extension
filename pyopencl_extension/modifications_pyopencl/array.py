import time

import pyopencl as cl
import pyopencl.array as cl_array

from pyopencl_extension.modifications_pyopencl.command_queue import CommandQueue, get_current_queue


class Array(cl.array.Array):
    @classmethod
    def from_array(cls, array: cl_array.Array):
        if isinstance(array, cl_array.Array):
            return Array((a := array).queue, a.shape, a.dtype, order="C", allocator=a.allocator,
                         data=a.data, offset=a.offset, strides=a.strides, events=a.events)
        elif isinstance(array, Array):
            return array

    def set(self, ary, queue=None, async_=None, **kwargs):
        res = super().set(ary, queue, async_, **kwargs)
        self.add_latest_event('set')
        return res

    def get(self, queue=None, ary=None, async_=None, **kwargs):
        t0 = time.perf_counter_ns()
        res = super().get(queue, ary, async_, **kwargs)
        self.queue.t_ns.add_blocking(time.perf_counter_ns() - t0)
        self.add_latest_event('get')
        return res

    def add_latest_event(self, name):
        if len(self.events) > 0:
            # [self.queue.add_event(event, name) for event in self.events]
            self.queue.add_event(self.events[-1], name)

    def view(self, dtype=None):
        return self.from_array(super().view(dtype))


def to_device(ary, queue=None, allocator=None, async_=None, array_queue=cl_array._same_as_transfer, **kwargs):
    queue = get_current_queue() if queue is None else queue
    res = Array.from_array(cl_array.to_device(queue, ary, allocator, async_, array_queue, **kwargs))
    if len(res.events) > 0:
        queue.add_event(res.events[-1], 'to_device')
    return res


def empty(shape, dtype, cq: CommandQueue = None, order="C", allocator=None,
          data=None, offset=0, strides=None, events=None, _flags=None):
    cq = get_current_queue() if cq is None else cq
    res = Array.from_array(cl_array.Array(cq, shape, dtype, order, allocator))
    res.add_latest_event('empty')
    return res


def zeros(shape, dtype, queue: CommandQueue = None, order="C", allocator=None):
    queue = get_current_queue() if queue is None else queue
    res = Array.from_array(cl_array.zeros(queue, shape, dtype, order, allocator))
    res.add_latest_event('zeros')
    return res


def empty_like(ary, queue=cl_array._copy_queue, allocator=None):
    res = Array.from_array(cl_array.empty_like(ary, queue, allocator))
    res.add_latest_event('empty_like')
    return res


def zeros_like(ary):
    res = Array.from_array(cl_array.zeros_like(ary))
    res.add_latest_event('zeros_like')
    return res
