import time
from dataclasses import dataclass

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_a


@dataclass
class QueueProperties:
    DEFAULT: int = 0
    ON_DEVICE: int = 4
    ON_DEVICE_DEFAULT: int = 8
    OUT_OF_ORDER_EXEC_MODE_ENABLE: int = 1
    PROFILING_ENABLE: int = 2


class CommandQueue(cl.CommandQueue):
    def __init__(self, context, device=None, properties=0, max_len_events=1e6):
        super().__init__(context, device, properties)
        self.max_len_events = int(max_len_events)
        self.events = []
        self.t_ns = {'queue_created': time.time_ns()}

    def get_profiler(self) -> 'Profiling':
        self.t_ns['profiler_started'] = time.time_ns()
        return Profiling(self)

    def add_event(self, event, name):
        if self.properties == cl.command_queue_properties.PROFILING_ENABLE:
            if len(self.events) == 0:
                self.t_ns['first_event_added'] = time.time_ns()
            if len(self.events) < self.max_len_events:
                self.events.append((name, event))
                self.t_ns['last_event_added'] = time.time_ns()
            else:
                raise ValueError('Forgot to disable profiling?')


class Profiling:
    def __init__(self, queue: CommandQueue):
        if queue.properties == cl.command_queue_properties.PROFILING_ENABLE:
            self.queue = queue
            self.events = queue.events
        else:
            raise ValueError('Profiling must be enabled using command_queue_properties.PROFILING_ENABLE')

    def get_event_names_time_ms(self):
        return [(event[0], 1e-6 * ((_ := event[1]).profile.end - _.profile.start)) for event in self.events]

    def list_cumulative_event_times_ms(self):
        knl_names = {_[0]: 0.0 for _ in self.get_event_names_time_ms()}
        for _ in self.get_event_names_time_ms():
            knl_names[_[0]] += _[1]
        indices = np.argsort([knl_names[_] for _ in knl_names])
        return [list(knl_names.items())[i] for i in np.flip(indices)]

    def get_sum_execution_time(self):
        return sum([_[1] for _ in self.list_cumulative_event_times_ms()])

    def show_histogram_cumulative_kernel_times(self):
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0.4)
        # Example data

        event_names = [_[0] for _ in self.list_cumulative_event_times_ms()]
        evemt_times = [_[1] for _ in self.list_cumulative_event_times_ms()]

        t_ns = self.queue.t_ns
        t0 = self.events[0][1].profile.start - (t_ns['first_event_added'] - t_ns['queue_created'])
        t1 = self.events[-1][1].profile.end + (t_ns['profiler_started'] - t_ns['last_event_added'])

        starts = np.array([e[1].profile.start for e in self.events]+[t1])
        ends = np.array([t0]+[e[1].profile.end for e in self.events])
        python_times_ms = (starts - ends).sum() / 1e6
        event_names.append('python')
        evemt_times.append(python_times_ms)

        y_pos = np.arange(len(event_names))
        ax.barh(y_pos, evemt_times, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(event_names)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Time ms')
        ax.set_title(f'Total time (python excluded): {self.get_sum_execution_time()} ms')
        plt.show()

    # todo: make event plot with rectangles corresponding to kernel times
    # def show_events(self):
    #     import numpy as np
    #     t0 = self.events[0][1].profile.start
    #     starts = np.array([[e[1].profile.start - t0 for e in self.events]])
    #     ends = np.array([[e[1].profile.end - t0 for e in self.events]])
    #     durations_ms = (ends-starts)/1e6
    #
    #     import matplotlib.pyplot as plt
    #     import matplotlib
    #     matplotlib.rcParams['font.size'] = 8.0
    #     plt.figure()
    #     # create a vertical plot
    #     time_center = starts  # np.array([[0.25, 1.0, 1.5]])
    #     time_length = durations_ms / 10  # np.array([[0.5, 0.3, 0.4]]) * 300
    #     plt.eventplot(time_center, colors='blue', lineoffsets=1,
    #                   linelengths=1, linewidths=time_length, orientation='horizontal')
    #     plt.show()


class Array(cl.array.Array):
    @classmethod
    def from_array(cls, array: cl_a.Array):
        if isinstance(array, cl_a.Array):
            return Array((a := array).queue, a.shape, a.dtype, order="C", allocator=a.allocator,
                         data=a.data, offset=a.offset, strides=a.strides, events=a.events)
        elif isinstance(array, Array):
            return array

    def set(self, ary, queue=None, async_=None, **kwargs):
        res = super().set(ary, queue, async_, **kwargs)
        self.add_latest_event('set')
        return res

    def get(self, queue=None, ary=None, async_=None, **kwargs):
        res = super().get(queue, ary, async_, **kwargs)
        self.add_latest_event('set')
        return res

    def add_latest_event(self, name):
        if len(self.events) > 0:
            self.queue.add_event(self.events[-1], name)

    def view(self, dtype=None):
        return self.from_array(super().view(dtype))


def to_device(queue, ary, allocator=None, async_=None,
              array_queue=cl_a._same_as_transfer, **kwargs):
    res = Array.from_array(cl_a.to_device(queue, ary, allocator, async_, array_queue, **kwargs))
    if len(res.events) > 0:
        queue.add_event(res.events[-1], 'to_device')
    return res


def empty(cq, shape, dtype, order="C", allocator=None,
          data=None, offset=0, strides=None, events=None, _flags=None):
    res = Array.from_array(cl_a.Array(cq, shape, dtype, order, allocator))
    res.add_latest_event('empty')
    return res


def zeros(queue: CommandQueue, shape, dtype, order="C", allocator=None):
    res = Array.from_array(cl_a.zeros(queue, shape, dtype, order, allocator))
    res.add_latest_event('zeros')
    return res


def empty_like(ary, queue=cl_a._copy_queue, allocator=None):
    res = Array.from_array(cl_a.empty_like(ary, queue, allocator))
    res.add_latest_event('empty_like')
    return res


def zeros_like(ary):
    res = Array.from_array(cl_a.zeros_like(ary))
    res.add_latest_event('zeros_like')
    return res
