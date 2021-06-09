import time
from dataclasses import dataclass

import numpy as np
import pyopencl as cl


@dataclass
class QueueProperties:
    DEFAULT: int = 0
    ON_DEVICE: int = 4
    ON_DEVICE_DEFAULT: int = 8
    OUT_OF_ORDER_EXEC_MODE_ENABLE: int = 1
    PROFILING_ENABLE: int = 2


@dataclass
class TimingsQueue:
    queue_created: int = 0
    compilation: int = 0
    blocking: int = 0
    profiling_finished: int = 0
    first_event_added: int = 0
    last_event_added: int = 0
    profiling_enabled: bool = True

    def __post_init__(self):
        self.queue_created = time.time_ns()

    def add_blocking(self, t):
        if self.profiling_enabled:
            self.blocking += t

    def add_compilcation(self, t):
        if self.profiling_enabled:
            self.compilation += t


class CommandQueue(cl.CommandQueue):
    def __init__(self, context, device=None, properties=0, max_len_events=1e6):
        super().__init__(context, device, properties)
        self.max_len_events = int(max_len_events)
        self.events = []
        self.profiling_enabled = True if self.properties == cl.command_queue_properties.PROFILING_ENABLE else False
        self.t_ns = TimingsQueue(profiling_enabled=self.profiling_enabled)

    def get_profiler(self) -> 'Profiling':
        self.t_ns.profiling_finished = time.time_ns()
        return Profiling(self)

    def add_event(self, event, name):
        if self.profiling_enabled:
            if len(self.events) == 0:
                self.t_ns.first_event_added = time.time_ns()
            if len(self.events) < self.max_len_events:
                self.events.append((name, event))
                self.t_ns.last_event_added = time.time_ns()
            else:
                raise ValueError('Forgot to disable profiling?')

    def finish(self):
        t0 = time.time_ns()
        super().finish()
        self.t_ns.add_blocking(time.time_ns() - t0)


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
        event_times = [_[1] for _ in self.list_cumulative_event_times_ms()]
        event_names_times = [item for item in zip(event_names, event_times)]
        t_ns = self.queue.t_ns
        t0 = self.events[0][1].profile.start - (t_ns.first_event_added - t_ns.queue_created)
        t1 = (_ := self.events[-1][1].profile).end + (t_ns.profiling_finished - t_ns.last_event_added -
                                                      (_.end - _.start))

        total_time_ms = (t_ns.profiling_finished - t_ns.queue_created) / 1e6
        time_blocking_ms = t_ns.blocking / 1e6  # time waiting for kernels/transfers to finish
        time_gross_python_ms = total_time_ms - time_blocking_ms

        starts = np.array([e[1].profile.start for e in self.events] + [t1])
        ends = np.array([t0] + [e[1].profile.end for e in self.events])
        time_python_net_ms = (starts - ends).sum() / 1e6 - time_blocking_ms

        event_names_times.append(('total opencl events', self.get_sum_execution_time()))
        event_names_times.append(('total overall program', total_time_ms))
        event_names_times.append(('opencl blocking', time_blocking_ms))
        # event_names_times.append(('python (gross)', time_gross_python_ms))
        event_names_times.append(('python (net)', time_python_net_ms))
        event_names_times.append(('compilation', t_ns.compilation / 1e6))

        y_pos = np.arange(len(event_names_times))
        ax.barh(y_pos, [i[1] for i in event_names_times], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([i[0] for i in event_names_times])
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Time ms')
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
