from threading import Thread
import time
import sys


class DummyJtop(Thread):
    def __init__(self, interval=0.5):
        self.Ts = interval
        self.cpu = {
            'CPU1': {'val': 1},
            'CPU2': {'val': 1},
            'CPU3': {'val': 1},
            'CPU4': {'val': 1}
        }
        self.gpu = {'val': 1}
        self.ram = {'use': 1, 'tot': 1}

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        pass


class JtopAdapter(Thread):
    def __init__(self, interval):
        self.interval = interval
        self.start_time = None
        self.jtop_inst = DummyJtop(interval=interval)
        if sys.platform == 'linux':
            from jtop import jtop
            self.jtop_inst = jtop(interval=interval)
        self.values = []
        self.stopped = True
        super().__init__()

    def __enter__(self):
        self.stopped = False
        self.start_time = time.time()
        self.jtop_inst.__enter__()
        self.start()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.stopped = True
        self.jtop_inst.__exit__(exception_type, exception_value, traceback)

    def read_stats(self):
        self.values.append((
            self.jtop_inst.cpu['CPU1']['val'],
            self.jtop_inst.cpu['CPU2']['val'],
            self.jtop_inst.cpu['CPU3']['val'],
            self.jtop_inst.cpu['CPU4']['val'],
            self.jtop_inst.gpu['val'],
            self.jtop_inst.ram['use']/self.jtop_inst.ram['tot'],
            time.time() - self.start_time
        ))

    def run(self):
        while not self.stopped:
            self.read_stats()
            time.sleep(self.interval)

    def export_stats(self):
        return self.values, [
            'CPU1', 'CPU2', 'CPU3', 'CPU4', 'GPU', 'RAM', 'time']
