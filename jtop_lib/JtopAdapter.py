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
            'CPU4': {'val': 1},
            'CPU5': {'val': 1},
            'CPU6': {'val': 1}
        }
        self.gpu = {'val': 1}
        self.ram = {'use': 1, 'tot': 1}
        self.disk = {'total': 1, 'used': 1}
        self.stats = {
            'CPU1': 1, 'CPU2': 1, 'CPU3': 1,
            'CPU4': 1, 'CPU5': 1, 'CPU6': 1,
            'GPU': 1, 'RAM': 1, 'SWAP': 1,
            'power cur': 1, 'power avg': 1
        }

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        pass


class JtopAdapter(Thread):
    def __init__(self, interval, cpu={}, gpu={}, ram={}, stats={}, disk={}):
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
            self.jtop_inst.ram['use'],
            self.jtop_inst.ram['tot'],
            self.jtop_inst.stats['SWAP'],
            self.jtop_inst.disk['used'],
            self.jtop_inst.disk['total'],
            time.time() - self.start_time
        ))

    def run(self):
        while not self.stopped:
            self.read_stats()
            time.sleep(self.interval)

    def export_stats(self):
        return self.values, [
            'CPU1', 'CPU2', 'CPU3', 'CPU4', 'GPU', 'RAM_use',
            'RAM_tot', 'swap', 'disk_use', 'disk_tot', 'time'
        ]
