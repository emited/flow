
from collections import OrderedDict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.5f} ({:.5f})'.format(self.val, self.avg)


class AverageMeters(object):

    def __init__(self):
        self._meters = OrderedDict()

    def update(self, meter_dict, n=1):
        for name, val in meter_dict.items():
            if not name in self._meters:
                self._meters[name] = AverageMeter()
            self._meters[name].update(val, n)
        self._check_integrity()

    def _check_integrity(self):
        for i, (name, meter) in enumerate(self._meters.items()):
            if i == 0:
                tmpcount = meter.count
            elif tmpcount != meter.count:
                raise RuntimeError('Forgot to update meter ' + name +
                                   '. Meter has count {} instead of {}.'.format(meter.count, tmpcount))

    def names(self):
        return list(self._meters.keys())

    def val(self, name):
        return self._meters[name].val

    def avg(self, name):
        return self._meters[name].avg

    def vals(self):
        return OrderedDict([(name, meter.val) for name, meter in self._meters.items()])

    def avgs(self):
        return OrderedDict([(name, meter.avg) for name, meter in self._meters.items()])

    def __repr__(self):
        tmpstr = ''
        for name, meter in self._meters.items():
            tmpstr += name + ' ' + meter.__repr__() + '\t'
        return tmpstr
