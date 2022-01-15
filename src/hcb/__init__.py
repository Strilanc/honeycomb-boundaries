import stim

_f = stim.Circuit.__repr__
stim.Circuit.__repr__ = lambda e: _f(e)
