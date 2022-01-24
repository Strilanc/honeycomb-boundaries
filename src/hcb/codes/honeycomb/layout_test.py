from hcb.codes.honeycomb.layout import HoneycombLayout
from hcb.tools.gen.noise import NoiseModel


def test_layout_properties():
    layout = HoneycombLayout.from_code_distance(distance=4,
                                                rounds=10,
                                                noisy_gate_set='EM3_v1',
                                                noise_level=0.001,
                                                tested_observable='EPR')
    assert layout.data_width == 4
    assert layout.data_height == 6
