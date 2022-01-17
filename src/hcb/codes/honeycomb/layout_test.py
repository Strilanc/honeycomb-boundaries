from hcb.codes.honeycomb.layout import HoneycombLayout


def test_layout_properties():
    layout = HoneycombLayout(data_width=4, data_height=12)
    assert layout.data_width == 4
    assert layout.data_height == 12
