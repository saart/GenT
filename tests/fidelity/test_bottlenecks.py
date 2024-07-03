from fidelity.bottlenecks import build_child_ratio_mapping, compare_distributions


def test_build_child_ratio_mapping_happy_flow():
    """
        a
      /  \
     b    c
    """
    transactions = [
        {
            "nodesData": {
                "a": {"resource": {"name": "a"}, "duration": 200},
                "b": {"resource": {"name": "b"}, "duration": 100},
                "c": {"resource": {"name": "c"}, "duration": 50},
            },
            "graph": {"edges": [{"source": "a", "target": "b"}, {"source": "a", "target": "c"}]},
        }
    ]
    result = dict(build_child_ratio_mapping(transactions))
    assert result == {
        ("a", "b"): [0.5],
        ("a", "c"): [0.25],
    }



def test_build_child_ratio_mapping_happy_flow_2():
    """
        a
      /  \
     b    c
           \
            d
    """
    transactions = [
        {
            "nodesData": {
                "a": {"resource": {"name": "a"}, "duration": 200},
                "b": {"resource": {"name": "b"}, "duration": 100},
                "c": {"resource": {"name": "c"}, "duration": 50},
                "d": {"resource": {"name": "d"}, "duration": 50},
            },
            "graph": {"edges": [
                {"source": "a", "target": "b"},
                {"source": "a", "target": "c"},
                {"source": "c", "target": "d"}
            ]},
        }
    ]
    result = dict(build_child_ratio_mapping(transactions))
    assert result == {
        ("a", "b"): [0.5],
        ("a", "c"): [0.25],
        ("a", "d"): [0.25],
        ("c", "d"): [1.0],
    }



def test_build_child_ratio_mapping_happy_flow_diamond():
    """
        a
      /  \
     b    c
      \  /
       d
    """
    transactions = [
        {
            "nodesData": {
                "a": {"resource": {"name": "a"}, "duration": 200},
                "b": {"resource": {"name": "b"}, "duration": 100},
                "c": {"resource": {"name": "c"}, "duration": 50},
                "d": {"resource": {"name": "d"}, "duration": 10},
            },
            "graph": {"edges": [
                {"source": "a", "target": "b"},
                {"source": "a", "target": "c"},
                {"source": "c", "target": "d"},
                {"source": "b", "target": "d"}
            ]},
        }
    ]
    result = dict(build_child_ratio_mapping(transactions))
    assert result == {
        ("a", "b"): [0.5],
        ("a", "c"): [0.25],
        ("a", "d"): [0.05, 0.05],
        ("c", "d"): [0.2],
        ("b", "d"): [0.1],
    }

def test_compare_distributions():
    assert compare_distributions([1, 2, 3], [1, 2, 3]) == 0.
    assert compare_distributions([1, 2, 3], [3, 2, 1]) == 0.
    assert compare_distributions([1, 2, 3, 4], [3, 2, 5, 1]) < 0.3


def test_dont_count_async_child():
    transactions = [
        {
            "nodesData": {
                "a": {"resource": {"name": "a"}, "duration": 200, "startTime": 10},
                "b": {"resource": {"name": "b"}, "duration": 100, "startTime": 5},
            },
            "graph": {"edges": [
                {"source": "a", "target": "b"},
            ]},
        }
    ]
    result = dict(build_child_ratio_mapping(transactions))
    assert result == {}
