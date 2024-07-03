from fidelity.monitor import Success, count_successes
from fidelity.utils import ComparableForest, ComparableTree
from tests.fidelity.utils import build_tx


def test_total_success_happy_flow():
    raw_forest = ComparableForest(
        {
            "1": ComparableTree.build(1, build_tx(with_issue=True)),
            "2": ComparableTree.build(1, build_tx(with_issue=True)),
            "4": ComparableTree.build(1, build_tx(with_issue=False)),
            "5": ComparableTree.build(1, build_tx(with_issue=False)),
        }
    )
    generate_forest = ComparableForest(
        {
            "1": ComparableTree.build(1, build_tx(with_issue=True)),
            "3": ComparableTree.build(1, build_tx(with_issue=True)),
            "4": ComparableTree.build(1, build_tx(with_issue=False)),
            "5": ComparableTree.build(1, build_tx(with_issue=True)),
        }
    )
    assert count_successes(raw_forest, generate_forest) == {
        Success.alerted_correctly: 1,
        Success.monitor_failed: 1,
        Success.didnt_alert_correctly: 1,
        Success.monitor_miss_fired: 1,
    }
