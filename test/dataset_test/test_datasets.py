from abus_classification.datasets import TDSC, Tumors


def test_tdsc():
    dataset = TDSC()
    assert len(dataset) == 100    
    
