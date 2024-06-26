from abus_classification.datasets import TDSC, TDSCTumors


def test_tdsc():
    dataset = TDSC()
    assert len(dataset) == 100    
    
def test_tumors():
    dataset = TDSCTumors(path="./data/tdsc")
    assert len(dataset) == 100
    x,m,y = dataset[0]
    a,b,c = x.shape
    
    assert a == 114
    assert b == 256
    assert c == 35
