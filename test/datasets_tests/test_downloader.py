from abus_classification import datasets


def test_tdsc():
    dataset = datasets.TDSC(path_to_dataset="./datasets/tdsc")