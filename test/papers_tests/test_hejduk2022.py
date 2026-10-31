import numpy as np
import torch

from abus_classification.papers.hejduk2022 import HejdukCNN, train_model
from abus_classification.papers.hejduk2022.model import HEIGHT, WIDTH
from abus_classification.papers.hejduk2022.pipeline import (LESION, MODELS, NORMAL, BACKGROUND, affine, augment, class_weights,
                                                            detect, predict, stepwise, to_image, training_set, volume_crops,
                                                            volume_maps, window_share, windows)


def test_model_layers_output_and_parameters():
    net = HejdukCNN().eval()
    convs = [m for m in net.modules() if isinstance(m, torch.nn.Conv2d)]
    pools = [m for m in net.modules() if isinstance(m, torch.nn.MaxPool2d)]
    assert len(convs) == 13 and len(pools) == 5 and net(torch.zeros(1, 1, HEIGHT, WIDTH)).shape == (1, 2)
    # VGG-16 convolutions on one input channel: 14,714,688 - 2 * 64 * 9 for the first layer's two missing input channels
    assert sum(p.numel() for p in net.features.parameters()) == 14_714_688 - 2 * 64 * 9
    assert sum(p.numel() for p in net.fc.parameters()) == (512 * 5 * 4 + 1) * 4096 + 4097 * 4096 + 4097 * 2


def test_windows_are_centred_and_give_the_paper_grid():
    image = torch.zeros(600, 600, dtype=torch.uint8)
    image[305, 15] = 200                                              # window row 30, column 1 at stride 10
    w = windows(image)
    assert w.shape == (3600, HEIGHT, WIDTH)
    assert w[30 * 60 + 1, HEIGHT // 2, WIDTH // 2] == 200
    assert windows(image, 20).shape == (900, HEIGHT, WIDTH)


def test_window_share_counts_pixels_and_padding():
    share = window_share(torch.ones(600, 600, dtype=torch.uint8))
    assert share.shape == (60, 60) and abs(share[30, 30] - 1) < 1e-6
    # the corner window is centred on (5, 5): 96 of 181 rows and 81 of 151 columns are inside the image
    assert abs(share[0, 0] - 96 * 81 / (181 * 151)) < 1e-6


def test_crops_by_kind():
    vol = np.full((100, 100, 40), 80, np.uint8)
    vol[:, :, :10] = 0                                                # empty slices are background
    mask = np.zeros_like(vol)
    mask[40:60, 40:60, 20:30] = 1
    vol[mask > 0] = 200
    crops, kinds = volume_crops(vol, mask, np.random.default_rng(0))
    assert crops.shape[1:] == (HEIGHT, WIDTH) and {(k, (kinds == k).sum()) for k in (LESION, NORMAL, BACKGROUND)} == {(LESION, 5), (NORMAL, 5), (BACKGROUND, 5)}
    assert crops[kinds == LESION][:, HEIGHT // 2, WIDTH // 2].min() == 200    # centred on the lesion
    assert crops[kinds == BACKGROUND].mean() < 26 and (crops[kinds == NORMAL] != 200).all()


def test_each_model_gets_its_classes():
    data = {"split": np.array([0, 0, 1, 1, 2]), "kind": np.array([BACKGROUND, NORMAL, LESION, LESION, LESION]), "y": np.array([1, 1, 0, 1, 1])}
    for model, want_idx, want_y in ((1, [0, 1, 2, 3], [0, 1, 1, 1]), (2, [1, 2, 3], [0, 1, 1]), (3, [2, 3], [0, 1])):
        idx, y = training_set(data, model)
        assert list(idx) == want_idx and list(y) == want_y              # the test crop is never used
    assert [m["epochs"] for m in MODELS.values()] == [120, 120, 160]


def test_class_weights_by_hand():
    assert np.allclose(class_weights(np.array([0, 1, 1, 1])), [2, 2 / 3])


def test_affine_identity_and_flip():
    zero, one = torch.zeros(1), torch.ones(1)
    x = torch.arange(4.).reshape(1, 1, 2, 2)
    sample = lambda t: torch.nn.functional.grid_sample(x, torch.nn.functional.affine_grid(t, x.shape, align_corners=False), align_corners=False)
    assert torch.allclose(sample(affine(zero, zero, one, zero, zero, one, 1.0)), x)
    assert torch.allclose(sample(affine(zero, zero, one, zero, zero, -one, 1.0)), x.flip(3))


def test_augmentation_keeps_shape_and_constant_images():
    x = torch.full((3, 1, HEIGHT, WIDTH), 0.5)
    assert torch.allclose(augment(x), x)


class Fixed(torch.nn.Module):
    """Accepts a window when its first pixel is at least `level`."""

    def __init__(self, level):
        super().__init__()
        self.level = level

    def forward(self, x):
        return torch.stack([torch.zeros(len(x)).to(x), (x[:, 0, 0, 0] * 255 - self.level) * 10], 1)


def test_later_models_only_see_accepted_windows():
    x = torch.zeros(3, HEIGHT, WIDTH, dtype=torch.uint8)
    x[1, 0, 0], x[2, 0, 0] = 10, 20
    maps = stepwise([Fixed(5), Fixed(15), Fixed(0)], x)
    assert maps[0, 0] < 0.5 < maps[0, 1] and maps[1, 0] == 0 and maps[1, 1] < 0.5 < maps[1, 2]
    assert maps[2, 0] == 0 and maps[2, 1] == 0 and maps[2, 2] > 0.5


def test_volume_maps_shape_is_the_papers():
    vol = np.zeros((90, 80, 8), np.uint8)
    assert volume_maps([Fixed(0)] * 3, vol, stride=100, slice_step=2).shape == (4, 3, 6, 6)


def test_boxes_map_back_to_the_volume():
    maps = np.zeros((10, 3, 60, 60), np.float32)
    maps[2:4, 1, 30:32, 10:15] = [[0.6], [0.9]]
    box, score = detect(maps, (900, 300, 100), stride=10, slice_step=5)[0]
    # axis 2: slices 10-20; axis 1: rows 300-320 x 0.5; axis 0: columns 100-150 x 1.5
    assert np.allclose(box, (15, 155, 187.5, 10, 10, 75)) and abs(score - 0.9) < 1e-6


def test_tiny_net_overfits_and_finished_model_is_not_trained_again(tmp_path, monkeypatch):
    monkeypatch.setitem(MODELS, 3, dict(classes=None, images=16, epochs=40))
    monkeypatch.setattr("abus_classification.papers.hejduk2022.pipeline.augment", lambda x: x)
    n = 8
    crops = np.zeros((n, HEIGHT, WIDTH), np.uint8)
    crops[n // 2:] = 200
    data = {"crops": crops, "kind": np.full(n, LESION), "split": np.zeros(n, int), "y": np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)]}

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(1, 2)

        def forward(self, x):
            return self.fc(x.mean((1, 2, 3))[:, None])

    net = train_model(data, 3, 0, tmp_path, lr=0.5, batch=4, make=Tiny)
    p = predict(net, torch.as_tensor(crops))
    assert p[:n // 2].max() < 0.5 < p[n // 2:].min()
    saved = tmp_path / "hejduk2022_model3_lr0.5_seed0.pt"
    before = saved.read_bytes()
    train_model(data, 3, 0, tmp_path, lr=0.5, batch=4, make=Tiny)
    assert saved.read_bytes() == before and (tmp_path / "hejduk2022_model3_lr0.5_seed0_history.csv").exists()


def test_to_image_axes():
    plane = np.zeros((100, 50), np.uint8)
    plane[:, 25:] = 90                                                # deep half: depth becomes the rows
    image = to_image(plane)
    assert image.shape == (600, 600) and image[590, 300] == 90 and image[10, 300] == 0


def tiny_setup(monkeypatch):
    monkeypatch.setitem(MODELS, 3, dict(classes=None, images=16, epochs=40))
    monkeypatch.setattr("abus_classification.papers.hejduk2022.pipeline.augment", lambda x: x)
    n = 16
    crops = np.zeros((n, HEIGHT, WIDTH), np.uint8)
    crops[1::2] = 200
    split = np.r_[np.zeros(8, int), np.ones(8, int)]
    return {"crops": crops, "kind": np.full(n, LESION), "split": split, "y": np.arange(n) % 2}


class TinyMean(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1, 2)

    def forward(self, x):
        return self.fc(x.mean((1, 2, 3))[:, None])


def test_early_stopping_uses_validation_and_stops(tmp_path, monkeypatch):
    data = tiny_setup(monkeypatch)
    train_model(data, 3, 0, tmp_path, lr=0.5, batch=4, make=TinyMean, early_stopping=True, patience=2)
    lines = (tmp_path / "hejduk2022_model3_es_lr0.5_seed0_history.csv").read_text().splitlines()
    assert "val_auc" in lines[0] and len(lines) - 1 < 40
    assert not (tmp_path / "hejduk2022_model3_es_lr0.5_seed0.ckpt").exists() and (tmp_path / "hejduk2022_progress.log").exists()


def test_stops_at_the_deadline_and_resumes(tmp_path, monkeypatch):
    data = tiny_setup(monkeypatch)
    assert train_model(data, 3, 0, tmp_path, lr=0.5, batch=4, make=TinyMean, deadline=0) is None
    net = TinyMean()
    opt = torch.optim.SGD(net.parameters(), lr=0.5)
    state = dict(epoch=2, best=None, best_auc=-1.0, best_epoch=0, since=0, history=[dict(epoch=1), dict(epoch=2)])
    torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), plateau=torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max").state_dict(),
                    numpy=np.random.get_state(), state=state), tmp_path / "hejduk2022_model3_lr0.5_seed0.ckpt")
    train_model(data, 3, 0, tmp_path, lr=0.5, batch=4, epochs=4, make=TinyMean)
    history = (tmp_path / "hejduk2022_model3_lr0.5_seed0_history.csv").read_text().splitlines()[1:]
    assert [int(l.split(",")[0]) for l in history] == [1, 2, 3, 4]
    assert "resumed after epoch 2" in (tmp_path / "hejduk2022_progress.log").read_text()
