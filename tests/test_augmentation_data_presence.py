from src.augmentation_data import musan_present, rirs_present


def test_musan_presence_accepts_canonical_nested_layout(tmp_path):
    base = tmp_path / "musan"
    noise = base / "noise" / "free-sound"
    music = base / "music" / "fma"
    noise.mkdir(parents=True)
    music.mkdir(parents=True)
    (noise / "noise.wav").write_bytes(b"placeholder")
    (music / "music.wav").write_bytes(b"placeholder")

    assert musan_present(base)


def test_musan_presence_requires_both_noise_and_music(tmp_path):
    noise = tmp_path / "musan" / "noise" / "free-sound"
    noise.mkdir(parents=True)
    (noise / "noise.wav").write_bytes(b"placeholder")

    assert not musan_present(tmp_path / "musan")


def test_rirs_presence_accepts_flat_layout(tmp_path):
    rirs = tmp_path / "rirs"
    rirs.mkdir()
    (rirs / "room.wav").write_bytes(b"placeholder")

    assert rirs_present(rirs)
