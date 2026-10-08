import os

from clouds import home


def test_set_home_dir_overrides(monkeypatch, tmp_path):
    monkeypatch.delenv('CLOUDS_HOME', raising=False)
    monkeypatch.setattr(home, '_home_dir', None)
    home.set_home_dir(str(tmp_path))
    assert home.get_home_dir() == str(tmp_path)


def test_env_var_used_when_unset(monkeypatch, tmp_path):
    monkeypatch.setattr(home, '_home_dir', None)
    monkeypatch.setenv('CLOUDS_HOME', str(tmp_path))
    assert home.get_home_dir() == os.path.expanduser(str(tmp_path))


def test_default_cache_dir(monkeypatch):
    monkeypatch.setattr(home, '_home_dir', None)
    monkeypatch.delenv('CLOUDS_HOME', raising=False)
    assert home.get_home_dir() == os.path.expanduser(home.DEFAULT_CACHE_DIR)


def test_get_dataset_root_uses_home(monkeypatch, tmp_path):
    monkeypatch.setattr(home, '_home_dir', None)
    monkeypatch.delenv('CLOUDS_HOME', raising=False)
    monkeypatch.setattr(home.sys, 'argv', ['prog'])
    assert home.get_dataset_root('ModelNet40') == os.path.join(
        os.path.expanduser(home.DEFAULT_CACHE_DIR), 'ModelNet40'
    )


def test_get_dataset_root_uses_argv(monkeypatch, tmp_path):
    monkeypatch.setattr(home.sys, 'argv', ['prog', str(tmp_path)])
    assert home.get_dataset_root('ModelNet40') == os.path.join(
        os.path.realpath(str(tmp_path)), 'ModelNet40'
    )
