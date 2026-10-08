import os
import os.path as osp
import sys

ENV_CLOUDS_HOME = 'CLOUDS_HOME'
DEFAULT_CACHE_DIR = osp.join('~', '.cache', 'clouds')

_home_dir: str | None = None


def get_home_dir() -> str:
    r"""Get the cache directory used for storing all :obj:`clouds`-related data.

    If :meth:`set_home_dir` is not called, the path is given by the environment
    variable :obj:`$CLOUDS_HOME` which defaults to :obj:`"~/.cache/clouds"`.
    """
    if _home_dir is not None:
        return _home_dir

    return osp.expanduser(os.getenv(ENV_CLOUDS_HOME, DEFAULT_CACHE_DIR))


def set_home_dir(path: str) -> None:
    r"""Set the cache directory used for storing all :obj:`clouds`-related data.

    Args:
        path (str): The path to a local folder.
    """
    global _home_dir
    _home_dir = path


def get_dataset_root(name: str) -> str:
    r"""Return the root directory for the dataset ``name``.

    Uses :obj:`sys.argv[1]` as the base data directory when available
    (as in ``python -m clouds.datasets.modelnet /data``), otherwise falls
    back to :meth:`get_home_dir`.
    """
    base = sys.argv[1] if len(sys.argv) > 1 else get_home_dir()
    return osp.join(osp.realpath(base), name)

