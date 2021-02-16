import subprocess
from pathlib import Path


def install(path_pyopencl):
    path_pip = Path(os.sys.executable).parent.joinpath('pip')
    if path_pip.exists():
        command = '{} install "{}"'.format(str(path_pip), str(path_pyopencl))
    else:
        path_python = Path(os.sys.executable).parent.joinpath('Scripts/pip')
        command = '{} install "{}"'.format(str(path_python), str(path_pyopencl))
    subprocess.call(command)


if __name__ == '__main__':
    import os

    cwd = os.getcwd()

    install(Path(cwd).joinpath('pyopencl-2021.1.1+cl21-cp38-cp38-win_amd64.whl'))
