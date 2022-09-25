# online-pricing
[![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg)](https://www.python.org/downloads/release/python-3100/)
## Setup

This project uses Python version 3.10, remember to install it  in the way you prefer.
You can use the [official site](https://www.python.org/downloads/) or [pyenv](https://github.com/pyenv/pyenv) or magic ;).

### Poetry Installation

`Poetry` is a super-cool and fresh package manager for Python. To install it:

#### osx / linux / bashonwindows install instructions
```sh 
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```
#### windows powershell install instructions

```sh
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
```

Restart the shell and type
```sh
poetry --version
```
to check that your installation went down successfully.

For more information/debugging/documentation check [this guide](https://python-poetry.org/docs/).

#### Configure virtual environments location
To configure Poetry to create the virtual environments in the `.venv` folder inside the local folder, run:
```sh
poetry config virtualenvs.in-project true
```
Setting this option to true is recommended because it makes it easy to explore the virtual environment, or to remove it if something goes wrong (`rm -rf .venv`).

### Poetry Usage

The basic workflow is the following:
* You want to add a Python module to your .venv? `poetry add <module>`
* You want to remove a Python module from your .venv? `poetry remove <module>`
* You want to use the virtual environment for the project? `poetry shell`
* You want to see the list of all the installed packages? `poetry show`

<ins>**Remember**</ins>:
the first time you clone the repo, use the command
```sh
poetry install
```
to generate your virtual environment.

# Running the Simulator

To start the simulator, from the source folder run this command:
```sh
poetry run simulator
```

It'll ask you which step has to be run.

Additionally, there are console arguments which you can specificy to your needs, more info with:

```sh
poetry run simulator --help
```
