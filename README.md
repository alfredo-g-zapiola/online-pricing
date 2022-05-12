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

## Structure 

#### class Simulator:
- ****Members****:
  - affinity matrix for each client group, made uncertain (some distribution) in step 5
  - dictionary: first & secondary product for each product (take 2 edges with max affinity). FIXED by assumption
  - 4 possible prices for each product
  - current price of every product: arm the learner pulls
  - lambda
  - demand curve for each product, for each of 3 groups (Simplest case: a straight line + some noise; else functional data with sklearn library). **For step 7** make them change along time. **NB** uncertain conversion rates means exaclty the demand curve has noise
  - units sold discrete distribution parameters (discrete weibull makes sense!) for each group 
  - parameters of distribution of daily potential customers
  - alpha ratios, later when uncertain (**step 4**) dirichlet distribution parameters for alpha ratios for each group of people
  - total number of each group


- __Methods__:
  - **sim_one_day()**:
    - obtain number of potential customers of each product by sampling dirichlet distribution
    - for each user:
      - **sim_one_user(day, product_landed)** and store the results
    - Update the prices of the products according to the learner output, as said on **step 2**


  - **sim_one_user(day, product_landed)**:
    1. Obtain the user's reservation price sampling from the demand curve
    2. **If price is lower than user's r price**,
       1. store quantity (Sample from a discrete distribution the number of units) of units sold of that product in that day
       2. with probability taken from matrix, go to secondary product, repeat C, without further secondary products
       3. with proba lambda * p taken from matrix go to second secondary product, decide if buy
       4. From the previous steps, store all data (if clicked or not, quanityt bought)

**Important note**:     from step 3 to 6, the changes are only on the members, but the functionality of the class is the same.
For example, the demand curves on step 7 have to change over time. But we just need to add a helper function to update those members and we're done!

### The list of random variables
- Number of daily clients each day
- Number of items user will buy
- alpha ratios (dirichlet)
- Conversion probability with every price (demand curve)
- graph probabilities (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wishart.html)
