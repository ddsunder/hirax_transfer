# Requirements

Other than standard scientific python libraries, this code requires:

- `driftscan`
- `cora`
- `caput`

and all of their dependencies to be installed.

# Installation

`python setup.py install`

or

`python setup.py develop`

# Example Configuration Files

There are example configuration files in `examples/configs`.

Runs using these files can be performed with:

`drift-makeproducts run prod_params.yaml` 

and then

`drift-runpipeline run pipe_params.yaml`

To make these examples run quicker, reduce the number of pointings or the size of the antenna grid.
