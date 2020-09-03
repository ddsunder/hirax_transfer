# Update log

- 3 September 2020, added in Sarea and FoV calculations
- 31 August 2020, added rotated square grid layout and a general user friend sample use file. 

# Requirements

Other than standard scientific python libraries, this code requires:

- `driftscan` (Currently only tested to work with the hirax_dev branch at https://github.com/hirax-array/driftscan)
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

For the second step you'll need to point to code to a simulated map generated with cora.

To make this example run quicker, reduce the number of pointings or the size of the antenna grid.
