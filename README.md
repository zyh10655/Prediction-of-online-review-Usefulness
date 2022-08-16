# CS760

This repositories is for CS760 Group 6. Welcome guys!

## Scripts

To run anything in `scripts/`, run from the project root directory:

`bash ./scripts/myScript`

### Script Descriptions:

- `exportConda` & `importConda`: export or import the conda environment via `environment.yml`. Used for recreating a consistent conda environment.
- `startJupyter`: Starts a jupyter lab server. Open jupyter in your browser by clicking on the URL printed out on the terminal.
- `initDB`: Create a new database file with empty tables (see `database/scripts/createTables.sql`). **Deletes old database file if exists**.
- `joinData`: Joins and then returns the relevant data for further pre-processing.

## Raw Data

Please place raw data (straight from the internet) into `preprocessing/raw_data/`. This is so that we have a consistent location for our code to operate on.

## A Note on Random Seeds

Please set random seeds to some constant so that we can keep multiple runs consistent.

## Conda Environment

For reproducibility and standard experimentation environment, a conda environment
should be used. The conda environment can be built from `environment.yml` by
running the script `./scripts/importConda`. The libraries and tools included
are those in CS 762 (numpy, scikit-learn, pandas, jupyterlab, matplotlib). More
can be added as needed.

Make sure to activate the environment by running:

`conda activate CS760`

To deactivate:

`conda deactivate`

## Preprocessing

- Importing raw data into SQL database:
Run `initDB` (see scripts) first. Then run `importYelp.ipynb` to import data into SQL.

- Joining data together:
Run `joinData`.
