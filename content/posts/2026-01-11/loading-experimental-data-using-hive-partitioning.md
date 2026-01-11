---
date: "2026-01-10"
title: "Structuring and loading your experiment's results using hive partitioning and duckdb"
description: If you structure your experiment's results in a certain way, you get the loading almost for free using duckdb.
summary: If you structure your experiment's results in a certain way, you get the loading almost for free using duckdb.
images: null
image: null
---

> This blogpost has a code companion. [Check it out here!](https://github.com/miguelgondu/code-in-blogposts/tree/main/hive-partitioning-example)

I learned a trick from [a friend called Oscar](https://github.com/odarbelaeze) recently: if you organize your experiments in a certain way, **you can easily load your results as pandas dataframes**. This is possible through a combination of hive partitioning and [duckdb](https://duckdb.org/docs/stable/data/partitioning/hive_partitioning).

# Probably your current setup

Imagine you're running an experiment, and your experiment's design matrix has several variables. For example, you could be running a computational analysis on how a molecule binds to a protein, and you might want to run it for several proteins, molecules, and seeds.

You'd usually have a function that runs the expensive experiment and returns some results, something like

```python
def some_simulation(
    protein: Protein,
    molecule: Molecule,
    seed: int
) -> pd.DataFrame:
    """
    Runs some expensive simulation and returns a
    dataframe of results.
    """
    ...

    return pd.Dataframe(
        [
            {
                "binding_affinity": some_number,
                "accuracy": some_other, 
            }
        ]
    )
```

And you might have a folder called `results` in which you're storing them like

```
# A pretty ugly and unstructured results dir
results/
    protein_9L76_molecule_mol1_seed_0.csv
    protein_9L76_molecule_mol1_seed_1.csv
    protein_9L76_molecule_mol1_seed_2.csv
    ...
    protein_9L76_molecule_molN_seed_0.csv
    protein_9L76_molecule_molN_seed_1.csv
    protein_9L76_molecule_molN_seed_2.csv
    ...
    protein_9ODL_molecule_molN_seed_0.csv
    protein_9ODL_molecule_molN_seed_1.csv
    protein_9ODL_molecule_molN_seed_2.csv
```

Then, when you're loading them up using pandas, you might be either concatenating all of them or, even worse, parsing the filename and extracting the protein, molecule and seed using ugly `name.split("_")` shenanigans.

# A small improvement: hive partitioning

I recently learned to structure my experimental results like this:
```
# A slightly better way to arrange results
results/
    protein=9L76/
        molecule=mol1/
            seed=0/
                results.csv
            ...
            seed=M/
                results.csv
```

That is, **name your folders with the variables, and nest them**. With this, you can easily load them into `pd.DataFrame`s using `duckdb`:

```python
import duckdb

def load_results() -> pd.DataFrame:
    local_connection = duckdb.connect()

    # An SQL query over your filesystem!
    query = """
    SELECT protein, molecule, seed, binding_affinity, accuracy
    FROM read_csv("results/**/*.csv", hive_partitioning=true)
    """

    df = local_connection.execute(query).df()
    ...
```

Now `df` will have `protein`, `molecule` `seed`, `binding_affinity`, and `accuracy` as columns. `print(df)` says

```
    protein molecule  seed  binding_affinity  accuracy
0      9L76     mol1     0          0.643167  0.269787
1      9L76     mol1     1          0.645366  0.950464
2      9L76     mol1     2          0.643800  0.298491
3      9L76     mol1     3          0.662319  0.236811
4      9L76     mol1     4          0.635392  0.511328
..      ...      ...   ...               ...       ...
195    9ODL     mol4     5          0.728823  0.807941
196    9ODL     mol4     6          0.747373  0.343271
197    9ODL     mol4     7          0.736854  0.897214
198    9ODL     mol4     8          0.719459  0.987277
199    9ODL     mol4     9          0.728814  0.286817

[200 rows x 5 columns]
```

You could even flex your SQL skills by running the computations you need (e.g. filtering, computing averages over seeds...) inside the SQL query itself. Indeed, one could run the entire data analysis in SQL, blazingly fast.

## Another improvement: dataclasses or Pydantic models

For a less trivial example, imagine you are also storing some metadata per experiment (e.g. which model you used to compute the binding affinity, or a dataset you used during training). You could also load all this data into `DataFrame`s using `duckdb`, because it supports `json` loading!

So, instead of returning a bare `DataFrame`, you could be returning a structured (and input-validated) Pydantic model:

```python
from pydantic import BaseModel

class ExperimentMetadata(BaseModel):
    run_at: datetime  # This could be a field with datetime.now() in its factory.
    model_used: Literal["GAABind", "Dockstring", "PBCNet", "AF3"]
    dataset_used: Literal["PDBbind", "CASF", "Binding MOAD", "BindingDB"]

# This replaces that ugly Pandas dataframe.
class ExperimentResults(BaseModel):
    binding_affinity: float
    accuracy: float
    metadata: ExperimentMetadata
```

Now your experimental run could return something like...
```python
def some_simulation(
    protein: Protein,
    molecule: Molecule,
    seed: int
) -> ExperimentResults:
    """
    Runs some expensive simulation and returns a
    pydantic model of results.
    """
    ...

    return ExperimentResults(
        binding_affinity=some_number,
        accuracy=another_number,
        metadata=ExperimentMetadata(
            run_at=datetime.now()
            model_used="GAABind",
            dataset_used="PDBbind"
        )
    )
```

...which now you are storing under e.g. `results/protein=9L76/molecule=mol1/seed=0/results.json`. With `duckdb`, you can still load these up trivially by switching `read_csv` with `read_json` in the code above:

```python
import duckdb

def load_results() -> pd.DataFrame:
    local_connection = duckdb.connect()

    # An SQL query over your filesystem!
    query = """
    SELECT protein, molecule, seed, binding_affinity, accuracy, metadata
    FROM read_json("results/**/*.json", hive_partitioning=true)
    """

    df = local_connection.execute(query).df()
    ...
```

And now your `df` will look like this:
```
    protein molecule  seed  binding_affinity  accuracy                                           metadata
0      9L76     mol1     0          0.643167  0.269787  {'run_at': '2026-01-11T08:51:11.536596', 'mode...
1      9L76     mol1     1          0.645366  0.950464  {'run_at': '2026-01-11T08:51:11.536876', 'mode...
2      9L76     mol1     2          0.643800  0.298491  {'run_at': '2026-01-11T08:51:11.536991', 'mode...
3      9L76     mol1     3          0.662319  0.236811  {'run_at': '2026-01-11T08:51:11.537080', 'mode...
4      9L76     mol1     4          0.635392  0.511328  {'run_at': '2026-01-11T08:51:11.537146', 'mode...
..      ...      ...   ...               ...       ...                                                ...
195    9ODL     mol4     5          0.728823  0.807941  {'run_at': '2026-01-11T08:51:11.551988', 'mode...
196    9ODL     mol4     6          0.747373  0.343271  {'run_at': '2026-01-11T08:51:11.552047', 'mode...
197    9ODL     mol4     7          0.736854  0.897214  {'run_at': '2026-01-11T08:51:11.552106', 'mode...
198    9ODL     mol4     8          0.719459  0.987277  {'run_at': '2026-01-11T08:51:11.552170', 'mode...
199    9ODL     mol4     9          0.728814  0.286817  {'run_at': '2026-01-11T08:51:11.552249', 'mode...

[200 rows x 6 columns]
```

And `duckdb` even allows for accessing inside the `metadata` json from the SQL query. If we were only interested in e.g. the `model_used` field, we could change the `query` to:
```python
# Replacing metadata with metadata.model_used to access only one key of the metadata object
query = """
SELECT protein, molecule, seed, binding_affinity, accuracy, metadata.model_used
FROM read_json("results/**/*.json", hive_partitioning=true)
"""
```