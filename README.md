# The Issues

This project is using data from `https://openparliament.ca/`,
which is derived from `https://www.parl.gc.ca/HousePublications/Publication.aspx`.

## Requirements

This README written assuming an Ubuntu operating system.

You need Python 3.6 or later.
To build the database, you need postgres installed as well.

## Installation

To install the python package, from the project root run:

```bash
pip install .
```

To build the data from scratch:

```bash
scripts/download-db.sh
scripts/build-hansards.sh
```
