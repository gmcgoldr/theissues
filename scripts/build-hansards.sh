#!/bin/bash

# Extracts the hansards statements from the DB into a file `data/hansards.jsonl`

if [[ $(dirname $0) != "scripts" ]]; then
    echo "Run from the project root"
    exit -1
fi

mkdir -p data/
sudo -u postgres psql -d theissues -f scripts/extract-hansards.sql -t > data/hansards.jsonl
