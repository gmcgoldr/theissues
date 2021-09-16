#!/bin/bash

# Extracts the hansards statements and politician names from the DB

if [[ $(dirname $0) != "scripts" ]]; then
    echo "Run from the project root"
    exit -1
fi

mkdir -p data/
sudo -u postgres psql -d theissues -f scripts/extract-hansards.sql -t > data/hansards.jsonl
sudo -u postgres psql -d theissues -f scripts/extract-politicians.sql -t > data/politicians.jsonl

