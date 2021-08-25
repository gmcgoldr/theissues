#!/bin/bash

# Download the Postgres database dump and create a local version

if [[ $(dirname $0) != "scripts" ]]; then
    echo "Run from the project root"
    exit -1
fi

mkdir -p data/

pushd data/
wget https://openparliament.ca/media/openparliament.public.sql.bz2
[[ $? != 0 ]] && echo "Failed to download DB" && exit -1;
bzip2 -d openparliament.public.sql.bz2
[[ $? != 0 ]] && echo "Failed to extract DB" && exit -1;
popd

sudo -u postgres psql -c "CREATE DATABASE theissues2;"
[[ $? != 0 ]] && echo "Failed to create DB" && exit -1;
sudo -u postgres psql -d theissues2 -f data/openparliament.public.sql
[[ $? != 0 ]] && echo "Failed to populate DB" && exit -1;
