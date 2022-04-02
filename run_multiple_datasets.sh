#!/bin/bash

languages=("en" "de" "fr" "jp")
datasets=("sentiment" "mnli")
for lang in "${languages[@]}"
do
  datasets+=("sentiment language ${lang}")
done
OPTIND=1
while getopts "g:" val
do
  case "$val" in
  g)
    gpu=${OPTARG}
    ;;
  *)
    echo 'Error'
  esac
done
for dataset in "${datasets[@]}"
do
  source run_single_dataset.sh -d "$dataset" -g "$gpu"
done