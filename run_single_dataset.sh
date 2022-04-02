#!/bin/bash

types=("hyper-dn" "hyper-drf" "hyper-pada")
OPTIND=1
while getopts "g:d:" val
do
  case "$val" in
  g)
    gpu=${OPTARG}
    ;;
  d)
    dataset=${OPTARG}
    ;;
  *)
    echo 'Error'
  esac
done
for type in "${types[@]}"
do
  source run_single_model_single_dataset.sh -t "$type" -d "$dataset" -g "$gpu"
done
