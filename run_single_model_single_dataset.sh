#!/bin/bash

OPTIND=1
while getopts "t:d:g:" val
do
  case "$val" in
  t)
    model_type=${OPTARG}
    ;;
  d)
    task=${OPTARG}
    ;;
  g)
    gpu=${OPTARG}
    ;;
  *)
    echo 'Error'
  esac
done
data_dir="$(grep 'data_dir' "src/config files/config - ""$task"".yaml")"
data_dir="$(cut -d':' -f2 <<<"$data_dir" |  xargs)"
if [[ $data_dir =~ .*/ ]]; then
  data_dir=${data_dir::-2}
fi
for dir in "$data_dir"/*/; do
    dir=${dir#"$data_dir/"}
    domain="${dir///}"
    if [[ "$model_type" =~ (hyper-pada|hyper-drf) ]]; then
      echo generator
      CUDA_VISIBLE_DEVICES="$gpu" python src/main.py --target "$domain" --model_type "generator" --task "$task"
    fi
    CUDA_VISIBLE_DEVICES="$gpu" python src/main.py --target "$domain" --model_type "$model_type" --task "$task"
done
