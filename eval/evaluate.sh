#!/bin/zsh
config_path="$1"
inference_cmd="python inference.py --config $config_path"
evaluate_cmd="python eval/evaluate_front3d.py --config $config_path"
evaluate_normal_cmd="python eval/evaluate_normal_consistency.py --config $config_path"
eval $inference_cmd
eval $evaluate_cmd
eval $evaluate_normal_cmd
