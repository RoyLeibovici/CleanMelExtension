cd ../

gpus=$1
mode=$2
size=$3
output=$4
huggingface_used=$5

speech_folder="./dev_utils/test_sample/"
output_folder="./my_output/"

# Sanity check + Configurations
# output
if [ $output != "mask" ] && [ $output != "map" ]; then
    echo "Invalid output, must be 'mask' or 'map'"
    exit 1
fi
# mode
if [ $mode != "offline" ] && [ $mode != "online" ]; then
    echo "Invalid mode, must be 'offline' or 'online'"
    exit 1
fi
# size
if [ $size == "S" ] && [ $mode == "offline" ]; then
    n_layers=8
    dim_hidden=96
elif [ $size == "L" ] && [ $mode == "offline" ]; then
    n_layers=16
    dim_hidden=144
elif [ $size == "S" ] && [ $mode == "online" ]; then
    n_layers=16
    dim_hidden=96
else
    echo "Invalid size, must be 'S' or 'L'"
    exit 1
fi

# huggingface_used
if [[ $huggingface_used == "huggingface" ]]; then
    vocos_ckpt=HF!ckpts/Vocos/vocos_${mode}.pt
    ckpt_path=HF!ckpts/CleanMel/${mode}_CleanMel_${size}_${output}.ckpt
else
    vocos_ckpt=./pretrained/vocos/vocos_${mode}.pt
    ckpt_path=./pretrained/enhancement/${mode}_CleanMel_${size}_${output}.ckpt
fi
python -m model.CleanMelTrainer_${output} predict \
    --config ./configs/model/cleanmel_${mode}.yaml \
    --config ./configs/dataset/inference.yaml \
    --trainer.devices=${gpus} \
    --model.vocos_ckpt ${vocos_ckpt} \
    --model.vocos_config ./configs/model/vocos_${mode}.yaml \
    --model.arch.init_args.n_layers=${n_layers} \
    --model.arch.init_args.dim_hidden=${dim_hidden} \
    --data.speech_dir ${speech_folder} \
    --model.output_path ${output_folder} \
    --model.arch_ckpt=${ckpt_path}