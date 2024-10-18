export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./runs/controlnet_fill50k_v2"   # output directory of each run

export CUDA_VISIBLE_DEVICES="4,5,6,7"

accelerate launch --main_process_port=29503 train.py \
--seed=0 \
--pretrained_model_name_or_path=$MODEL_DIR \
--output_dir=$OUTPUT_DIR \
--dataset_name=fusing/fill50k \
--resolution=512 \
--learning_rate=1e-5 \
--validation_image "./data/conditioning_image_1.png" "./data/conditioning_image_2.png" \
--validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
--train_batch_size=1 \
--gradient_accumulation_steps=4 \
--gradient_checkpointing \
--set_grads_to_none \
--use_8bit_adam \
--checkpoints_total_limit 2 \
--validation_steps 100 \
--report_to "tensorboard" \
--num_train_epochs 4