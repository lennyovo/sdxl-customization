import torch
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
import subprocess

import sys

if len(sys.argv) < 2:
        print("Usage: python script.py <your_parameter>")
        sys.exit(1)
else:
        # Print the command line parameter
        customer = sys.argv[1]
        print("Command line parameter:", customer)


print ('Importing Pipeline')

base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

print ('Training Model')

from accelerate.utils import write_basic_config
write_basic_config()

launch_command = 'accelerate launch /workspace/diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0  \
  --instance_data_dir=/project/data/'+customer+' \
  --output_dir=/project/models/tuned-'+customer+' \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of "'+customer+' \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100 \
  --seed="0"'

trial = 'accelerate launch'

subprocess.run(launch_command,shell=True)

print ('Finished')


