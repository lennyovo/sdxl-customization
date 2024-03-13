import sys

if len(sys.argv) < 2:
        print("Usage: python makepic.py <customer> <situation>")
        sys.exit(1)
else:
        # Print the command line parameter
        customer = sys.argv[1]
        print("Command line parameter: ", customer)

if len(sys.argv) == 3:
	situation = sys.argv[2]
	print("Situation: ", situation)

print ('Import AI Libraries')
import torch
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline

print ('Import File Libraries')
import cv2
import os
from PIL import Image

directory = '/project/aipics/'
if not os.path.exists(directory):
	os.mkdir(directory)
directory = directory + customer
if not os.path.exists(directory):
	os.mkdir(directory)
os.chdir(directory)

print ('Download the Stable Diffusion Pipelines')

base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
lora ='/project/models/tuned-'+customer
print ('Making ', customer, situation)

#pipe.load_lora_weights(lora,adapter_name=customer)
pipe.load_lora_weights(lora)

#tj = '/project/models/tuned-toy-jensen'
#pipe.load_lora_weights(tj,adapter_name="jensen")

print ('Loaded Loras')

#pipe.set_adapters([customer,"jensen"], adapter_weights=[1.0,1.0])
 
print ('Making images')
for i in range(1,10):
	prompt = customer+situation
#	img = pipe(prompt=prompt,num_inference_steps=70, cross_attention_kwargs={"scale": 1.0}).images[0]
	img = pipe(prompt=prompt,num_inference_steps=100).images[0]
	filename=customer+str(i)+'.png'
	img.save(filename)

print ('Finished...')

