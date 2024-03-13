print ('Import AI Libraries')
import torch
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline

print ('Import File Libraries')
import cv2
import os
from PIL import Image

directory = '/project/code'
os.chdir(directory)
# print (os.listdir(directory))

print ('Download the Stable Diffusion Pipelines')

base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.load_lora_weights("/project/models/tuned-toy-jensen")

print ('Making an image')

prompt = "toy jensen in space"
img = pipe(prompt=prompt).images[0]

filename='test.jpg'
img.save("test1.png")


# Image.fromarray(img.save("test.png"))

print ('Finished...')

