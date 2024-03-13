from PIL import Image
from PIL import ImageDraw

import sys

if len(sys.argv) < 3:
        print("Usage: python create_picture.py <customer> <Image Number>")
        sys.exit(1)
else:
        # Print the command line parameter
        customer = sys.argv[1]
        print("Command line parameter:", customer)

top = Image.open("/project/data/GTC-Frame.png")
bottom = Image.open("/project/aipics/"+customer+"/"+customer+str(sys.argv[2])+".png")

r,g,b,a = top.split()
top = Image.merge("RGB",(r,g,b))
mask = Image.merge("L", (a,))
bottom.paste(top, (0,0),mask)


bottom.save("/project/pictures/"+customer+".png")

print("Finished")
