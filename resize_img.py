from PIL import Image
from os import listdir
from os.path import isfile, join

# Code used to extract the DIV2K subset for training and validation steps.
IMAGES_DIR = "./DIV2K_valid_LR_bicubic_X4"
IMAGES_DEST_DIR = "./DIV2K_valid_LR_bicubic_X4_resized"
only_files = [f for f in listdir(IMAGES_DIR) if isfile(join(IMAGES_DIR, f))]

for file in only_files:
    print(file)
    img = Image.open(join(IMAGES_DIR, file))
    if img.size[0] % 2 == 0 and img.size[1] % 2 == 0:
        print(file)
        img = img.resize((int(img.size[0]/2), int(img.size[1]/2)), Image.ANTIALIAS)
        img.save(join(IMAGES_DEST_DIR, file))

