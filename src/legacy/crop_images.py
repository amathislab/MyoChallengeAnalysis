from PIL import Image
import os 
from definitions import ROOT_DIR


for filename in os.listdir(os.path.join(ROOT_DIR,"Screenshots_PAs")):
    fp = os.path.join(ROOT_DIR,"Screenshots_PAs/"+filename)
    im = Image.open(fp=fp)
    w,h = im.size
    left = w/3
    top = h/5
    bottom = 4.5*h/5
    right = 2*w/3
    im1 = im.crop((left,top,right,bottom))
    im1.save(os.path.join(ROOT_DIR,"Cropped-screenshots/"+filename))
