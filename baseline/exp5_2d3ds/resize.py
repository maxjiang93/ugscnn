from PIL import Image
from resizeimage import resizeimage
from glob import glob
import os
from scipy.misc import imread, imsave

res = [512, 256]
areas = ["1", "2", "3", "4", "5a", "5b", "6"]

rgb_form = "data/area_{}/rgb/*.png"
d_form = "data/area_{}/depth/*.png"
l_form = "data/area_{}/semantic/*.png"

rgb_files = []
d_files = []
l_files = []
for a in areas:
	rgb = glob(rgb_form.format(a))
	d = glob(d_form.format(a))
	l = glob(l_form.format(a))
	rgb_files += rgb
	d_files += d
	l_files += l
rgb_files = sorted(rgb_files)
d_files = sorted(d_files)
l_files = sorted(l_files)

from tqdm import tqdm
for r, d, l in tqdm(zip(rgb_files, d_files, l_files), total=len(l_files)):
	r_sm = r.replace("data", "data_small")
	d_sm = d.replace("data", "data_small")
	l_sm = l.replace("data", "data_small") 
	dname = os.path.dirname(r_sm)
	if not os.path.exists(dname):
		os.makedirs(dname)
	dname = os.path.dirname(d_sm)
	if not os.path.exists(dname):
		os.makedirs(dname)
	dname = os.path.dirname(l_sm)
	if not os.path.exists(dname):
		os.makedirs(dname)
	for infile, outfile in zip([r, d], [r_sm, d_sm]):
		with open(infile, 'r+b') as f:
		    with Image.open(f) as image:
		        cover = resizeimage.resize_cover(image, res)
		        cover.save(outfile, image.format)
		# treat labels separately
		l_mat = imread(l)[4::8, 4::8]
		imsave(l_sm, l_mat)
