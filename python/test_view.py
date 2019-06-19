

import numpy as np 
import openslide as ops
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2
import skimage.io as io
import matplotlib.colors as mcolors
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.measure import label, regionprops
import h5py
from xml.etree import ElementTree as ET

# make colormap to use for plotting pred on 2D-slice
colors = [(0,1,0,i) for i in np.linspace(0,1,3)]
cmap = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=10)


# paths
loc = "/hdd/NUCLEI/MoNuSeg Training Data/Tissue images/TCGA-38-6178-01Z-00-DX1.tif"
data_path = '/hdd/NUCLEI/MoNuSeg Training Data/Tissue images/'
gt_path = '/hdd/NUCLEI/MoNuSeg Training Data/Annotations/'

# for each WSI
for path in tqdm(os.listdir(data_path), "WSI"):

	print(path)

	# get image patch
	data = plt.imread(data_path + path)

	# get GT
	gt_loc = path.split('.tif')[0] + '.xml'
	tree = ET.parse(gt_path + gt_loc)
	root = tree.getroot()
	locations = []
	gt = np.zeros(data.shape[:-1])
	gts = []
	regions = root[0][1]

	# for each nuclei
	for region in regions[1:]:
		vertices = region[1]
		locs = []
		gt_curr = np.zeros_like(gt)

		# extract vertices for each nuclei
		for vert in vertices:
			vert = vert.attrib
			x = int(round(float(vert['X'])))
			y = int(round(float(vert['Y'])))
			locs.append((x,y))
			locations.append((x,y))

		# fill set of vertices to form segmented nuclei
		coords = np.array(locs)
		cv2.fillPoly(gt_curr, pts=[coords], color=(255,255,255))
		gts.append(gt_curr.astype(np.uint8))

		# add segmented nuclei to GT image
		gt += gt_curr
	gts = np.array(gts).astype(np.uint8)
	print(gts.shape)





	#'''
	#gt = np.zeros(data.shape[:-1])
	#for loc in locations:
	#	gt[loc[::-1]] = 1

	#gt = gt.astype(np.uint8)
	#gt = np.zeros(data.shape[:-1])
	#coords = []
	#for loc in locations:
	#	coords.append(loc)
	#coords = np.array(coords)
	#cv2.fillPoly(gt, pts = [coords], color=(255,255,255))
	#'''

	#gt[gt > 0] = 1

	from dynamic_watershed import post_process
	p1, p2 = 200, 0.5
	nuclei = post_process(gt, p1, thresh=p2)

	#'''
	#nuclei = label(gt)

	vals = np.linspace(0,1,256)
	np.random.shuffle(vals)
	vals = plt.cm.jet(vals)
	vals[0, :] = [0,0,0,1]
	cmap2 = plt.cm.colors.ListedColormap(vals)


	names = ['WSI', 'Overlay', 'GT', 'GT fixed']
	fig, ax = plt.subplots(1, 4, figsize=(26,13))
	fig.tight_layout()
	ax[0].imshow(data)
	ax[1].imshow(data)
	ax[1].imshow(gt, cmap, alpha=0.8)
	ax[2].imshow(gt, cmap="gray")
	ax[3].imshow(nuclei, cmap=cmap2)

	for i,n in enumerate(names):
		ax[i].set_title(n)
		ax[i].set_xticks([])
		ax[i].set_yticks([])

	plt.show()
	#'''



	print('---')

	#exit()
	#plt.imshow(data)
	#plt.show()

