import torch
import numpy as np
from scipy.ndimage.measurements import center_of_mass
from skimage.measure import regionprops
import sys


def compare_mask_intersection(mask1, mask2,prop_threshold=0.5):
	'''
	Compare two binary masks and return id of mask with a higher proportion of it's
	area overlapping with the other mask
	'''
	mask1_area = np.count_nonzero(mask1 == 1)
	mask2_area = np.count_nonzero(mask2 == 1)
	intersection = np.count_nonzero(np.logical_and( mask1==1,  mask2==1 ))
	try:
		max_proportion = np.max([intersection/mask1_area,intersection/mask2_area])
		if max_proportion > prop_threshold:
			more_overlapped = np.argmax([intersection/mask1_area,intersection/mask2_area])
		else:
			more_overlapped = -1
		# iou = intersection/(mask1_area+mask2_area-intersection+np.finfo(float).eps)
	except Exception:
		more_overlapped = -1
	return more_overlapped

class MaskRCNN():
	'''
	Mask R-CNN-based nuclei segmentation 
	Attributes:
		weights: string
			path to pytorch model weights .pt file
		model: pytorch object
			model used to evaluate images
		use_cpu: bool
			flag to load model and evaluate images on a CPU (instead of a GPU)
		
		device: torch.device object
			device to load model and evaluate images on (depends on use_cpu)
	'''	

	weights = None
	
	use_cpu = True
	device = None

	def __init__(self, weights,use_cpu):
		self.weights = weights
		self.use_cpu = use_cpu
		self.model = self.load_model()
		self.device = torch.device('cuda')

	def load_model(self):
		'''
		Load pytorch model with weights stored at self.weights into memory on either CPU or GPU
		'''
		if self.use_cpu:
			self.device = torch.device('cpu')
			print('running on cpu')
		elif torch.cuda.is_available():
			self.device = torch.device('cuda')
			print('running on gpu')
		else:
			self.device = torch.device('cpu')
		model = torch.load(self.weights,map_location=self.device)
		return model

	def generate_masks(self,img,score_thresh=0.8,in_quadrants=False,overlap=75,flatten=False,area_thresh=1000):
		'''
		Generate a stack of segmentation masks (one component per mask)
		Parameters:
		img: np.ndarray
			image to be evaluated
		score_thresh: float
			threshold for accepting components output by Mask R-CNN, must be in range [0,1]
		in_quadrants: bool
			GPU RECOMMENDED: flag for segmenting original image in quadrants instead of all at once, 
			useful for very dense images (>200 nuclei)
		
		overlap: int
			number of pixels to extend quadrants by in order to accurately segment nuclei on
			the border of the quadrants
		flatten: bool
			compresses stack of masks to 2D max projection (conserves memory)
		area_thresh: int
			pixel area threshold for minimum nucleus size, removes all nuclei with size < area_thresh
		Return:
			masks: np.ndarray
				3D array of binary masks (n_masks x image_size[0] x image_size[1])
		'''
		model = self.model
		model.eval().to(self.device)
		#print("Generating nuclei masks...")
		img = np.array(img)
		
		# Segment quadrants of original image, then merge together
		if in_quadrants:
			center_coord = (int(len(img[0])//2),int(len(img[0,0])//2))

			imgs = [img[:,0:(center_coord[0]+overlap),0:(center_coord[1]+overlap)],
					img[:,0:(center_coord[0]+overlap),(center_coord[1]-overlap):],
					img[:,(center_coord[0]-overlap):,0:(center_coord[1]+overlap):],
					img[:,(center_coord[0]-overlap):,(center_coord[1]-overlap):]]
			padding = [((0,0),(0,len(img[0])-center_coord[0]-overlap),(0,len(img[0])-center_coord[1]-overlap)),
					   ((0,0),(0,len(img[0])-center_coord[0]-overlap),(center_coord[1]-overlap,0)),
					   ((0,0),(center_coord[0]-overlap,0),(0,len(img[0])-center_coord[1]-overlap)),
					   ((0,0),(center_coord[0]-overlap,0),(center_coord[1]-overlap,0))]
			
			for i in range(len(imgs)):
				prediction = model([torch.Tensor(np.array(imgs[i])).to(self.device)])
				
				scores = prediction[0]['scores']
				if not self.use_cpu:
					scores = scores.cpu()
				scores = scores.detach().numpy()
				above_threshold = np.where(scores>score_thresh)
				
				masks = prediction[0]['masks'][above_threshold]
				if not self.use_cpu:
					masks = masks.cpu()
				masks = masks.detach().numpy()

				centroids = np.zeros((len(masks),2))
				areas = np.zeros(len(masks))

				# Delete cells on edges of quadrant within the overlapping areas
				for j in range(len(masks)):
					bin_mask = np.where(masks[j,0]>0.5,1,0)
					props = regionprops(bin_mask)[0]
					centroids[j] = props.centroid[0:2]
					areas[j] = props.area
				if i == 0:
					to_delete= np.nonzero(np.logical_or(centroids[:,0]>=(len(bin_mask)-overlap/2),centroids[:,1]>=(len(bin_mask[0])-overlap/2)))[0]
				elif i == 1:
					to_delete= np.nonzero(np.logical_or(centroids[:,0]>=(len(bin_mask)-overlap/2),centroids[:,1]<=(overlap/2)))[0]
				elif i == 2:
					to_delete= np.nonzero(np.logical_or(centroids[:,0]<=(overlap/2),centroids[:,1]>=(len(bin_mask[0])-overlap/2)))[0]
				elif i == 3:
					to_delete= np.nonzero(np.logical_or(centroids[:,0]<=(overlap/2),centroids[:,1]<=(overlap/2)))[0]

				# Delete cells below area threshold
				to_delete = np.union1d(to_delete,np.nonzero(areas<area_thresh))
				masks = np.delete(masks,to_delete,axis=0)
				quadrant_masks = np.zeros((len(masks),img.shape[1],img.shape[2]))

				# Pad quadrant image to size of original image
				quadrant_centroids=np.zeros((len(masks),2))
				for j in range(len(masks)):
					quadrant_masks[j] = np.pad(masks[j],padding[i])
					bin_mask = np.where(quadrant_masks[j]>0,1,0)
					props = regionprops(bin_mask)
					quadrant_centroids[j] = list(props[0].centroid)
				x_coords = quadrant_centroids[:,1]
				y_coords = quadrant_centroids[:,0]
				in_overlap_ids = np.nonzero(np.logical_or(np.logical_and(x_coords>=(center_coord[1]-overlap),x_coords<=(center_coord[1]+overlap)),
												   np.logical_and(y_coords>=(center_coord[0]-overlap),y_coords<=(center_coord[0]+overlap))))[0]
				in_overlap_ids=set(in_overlap_ids)
				in_overlap = np.array([(i in in_overlap_ids) for i in range(len(quadrant_masks))])
				if i == 0:
					#print(in_overlap)
					all_masks = quadrant_masks[~in_overlap]
					all_centroids = quadrant_centroids[~in_overlap]
					masks_in_overlap = quadrant_masks[in_overlap]
					overlap_centroids = quadrant_centroids[in_overlap]
				else:
					all_masks = np.concatenate((all_masks,quadrant_masks[~in_overlap]),axis=0)
					all_centroids = np.concatenate((all_centroids,quadrant_centroids[~in_overlap]),axis=0)
					# print(all_centroids.shape)
					masks_in_overlap = np.concatenate((masks_in_overlap,quadrant_masks[in_overlap]),axis=0)
					overlap_centroids = np.concatenate((overlap_centroids,quadrant_centroids[in_overlap]),axis=0)
				if flatten:
					# print(all_centroids.shape)
					all_masks = np.expand_dims(np.max(all_masks,axis=0),axis=0)

			torch.cuda.empty_cache()

			sorted_ids = np.lexsort((overlap_centroids[:,0],overlap_centroids[:,1]))
			masks_in_overlap = masks_in_overlap[sorted_ids]
			overlap_centroids = overlap_centroids[sorted_ids]
			# all_masks = np.delete(all_masks,in_overlap,axis=0)
			masks_to_delete = []
			for i in range(len(masks_in_overlap)):
				x_dist = np.absolute(overlap_centroids[:,1]-overlap_centroids[i,1])
				y_dist = np.absolute(overlap_centroids[:,0]-overlap_centroids[i,0])
				dist = x_dist + y_dist

				close_ids = np.argsort(dist)[1:4]

				
				# Compare current mask to closest masks
				for j in close_ids:
					if i != j:
						mask_a = np.where(masks_in_overlap[i]>0.5,1,0)
						mask_b = np.where(masks_in_overlap[j]>0.5,1,0)
						max_id = compare_mask_intersection(mask_a,mask_b)
						if max_id==0:
							masks_to_delete.append(i)
						elif max_id==1:
							masks_to_delete.append(j)
						else:
							pass
			
			# delete redundant masks in overlapping areas
			overlap_centroids = np.delete(overlap_centroids,list(set(masks_to_delete)),axis=0)
			masks_in_overlap = np.delete(masks_in_overlap,list(set(masks_to_delete)),axis=0)
			
			all_centroids = np.concatenate((all_centroids,overlap_centroids),axis=0)
			all_masks = np.concatenate((all_masks,masks_in_overlap),axis=0)
			if flatten:
				all_masks = np.expand_dims(np.max(all_masks,axis=0),axis=0)
			#print("Final cell count: %d"%len(all_centroids))
			return all_masks, all_centroids
			
		else:
			prediction = model([torch.Tensor(img).to(self.device)])
			scores = prediction[0]['scores']
			

			if not self.use_cpu:
				scores = scores.cpu()
			scores = scores.detach().numpy()
			above_threshold = np.where(scores>score_thresh)
			
			masks = prediction[0]['masks'][above_threshold]
			if not self.use_cpu:
				masks = masks.cpu()
			masks = masks.detach().numpy()
			
			if len(masks.shape) == 2:
				masks = np.expand_dims(masks,axis=0)
			if len(masks)==200:
				print("More than 200 cells found, evaluating in quadrants...")
				del prediction
				del scores
				del above_threshold
				del masks
				torch.cuda.empty_cache()
				return self.generate_masks(img,score_thresh=score_thresh,in_quadrants=True,overlap=overlap,flatten=flatten,area_thresh=area_thresh)
			else:
				#print("Final cell count: %d"%len(masks))
				return masks, None


	def extract_centroids(self,img,score_thresh=0.8, in_quadrants=True,overlap=75,flatten=False,area_thresh=1000):
		'''
		Extract centroids of components recognized by self.model
		Parameters:
			img: np.ndarray
				image to be evaluated
			score_thresh: float
				threshold for accepting components output by Mask R-CNN, must be in range [0,1]
			in_quadrants: bool
				GPU RECOMMENDED: flag for segmenting original image in quadrants instead of all at once, 
				useful for very dense images (>200 nuclei)
		
			overlap: int
				number of pixels to extend quadrants by in order to accurately segment nuclei on
				the border of the quadrants
			flatten: bool
				compresses stack of masks to 2D max projection (conserves memory)
			area_thresh: int
				pixel area threshold for minimum nucleus size, removes all nuclei with size < area_thresh
		Returns:
			centroids: np.ndarray
				N x 2 of centroids (N is number of components)
			
			masks: np.ndarray
				3D array of binary masks (n_masks x image_size[0] x image_size[1])
		'''
		
		masks,centroids = self.generate_masks(img,score_thresh,in_quadrants,overlap,flatten,area_thresh)
		if centroids is None:
			# centroids= np.zeros((len(masks),2))
			centroids= []
			for i in range(len(masks)):
				CoM = np.array(list(center_of_mass(masks[i]))).astype(int)
				# centroids[i] = CoM
				centroids.append([CoM[1],CoM[2]])
		if flatten==True:
			masks= np.expand_dims(np.max(masks,axis=0),axis=0)
		return centroids, masks

