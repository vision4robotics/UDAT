import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import DCE.dce_model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
from tqdm import tqdm
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def lowlight(DCE_net, image_path):
	
	scale_factor = 12
	data_lowlight = Image.open(image_path)

 

	data_lowlight = (np.asarray(data_lowlight)/255.0)


	data_lowlight = torch.from_numpy(data_lowlight).float()

	h=(data_lowlight.shape[0]//scale_factor)*scale_factor
	w=(data_lowlight.shape[1]//scale_factor)*scale_factor
	data_lowlight = data_lowlight[0:h,0:w,:]
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)


	enhanced_image,params_maps = DCE_net(data_lowlight)

	image_path = image_path.replace('train_clip','train_clip_enhanced')

	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))
	# import pdb;pdb.set_trace()
	torchvision.utils.save_image(enhanced_image, result_path)


if __name__ == '__main__':
	with torch.no_grad():

		filePath = '/Your/PATH/NAT2021-train/train_clip/' # the path of original imgs
		file_list = os.listdir(filePath)
		file_list.sort()
		scale_factor = 12
		DCE_net = DCE.dce_model.enhance_net_nopool(scale_factor).cuda()
		DCE_net.eval()
		DCE_net.load_state_dict(torch.load('DCE/Epoch99.pth'))
		for file_name in tqdm(file_list):
			test_list = glob.glob(filePath+file_name+"/*") 
			for image in test_list:
				if not os.path.exists(image.replace('train_clip','train_clip_enhanced')):
					lowlight(DCE_net, image)