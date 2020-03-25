from os.path import join
import numpy as np
from scipy.misc import imread, imresize
from googlenet_infer import GoogleNet
from utils.manifest_api import Annotations
import glob
import json 

#####################################################################################################################3	
cities_list = ['mumbai', 'kolkata'] #delhi

# #################################################################################
attributes = ['clothing_pattern', 'major_color', 'wearing_necktie', 'collar_presence', 'wearing_scarf',\
	'sleeve_length', 'neckline_shape', 'clothing_category', 'wearing_jacket', 'wearing_hat', 'wearing_glasses', 'multiple_layers']
categories = [['Solid', 'Striped', 'Graphics', 'Floral', 'Plaid', 'Spotted'], \
	['Black', 'Red', 'White', 'Blue', 'Gray', 'Yellow', 'More than 1 color', 'Brown', 'Green', 'Pink', 'Orange', 'Purple', 'Cyan'], \
	['No', 'Yes'], ['No', 'Yes'], ['No', 'Yes'], ['Short sleeve', 'Long sleeve', 'No sleeve'], ['Round', 'Folded', 'V-shape'], \
	['Dress', 'Outerwear', 'T-shirt', 'Suit', 'Shirt', 'Sweater', 'Tank top'], ['No', 'Yes'], ['No', 'Yes'], ['No', 'Yes'], \
	['One layer', 'Multiple layers']]

# #################################################################################
glnet = GoogleNet(attributes, categories, 224)


for city in cities_list:
		
	print("reading images for city:", city)
	input_dir= '/media/chintu/bharath_ext_hdd/Bharath/From Home/Trend Analysis/data/' + city + '_outputs_cropped/'
	images=[]
	im_list=[]
	for i in glob.glob(input_dir+'*.jpg'):
		images.append(i)
		im_list.append(i.replace(input_dir,''))

	print("images",len(images))

	# #break images list into batches

	step = 100

	res2=[]
	j=0
	for i in range(0,len(images),step):
		if i%step==0:
			image_batch = images[i:i+step]
			
			images1 = [imread(fname) for fname in image_batch]
			# print("before",[image.shape for image in images1])
			images1 = [imresize(image, (224, 224)) for image in images1 if len(image.shape)==3]
			# print("after",[image.shape for image in images1])
			images1 = np.array(images1)
			# print(images1.shape)
			
			classes = glnet.get_classes(images1)
			
			res=[]
			output_dict={}
			# print("classes:", len(classes))
			for img in classes:
				if j<len(im_list):
					for z in range(len(img)):
						output_dict['image_name']=im_list[j]
						output_dict[attributes[z]]=categories[z][img[z]]
					res.append(output_dict)
					output_dict={}
					j+=1
				else:
					continue
			res2.extend(res)
			# print(len(res2))
			print("# of images",i)

	with open('/media/chintu/bharath_ext_hdd/Bharath/From Home/Trend Analysis/data/features_' + city + '.json','w') as outfile:
		json.dump(res2,outfile)

	print("json created for ", city)