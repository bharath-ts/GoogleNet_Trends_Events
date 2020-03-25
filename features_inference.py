from os.path import join
import numpy as np
from scipy.misc import imread, imresize
from googlenet_infer import GoogleNet
from utils.manifest_api import Annotations
import glob
import json 

#####################################################################################################################3	
attributes = ['clothing_pattern', 'major_color', 'wearing_necktie', 'collar_presence', 'wearing_scarf',\
    'sleeve_length', 'neckline_shape', 'clothing_category', 'wearing_jacket', 'wearing_hat', 'wearing_glasses', 'multiple_layers']

categories = [['Solid', 'Striped', 'Graphics', 'Floral', 'Plaid', 'Spotted'], \
    ['Black', 'Red', 'White', 'Blue', 'Gray', 'Yellow', 'More than 1 color', 'Brown', 'Green', 'Pink', 'Orange', 'Purple', 'Cyan'], \
    ['No', 'Yes'], ['No', 'Yes'], ['No', 'Yes'], ['Short sleeve', 'Long sleeve', 'No sleeve'], ['Round', 'Folded', 'V-shape'], \
    ['Dress', 'Outerwear', 'T-shirt', 'Suit', 'Shirt', 'Sweater', 'Tank top'], ['No', 'Yes'], ['No', 'Yes'], ['No', 'Yes'], \
    ['One layer', 'Multiple layers']]

print(len(attributes), len(categories))
##################################################################################

glnet = GoogleNet(attributes, categories, 224)

input_dir= '/media/chintu/bharath_ext_hdd/Bharath/From Home/Trend Analysis/data/without_location/'
images=[]
im_list=[]
for i in glob.glob(input_dir+'*.jpg'):
    images.append(i)
    im_list.append(i.replace(input_dir,''))

print("images",len(images))

# output_dict={}
res=[]

for i in range(0,len(images)):

    try:
        output_dict={}
        images1 = imread(images[i])
        images1 = np.expand_dims(imresize(images1, (224, 224)), axis=0)
        
        features = glnet.get_features(images1)
        
        output_dict['image_name']=im_list[i]
        output_dict['features']=features.tolist()[0]
        # print(i, images[i], im_list[i], output_dict['image_name'])    
        res.append(output_dict)
        # print("res", res[0], res[10])
        if i%100==0:
            print("images done", i)

    except:
        print("Exception occurred")
        pass
print(len(res))


with open('/media/chintu/bharath_ext_hdd/Bharath/From Home/Trend Analysis/data/filtered_features_1024.json','w') as outfile:
    json.dump(res, outfile)

