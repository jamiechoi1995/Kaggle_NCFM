import json
import os
import shutil
root_train = '/home/wcai/Downloads/ai_challenger_scene_validation_20170908/scene_validation_images_20170908'

f = file('/home/wcai/Downloads/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json')
jsonobj = json.load(f)

for num in range(0,80):
	# print str(num)
	num = str(num)
	if num not in os.listdir(root_train):
		os.mkdir(os.path.join(root_train, num))

# total_images = os.listdir(root_train)
# print total_images
	for json in jsonobj:
		if str(json["label_id"]) == num:
			source = os.path.join(root_train, json["image_id"])
			target = os.path.join(root_train, num, json["image_id"])
			shutil.move(source, target)
		# print json["image_id"]

