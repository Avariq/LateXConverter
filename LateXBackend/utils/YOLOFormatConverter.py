import shutil
import os
import json

init_dataset_path = 'D:/YOLOFormatConverter/AidaDS'

dest_dataset_path = 'D:/YOLOFormatConverter/AidaDSYolov8XS'
dest_dataset_train_path = f'{dest_dataset_path}/train'
dest_dataset_valid_path = f'{dest_dataset_path}/valid'
dest_dataset_test_path = f'{dest_dataset_path}/test'

# forming batch paths

batch_paths = []
for folder in os.listdir(init_dataset_path):
	if folder != 'extras':
		batch_paths.append(init_dataset_path + f'/{folder}')

# start converting
curr_batch = 0
batches_to_convert = [6]  # starts from 1
for batch_path in batch_paths:
	curr_batch += 1

	if curr_batch not in batches_to_convert:
		continue

	with open(f'{batch_path}/JSON/kaggle_data_{batch_path.split("batch_")[1]}.json') as conf_file:
		conf_dict = json.load(conf_file)

	for idx in range(len(conf_dict)):
		print(f'Batch: {batch_path}. Progress: {((idx+1)/10000) * 100}%')
		if idx < 8000:
			curr_dest_path = dest_dataset_train_path
		elif 8000 <= idx < 9350:
			curr_dest_path = dest_dataset_valid_path
		else:
			curr_dest_path = dest_dataset_test_path

		image_width, image_height = conf_dict[idx]['image_data']['width'], conf_dict[idx]['image_data']['height']

		image_filename = conf_dict[idx]['filename']
		image_uuid = conf_dict[idx]['uuid']

		x_mins, y_mins, x_maxs, y_maxs = conf_dict[idx]['image_data']['xmins_raw'], \
										 conf_dict[idx]['image_data']['ymins_raw'], \
										 conf_dict[idx]['image_data']['xmaxs_raw'], \
										 conf_dict[idx]['image_data']['ymaxs_raw']

		class_maps = conf_dict[idx]['image_data']['visible_char_map']

		lines = []
		for bbox_idx in range(len(x_mins)):
			x_center = (x_mins[bbox_idx] + x_maxs[bbox_idx]) / 2
			y_center = (y_mins[bbox_idx] + y_maxs[bbox_idx]) / 2

			bbox_width = x_maxs[bbox_idx] - x_mins[bbox_idx]
			bbox_height = y_maxs[bbox_idx] - y_mins[bbox_idx]

			# normalized bbox coords (x, y, w, h)
			x_center_norm = x_center / image_width
			y_center_norm = y_center / image_height

			bbox_width_norm = bbox_width / image_width
			bbox_height_norm = bbox_height / image_height

			# class idx
			class_idx = class_maps[bbox_idx] - 1

			lines.append(f'{class_idx} {x_center_norm} {y_center_norm} {bbox_width_norm} {bbox_height_norm}')

		with open(f'{curr_dest_path}/labels/{image_uuid}.txt', 'w') as out_f:
			out_f.write('\n'.join(lines))

		curr_image_filepath = f'{batch_path}/background_images/{image_filename}'
		dest_image_filepath = f'{curr_dest_path}/images/{image_filename}'

		try:
			shutil.copy(curr_image_filepath, dest_image_filepath)
		except Exception as e:
			print(f'Something went wrong: {e}')