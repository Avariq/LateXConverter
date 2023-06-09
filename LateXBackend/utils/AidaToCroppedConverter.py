import os
import glob
import json
from pprint import pprint

from PIL import Image

dataset = os.path.join("dataset", "AIDA")
dataset2 = os.path.join("dataset", "AIDA2")
try:
    os.makedirs(dataset2)
except FileExistsError:
    pass

counter = 0

for root, dirs, files in os.walk(dataset):
    for batchdir in dirs:
        if batchdir.startswith("batch"):

            print(batchdir)
            images_path = os.path.join(root, batchdir, "background_images")
            json_path = os.path.join(root, batchdir, "JSON")
            json_file = glob.glob(os.path.join(json_path, "*.json"))[0]
            with open(json_file, "r", encoding="utf-8") as fp:
                batch_info = json.load(fp)
            for obj in batch_info:
                del obj["image_data"]["png_masks"]
                uid = obj['uuid']
                idata = obj["image_data"]
                latex_chars = idata['visible_latex_chars']
                classes_chars = idata['visible_char_map']
                xmins = idata['xmins_raw']
                xmaxs = idata['xmaxs_raw']
                ymins = idata['ymins_raw']
                ymaxs = idata['ymaxs_raw']
                # pprint(obj)
                img_path = os.path.join(images_path, uid + ".jpg")
                img = Image.open(img_path)
                for i, ch in enumerate(latex_chars):
                    cropped = img.crop((xmins[i], ymins[i], xmaxs[i], ymaxs[i]))
                    cropped.save(os.path.join(dataset2, f"{classes_chars[i]}_{counter}.jpg"))
                    cropped.close()
                    counter += 1
                img.close()

