# LateXConverter

## Short Description
This project uses YOLOv8 model trained on Aida public dataset (70000 images out of 100000) to detect and classify handwritten math charachters in order to convert them to LateX string.

![image](https://github.com/Avariq/LateXConverter/assets/48154142/d8445161-0543-450c-b514-c209eab762b0)

## Training
Every batch (10000 images) was iterated for 12 times (epochs) to achieve better precision/recall scores.

We also used a pretrained yolov8n.pt model to achieve better segmentation (COCO dataset) and basic classification (ImageNet)

See [Detection Docs](https://docs.ultralytics.com/tasks/detect/) for usage examples with pretrained Yolov8 models.

| Model                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2               | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9               | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7               | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2               | 257.8             |

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](http://cocodataset.org) dataset.
  <br>Reproduce by `yolo val detect data=coco.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance.
  <br>Reproduce by `yolo val detect data=coco128.yaml batch=1 device=0|cpu`

  
## Model performance
![results](https://github.com/Avariq/LateXConverter/assets/48154142/cb1965b4-9eb3-4387-bd73-02a7ef38736e)
![F1_curve (1)](https://github.com/Avariq/LateXConverter/assets/48154142/8ca2b9a2-516d-4c36-b64e-e3a4ac1a8141)
  
## Dataset

### Aida overview
  
The Aida Calculus Math Handwriting Recognition Dataset consists of 100,000 images in 10 batches. Each image contains a photo of a handwritten calculus math expression (specifically within the topic of limits) written with a dark utensil on plain paper. 

Each image is accompanied by ground truth math expression in LaTeX as well as bounding boxes and pixel-level masks per character. All images are synthetically generated.
  
  
![train_batch0](https://github.com/Avariq/LateXConverter/assets/48154142/0a731c1f-a4d2-4b8a-8c78-6c5c8244c69b)
  
### Aida structure
  
batch_1/<br>
├── background_images/<br>
│ &emsp; ├── filename1.jpeg<br>
│ &emsp; ├── filename2.jpeg<br>
....<br>
│ &emsp; ├── filename10000.jpeg<br>
└── JSON/<br>
│ &emsp; ├── data_1.json<br>
batch_2/<br>
├── background_images/<br>
└── JSON/<br>
.....<br>
batch_10/<br>
extras/<br>
├── latex_unicode_map.json<br>
├── visible_char_map.json<br>
├── single_example/<br>
│ &emsp; ├── example_img.jpeg<br>
│ &emsp; ├── example.json<br>
│ &emsp; ├── Roboto-Regular.ttf<br>
│ &emsp; ├── visible_char_map_colors.json<br>


Each batch's JSON contains the following: 
{<br>
"latex":&ensp; ```String``` --the ground truth latex, <br>
"uuid":&ensp; ```String```, <br>
"unicode_str":&ensp; ```String``` --latex as unicode per latex_unicode_map.json, <br>
"unicode_less_curlies":&ensp; ```String``` --latex as unicode with curlies removed,<br>
"image_data": {<br>
&emsp;"full_latex_chars":&ensp;```List``` of Strings--each LaTeX token, <br>
&emsp;"visible_latex_chars":&ensp;```List``` of Strings--only visible LaTeX tokens, <br>
&emsp;"visible_char_map":&ensp;```List of Ints``` --LaTeX tokens per visible_char_map.json, <br>
&emsp;"width":&ensp;```Int``` --number of pixels of width of image, <br>
&emsp;"height":&ensp;```Int``` --number of pixels of height of image, <br>
&emsp;"depth":&ensp;```Int``` --channels of image RGBA, <br>
&emsp;"xmins":&ensp;```List of Floats``` --normalized position of xmin of bounding box per character, <br>
&emsp;"xmaxs":&ensp;```List of Floats```, <br>
&emsp;"ymins":&ensp;```List of Floats```, <br>
&emsp;"ymaxs":&ensp;```List of Floats```, <br>
&emsp;"xmins_raw":&ensp;```List of Ints``` --pixel position of xmin of bounding box per character, <br>
&emsp;"xmaxs_raw":&ensp;```List of Ints```,<br>
&emsp;"ymins_raw":&ensp;```List of Ints```,<br>
&emsp;"ymaxs_raw":&ensp;```List of Ints```,<br>
&emsp;"png_masks":&ensp;```List of Strings``` --the encoded mask per character<br>
&emsp;},<br>
"font":&ensp;```String``` --identifier for the generating font,<br>
"filename":&ensp;```String``` --filename of corresponding image<br>
}
  
## Aida Preprocessing

Since Aida dataset is incopatible with Yolov8's Darknet-TXT format we created a script to perform the conversion.
  
### Prerequisites
  1. Download Aida DS from [Kaggle](https://www.kaggle.com/datasets/aidapearson/ocr-data), unzip it, and save the destination (in-dest)
  2. Create a destination folder where YOLOv8 format will be stored and save that destination (out-dest)
  
  To use the script move to the ```utils``` directory and execute the following cmd line:
  ```python YOLOFormatConverter.py -in_ds_path [in-dest] -out_ds_path [out-dest] -train_perc [train_perc] -batches [b_list]```
  
  Where ```train-perc``` is floating point number in range 0-1 representing the percentage of images to go into train folder (the rest will be moved to valid folder respectively)
  And ```batches``` is the space-separated list of int number starting from ```1``` representing the corresponding batch numbers to be converted
  
  
## UI
We have made a web-based static UI with Django API backend to provide better user experience: simply use the minimalistic UI to upload your images and get the results.
  
  
## How to use
  
Firstly, create an environment and install all the requirements from requirements.txt via ```pip install -r requirements.txt```
  
After all the requirements are successfully installed you will have to start the API by invoking ```python manage.py runserver```
You have to see the following output:
  ![image](https://github.com/Avariq/LateXConverter/assets/48154142/2fed0bb1-946d-429f-a20b-0c56bb149ea8)
  
```Don't close the terminal while using API```
  
Next step is to open ```LateXConverter\front\index.html``` file and use the UI to perform converting
  
  ![image](https://github.com/Avariq/LateXConverter/assets/48154142/b73d4953-4a56-4681-b6bd-207097a8ea92)

