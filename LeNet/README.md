## Short Description

This model implements and trains LeNet5 using Aida dataset

## Training

Every batch (1856227 images) was iterated for 20 times (epochs) to achieve better precision/recall scores.

## Dataset

### Aida overview

The Aida Calculus Math Handwriting Recognition Dataset consists of 100,000 images in 10 batches. Each image contains a
photo of a handwritten calculus math expression (specifically within the topic of limits) written with a dark utensil on
plain paper.

Each image is accompanied by ground truth math expression in LaTeX as well as bounding boxes and pixel-level masks per
character. All images are synthetically generated.

![train_batch0](https://github.com/Avariq/LateXConverter/assets/48154142/79a12e34-ced6-4fec-bcb8-bd0ff9a7fc66)

Original Aida dataset is being processed into Aida2 to better suit out needs for unit detection

### Aida2 structure

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

In order to use Aida dataset for classification we need to extract all the separate images from their bounding boxes

The resulting folder should contain images in form `<class>_<id>.jpg`, where `class` is one of the 91 classes
from `map.json`

### Prerequisites

1. Install packages from requirements.txt
2. Download Aida DS from [Kaggle](https://www.kaggle.com/datasets/aidapearson/ocr-data)
3. Extract downloaded file to LeNet/dataset/AIDA
4. Run extract.py

## How to train

Run train.py, resulting checkpoints will be in `checkpoint` directory