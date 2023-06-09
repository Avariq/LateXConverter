import shutil

from ultralytics import YOLO
from PIL import Image
import os

model = YOLO('bestYolov8.pt')

map_dict = {
	"\\alpha": 1,
	"\\approx": 2,
	"\\beta": 3,
	"\\cdot": 4,
	"\\delta": 5,
	"\\div": 6,
	"\\frac": 7,
	"\\gamma": 8,
	"\\geq": 9,
	"\\infty": 10,
	"\\int": 11,
	"\\left(": 12,
	"\\left[": 13,
	"\\left\\{": 14,
	"\\left|": 15,
	"\\leq": 16,
	"\\neq": 17,
	"\\pi": 18,
	"\\pm": 19,
	"\\prime": 20,
	"\\right)": 21,
	"\\right]": 22,
	"\\right|": 23,
	"\\sqrt": 24,
	"\\theta": 25,
	"+": 26,
	",": 27,
	"-": 28,
	".": 29,
	"/": 30,
	"0": 31,
	"1": 32,
	"2": 33,
	"3": 34,
	"4": 35,
	"5": 36,
	"6": 37,
	"7": 38,
	"8": 39,
	"9": 40,
	";": 41,
	"<": 42,
	"=": 43,
	">": 44,
	"A": 45,
	"C": 46,
	"F": 47,
	"G": 48,
	"H": 49,
	"L": 50,
	"a": 51,
	"b": 52,
	"c": 53,
	"d": 54,
	"e": 55,
	"f": 56,
	"g": 57,
	"h": 58,
	"k": 59,
	"n": 60,
	"p": 61,
	"r": 62,
	"s": 63,
	"t": 64,
	"u": 65,
	"v": 66,
	"w": 67,
	"x": 68,
	"y": 69,
	"z": 70,
	"\\lim_": 71,
	"\\log": 72,
	"\\cot": 73,
	"\\csc": 74,
	"\\to": 75,
	"\\cos": 76,
	"\\sec": 77,
	"\\sin": 78,
	"\\ln": 79,
	"\\tan": 80,
	"\\arcsin": 81,
	"\\arccos": 82,
	"\\arctan": 83,
	"\\arccot": 84,
	"\\arccsc": 85,
	"\\arcsec": 86,
	"\\textup{Undefined}": 87,
	"\\textup{Does not exist}": 88,
	"\\textup{True}": 89,
	"\\textup{False}": 90,
	"\\stackrel{\\textup{H}}{=}": 91
}

map_dict = {v: k for k, v in map_dict.items()}


def detect_bboxes(file_path):
	res = model(file_path)

	box_predictions = res[0].boxes
	bboxes = box_predictions.xyxy

	current_dir = os.getcwd()
	dest_dir = current_dir + '/temp'

	if os.path.exists(dest_dir):
		shutil.rmtree(dest_dir)

	os.mkdir(dest_dir)

	pred_list = []

	# [xmin, ymin, xmax, ymax], [class_idx, class_confidence]
	for i in range(len(bboxes)):
		pred_list.append([[bboxes[i][0].item(), bboxes[i][1].item(), bboxes[i][2].item(),
						   bboxes[i][3].item()], [box_predictions.cls[i].item(), box_predictions.conf[i].item()]])

	pred_list.sort(key=lambda x: x[0][0])  # bboxes sorted from left side of the image

	latex_out = []

	img = Image.open(file_path)  # init image
	curr = 0

	for pred in pred_list:
		cropped = img.crop((pred[0][0], pred[0][1], pred[0][2], pred[0][3]))
		cropped.save(f'{dest_dir}/bbox_{curr}.jpg')
		curr += 1

		latex_out.append(map_dict[pred[1][0] + 1])

	return ' '.join(latex_out)
