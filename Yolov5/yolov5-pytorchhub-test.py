import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt 

image = cv2.imread('./data/images/3.jpg')

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5x.pt')

time_list = []
for i in range(22):
	results = model(image)
	time = results.print()
	if i > 1:
		time_list.append(time[1])

avg_time = int(sum(time_list)/len(time_list))
print('-------------------------------------')
print('Inference time : ', avg_time)
print('-------------------------------------')