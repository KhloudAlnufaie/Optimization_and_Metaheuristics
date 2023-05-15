import os
import argparse
import cv2
from PIL import Image
from facenet_pytorch import MTCNN
from os import listdir
import torch


def detect_face(frames, mtcnn):
	"""
	Input:
		frames: list of frames for specific video
	    mtcnn: face extraction model
	Return:
	    extract faces from each frame and return these frames
	"""

	# list to store the results
	faces_list = []
	for frame in frames:
		# read frame
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		# convert it to PIL image type
		frame = Image.fromarray(frame)
		# extract faces
		face = mtcnn(frame)
		# added to the list
		faces_list.append(face)
	return faces_list


def save_frames(frames, video_path, output_path):
	"""
	Input:
		frames: list of frames for specific video
	    video_path: input directory for videos
	    output_path: output directory for frame
	Return:
	    save the frame as jpg in output directory
	"""
	for frame in frames:
		# get the name of the video
		frame_name = video_path.split('.')[0]
		# save the frame
		cv2.imwrite(os.path.join(output_path, "{}.jpg".format(frame_name)),
		            frame.permute(1, 2, 0).int().numpy())  # save frame as JPEG file


def extract_frames(video_path):
	"""
	Input:
	    video_path: input directory for videos
	Return:
	    frame at second 5 of this video
	"""

	# read video from file
	video = cv2.VideoCapture(video_path)

	# check if the video is opened
	if not video.isOpened():
		print("Could not open video")
		return False

	# lists to store the frames
	frame_list = []
	# # ---------------------- just take the first frame ---------------
	#
	# ret, frame = video.read()
	# if ret == True:
	# 	frame_list.append(frame)
	# video.release()
	# return frame_list
	# # ---------------------- just take the first frame ---------------

	# ---------------------- just take the frame at second 5 ---------------

	# get the frame rate
	fps = video.get(cv2.CAP_PROP_FPS)

	i = 0
	stride = 5  # capture the frame by step of second
	while video.isOpened():
		# capture frame by frame
		ret, frame = video.read()

		# if the frame is read correctly
		if ret == True:

			# read only the frame of second 5
			if i == 5:
				frame_list.append(frame)
				break
			else:
				i += stride
				video.set(1, round(i * fps))
		else:
			video.release()
			break
	return frame_list


# ---------------------- just take the frame at second 5 ---------------


def generate_labels(path_in, filename, subfolder, value):
	"""
	Input:
	    path_in: input directory for images
	    filename: image file name
	    subfolder: subfolder name to this image
	    value: class value, 1 for real and 0 for fake
	Return:
	    write on the file the image path and its label
	"""
	list_of_files = listdir(path_in)
	f = open(filename, "a")
	for image in list_of_files:
		f.write("{v} {s}{i}\n".format(v=value, s=subfolder, i=image))
	f.close()


def listdiff(train_file, test_file, full_file):
	"""
	Input:
	    train_file: train file path
	    test_file: test file path
	    full_file: full data file path
	Return:
	    compare between test and full file image name /
	    if image name contain on test file, so it is belong to train set /
	    it will be saved in train file
	"""

	# open files
	train_file_f = open(train_file, 'w')
	test_file_f = open(test_file, 'r')
	full_file = open(full_file, 'r')

	# read full and test file
	full_file_list = full_file.readlines()
	test_file_list = test_file_f.readlines()

	# for all image name
	for element in full_file_list:
		# if it is existed in test file, so it will not included on train file
		if element in test_file_list:
			continue
		## if it is not existed in test file, so it will be included on train file
		train_file_f.write(element)

	# close files
	train_file_f.close()
	test_file_f.close()
	full_file.close()


def main():

	# check device
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print('Using device:', device)

	# get the needed paths
	video_root_path = args.video_root_path
	image_root_path = args.image_root_path
	label_image_root_path = args.label_image_root_path

	# create folders
	if not os.path.isdir(image_root_path):
		os.mkdir(image_root_path)

	if not os.path.isdir(label_image_root_path):
		os.mkdir(label_image_root_path)


	Celeb_real_image = os.path.join(image_root_path, 'Celeb-real')
	if not os.path.isdir(Celeb_real_image):
		os.mkdir(Celeb_real_image)

	Celeb_synthesis_image = os.path.join(image_root_path, 'Celeb-synthesis')
	if not os.path.isdir(Celeb_synthesis_image):
		os.mkdir(Celeb_synthesis_image)

	YouTube_real_image = os.path.join(image_root_path, 'YouTube-real')
	if not os.path.isdir(YouTube_real_image):
		os.mkdir(YouTube_real_image)

	# dataset sub-folders name
	sub_folders = {'Celeb-real': Celeb_real_image, 'Celeb-synthesis': Celeb_synthesis_image,
	               'YouTube-real': YouTube_real_image}

	# build the Face Extraction model
	mtcnn = MTCNN(
		margin=40,  # add more (or less) of a margin around the detected faces
		select_largest=False,  # ensure detected faces are ordered according to detection probability rather than size
		post_process=False,  # normalization prevented
		image_size=256,  # resize image
		device=device
	)


	# pass through each video, perform four basic operations:
	# frame extraction, face extraction and frame resize,  save the frames
	for folder in sub_folders.keys():
		root_path = os.path.join(video_root_path, folder)
		for video_path in os.listdir(root_path):
			# extract frames
			frames = extract_frames(os.path.join(root_path, video_path))
			# extract faces + resize frame
			frames = detect_face(frames, mtcnn)
			# save frames
			save_frames(frames, video_path, sub_folders[folder])


	# generate full labels file
	full_file = os.path.join(label_image_root_path, "full_file.txt")
	for folder in sub_folders.keys():
		root_path = os.path.join(image_root_path, folder)
		label = folder.split('-')[-1]
		print(label)
		if label == 'real':
			generate_labels(root_path, full_file, folder + '/', 0)
		elif label == 'synthesis':
			generate_labels(root_path, full_file, folder + '/', 1)

	# generate labels files for each data split
	test_file = os.path.join(label_image_root_path, "test_file.txt")
	train_file = os.path.join(label_image_root_path, "train_file.txt")
	test_video = os.path.join(video_root_path, "List_of_testing_videos.txt")
	test_video_f = open(test_video, 'r')
	test_file_f = open(test_file, 'w')
	test_video_list = test_video_f.readlines()
	for element in test_video_list:
		test_file_f.write(element.replace('mp4', 'jpg'))
	test_file_f.close()
	test_video_f.close()
	listdiff(train_file, test_file, full_file)


def parse_args():
	parser = argparse.ArgumentParser(description='Extract frame from videos')
	parser.add_argument('--video_root_path', type=str, default='./small data/Celeb-DF-v2')
	parser.add_argument('--image_root_path', type=str, default='./small data/Celeb-DF-v2-frame')
	parser.add_argument('--label_image_root_path', type=str, default='./small data/Celeb-DF-v2-frame/labels')
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = parse_args()
	main()
