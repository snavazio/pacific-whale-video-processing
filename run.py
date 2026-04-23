from datetime import timedelta
import math
import random
import sys
import time
import multiprocessing
import os
from PIL import Image
from numpy import asarray

from model.model_pytorch import model_pytorch
from video.ffmpeg_processor import ffmpeg_processor

def frames_to_ts(frame_number, fps=30):
  span = timedelta(milliseconds= frame_number*1000/fps)
  return "{:02d}{:02d}{:02d}".format(int(span.seconds/60), int(span.seconds%60), int(span.microseconds/1000))

def seconds_to_ts(seconds_number):
  span = timedelta(seconds= seconds_number)
  return "{:02d}:{:02d}".format(int(span.seconds/60), int(span.seconds%60))

def get_final_image_name(original_vid, out_dir, frame, number):
  # File name templae: Species_Location_Date_OriginalVideoframe#_screencapture#_VideoTimeStamp(MMSS)
  # File name example: MN_HI_20200824_0001_001_0301
  original_vid_name = os.path.basename(origin
