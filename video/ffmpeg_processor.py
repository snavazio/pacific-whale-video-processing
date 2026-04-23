"""
This class wraps the ffmpeg video processor on the command line.

All ffmpeg invocations go through subprocess.run with argv lists and shell=False.
This avoids the two issues the previous os.system implementation had:
  1. Paths with spaces / quotes / other shell metacharacters broke or, worse,
     were interpreted by the shell as command fragments (injection surface).
  2. Failures from ffmpeg were silently discarded. With check=True a non-zero
     exit from ffmpeg raises CalledProcessError so the pipeline fails fast.

Filter-expression escapes (e.g. the backslash-comma inside between(t\\,0\\,10))
are intentionally preserved: they're consumed by ffmpeg's own filter parser,
not by the shell, so they're still required even with shell=False.
"""

import math
import os
import subprocess
from video.video_processor_base import video_processor_base

class ffmpeg_processor(video_processor_base):
  def __init__(self):
    video_processor_base.__init__(self)

  def scale_video(self, original_video, scaled_video, scale):
    """
    Scales a video (original) to a new video (scaled) into
    given pixel dimensions (scale x scale)
    """
    subprocess.run(
      ["ffmpeg", "-i", original_video, "-vf", "scale={0}:{0}".format(scale), scaled_video],
      check=True,
    )

  def get_per_second_frames(self, original_video, out_dir):
    """
    Extracts a frame for each second of a given video into out_dir.
    """
    subprocess.run(
      ["ffmpeg", "-i", original_video, "-r", "1/1", os.path.join(out_dir, "%03d.png")],
      check=True,
    )

  def get_frame_range_images(self, original_video, out_dir, ranges):
    """
    Extracts all the frames between the given
    timestamp ranges (tuples of start/end timestamps in second increments)
    for a given video (original_video - path to original video).

    Extracted items are written to out_dir.
    """
    if len(ranges) == 0:
      return

    q = "between(t\\,{0}\\,{1})".format(ranges[0][0], ranges[0][1])
    for r in ranges[1:]:
      q += "+between(t\\,{0}\\,{1})".format(r[0], r[1])

    subprocess.run(
      [
        "ffmpeg", "-i", original_video,
        "-vf", "select={0}".format(q),
        "-vsync", "0",
        "-frame_pts", "1",
        os.path.join(out_dir, "%03d.png"),
      ],
      check=True,
    )

  def get_frame_range_clips(self, original_video, out_dir, ranges, max_number_queries=3):
    """
    Extracts all the video clips between the given
    timestamp ranges (tuples of start/end timestamps in second increments)
    for a given video (original_video - path to original video).

    Extracted items are written to out_dir.

    Batched at most max_number_queries clips per ffmpeg invocation to keep
    CPU use bounded — multi-output ffmpeg runs can spike load significantly.
    """
    base_video_name = os.path.basename(original_video)

    number_of_clips = len(ranges)
    current_point = 0
    while current_point < number_of_clips:
      cmd = ["ffmpeg"]
      batch = ranges[current_point:min(number_of_clips, current_point + max_number_queries)]
      for r in batch:
        outfile = os.path.join(
          out_dir,
          "{0}_Surface{1}-{2}.mp4".format(base_video_name, r[0], r[1]),
        )
        # -ss/-t before each output means "seek then limit duration for this output";
        # the single -i below supplies the shared input for all outputs in this batch.
        cmd += ["-ss", str(r[0]), "-t", str(r[1] - r[0]), outfile]
      cmd += ["-i", original_video]

      subprocess.run(cmd, check=True)
      current_point += max_number_queries

  def get_frame_images(self, original_video, out_dir, frames):
    """
    Given a list of frame numbers, extracts those frames from the video (original_video).
    Writes said frames to out_dir.
    For ffmpeg extracts 500 at a time as there are command line character limits.
    """
    if len(frames) == 0:
      return

    batch_size = 500
    batches = math.ceil(len(frames)/batch_size)

    for i in range(batches):
      q = "eq(n\\,{0})".format(frames[i*batch_size])
      for frame in frames[(i*batch_size + 1):((i+1)*batch_size)]:
        q += "+eq(n\\,{0})".format(frame)

      subprocess.run(
        [
          "ffmpeg", "-i", original_video,
          "-vf", "select={0}".format(q),
          "-vsync", "0",
          "-frame_pts", "1",
          os.path.join(out_dir, "%d.jpg"),
        ],
        check=True,
      )
