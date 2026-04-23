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

  d
