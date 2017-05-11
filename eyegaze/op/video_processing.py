from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import scipy
import numexpr

def _load_and_decode_video_frms(frm_paths, length, image_format):
    """
    load video frame images from frm_paths.

    Args:
        frm_paths: list of string tensors.
        length: size of video frames.
        image_format: jpeg or png

    return:
        4-D tensor; [length, width, height, channel]
    """
    #indices = tf.lin_space(0.0, tf.cast(tf.shape(frm_paths)[0] - 1, tf.float32), num=length)
    #selected_frms = tf.nn.embedding_lookup(frm_paths, tf.cast(indices, tf.int32))
    selected_frms = frm_paths
    frm_list = []
    for i in range(length):
        frm_path = selected_frms[i]
        encoded_frm = tf.read_file(frm_path)
        if image_format == "jpeg":
            frm = tf.image.decode_jpeg(encoded_frm, channels=3)
        elif image_format == "png":
            frm = tf.image.decode_png(encoded_frm, channels=3)
        else:
            raise ValueError("Invalid image format: %s" % image_format)
        frm = tf.image.convert_image_dtype(frm, dtype=tf.float32)
        frm_list.append(frm)

    return tf.stack(frm_list, 0)

def distort_image(image, thread_id):
  """Perform random distortions on an image.
  Args:
    image: A float32 Tensor of shape [height, width, 3] with values in [0, 1).
    thread_id: Preprocessing thread id used to select the ordering of color
      distortions. There should be a multiple of 2 preprocessing threads.
  Returns:
    distorted_image: A float32 Tensor of shape [height, width, 3] with values in
      [0, 1].
  """
  # Randomly flip horizontally.
  with tf.name_scope("flip_horizontal", values=[image]):
    image = tf.image.random_flip_left_right(image)

  # Randomly distort the colors based on thread id.
  color_ordering = thread_id % 2
  with tf.name_scope("distort_color", values=[image]):
    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.032)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.032)

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)

  return image

def process_image(encoded_image,
                  is_training,
                  height,
                  width,
                  resize_height=346,
                  resize_width=346,
                  thread_id=0,
                  augment=False,
                  image_format="jpeg"):
  """Decode an image, resize and apply random distortions.

  In training, images are distorted slightly differently depending on thread_id.

  Args:
    encoded_image: String Tensor containing the image.
    is_training: Boolean; whether preprocessing for training or eval.
    height: Height of the output image.
    width: Width of the output image.
    resize_height: If > 0, resize height before crop to final dimensions.
    resize_width: If > 0, resize width before crop to final dimensions.
    thread_id: Preprocessing thread id used to select the ordering of color
      distortions. There should be a multiple of 2 preprocessing threads.
    image_format: "jpeg" or "png".

  Returns:
    A float32 Tensor of shape [height, width, 3] with values in [-1, 1].

  Raises:
    ValueError: If image_format is invalid.
  """
  # Helper function to log an image summary to the visualizer. Summaries are
  # only logged in thread 0.
  def image_summary(name, image):
    if not thread_id:
      tf.summary.image(name, tf.expand_dims(image, 0))

  # Decode image into a float32 Tensor of shape [?, ?, 3] with values in [0, 1).
  encoded_image = tf.read_file(encoded_image)
  with tf.name_scope("decode", values=[encoded_image]):
    if image_format == "jpeg":
      image = tf.image.decode_jpeg(encoded_image, channels=3)
    elif image_format == "png":
      image = tf.image.decode_png(encoded_image, channels=3)
    else:
      raise ValueError("Invalid image format: %s" % image_format)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image_summary("original_image", image)

  # Resize image.
  assert (resize_height > 0) == (resize_width > 0)
  if resize_height:
    image = tf.image.resize_images(image,
                                   size=[resize_height, resize_width],
                                   method=tf.image.ResizeMethod.BILINEAR)

  # Crop to final dimensions.
  if is_training and augment:
    image = tf.random_crop(image, [height, width, 3])
  else:
    # Central crop, assuming resize_height > height, resize_width > width.
    image = tf.image.resize_image_with_crop_or_pad(image, height, width)

  image_summary("resized_image", image)

  # Randomly distort the image.
  if is_training and augment:
    image = distort_image(image, thread_id)

  image_summary("final_image", image)

  # Rescale to [-1,1] instead of [0, 1]
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image

def process_video_pupil(frm_paths,
                        is_training,
                        step_length,
                        height,
                        width,
                        resize_height=299,
                        resize_width=299,
                        thread_id=0,
                        image_format="jpeg",
                        mean=None):
    """Decode video frames, resize.

    Args:
        frm_paths: list of string tensors.
        pupil_size: list of pupil size
        is_training: Boolean; whether preprocessing for training or eval.
        height: Height of the output image.
        width: Width of the output image.
        resize_height: If > 0, resize height before crop to final dimensions.
        resize_width: If > 0, resize width before crop to final dimensions.
        thread_id: Preprocessing thread id.
        image_format: "jpeg" or "png".

    Returns:
        A float32 Tensor of shape [step_length, height, width, 3] with values in [-1, 1].

    Raises:
        ValueError: If image_format is invalid
    """
    # Helper function to log an image summary to the visualizer. Summaries are
    # only logged in thread 0.
    def video_summary(name, video):
        if not thread_id:
            indices = tf.lin_space(0.0, tf.cast(tf.constant(step_length-1), tf.float32), num=5)
            sampled_frame = tf.nn.embedding_lookup(video, tf.cast(indices, tf.int32))
            tf.summary.image(name, video, max_outputs=5)

    video = _load_and_decode_video_frms(frm_paths, step_length, image_format)
    video_summary("original_video", video)

    # Resize video.
    assert (resize_height > 0) == (resize_width > 0)
    if resize_height:
        video = tf.image.resize_images(video,
                                       size=[resize_height, resize_width],
                                       method=tf.image.ResizeMethod.BILINEAR)
    video_summary("resized_video", video)


    # Central crop, assuming resize_height > height, resize_width > width.
    crop_list = []
    for i in range(step_length):
        frm = video[i, :, :, :]
        frm = tf.image.resize_image_with_crop_or_pad(frm, height, width)
        crop_list.append(frm)
    video = tf.subtract(video, 0.5)
    video = tf.multiply(video, 2.0)
    return video


def process_video(frm_paths,
                  is_training,
                  step_length,
                  height,
                  width,
                  resize_height=112,
                  resize_width=112,
                  thread_id=0,
                  image_format="jpeg",
                  mean=None):
    """Decode video frames, resize.

    Args:
        frm_paths: list of string tensors.
        is_training: Boolean; whether preprocessing for training or eval.
        height: Height of the output image.
        width: Width of the output image.
        resize_height: If > 0, resize height before crop to final dimensions.
        resize_width: If > 0, resize width before crop to final dimensions.
        thread_id: Preprocessing thread id.
        image_format: "jpeg" or "png".

    Returns:
        A float32 Tensor of shape [step_length, height, width, 3] with values in [-1, 1].

    Raises:
        ValueError: If image_format is invalid
    """
    # Helper function to log an image summary to the visualizer. Summaries are
    # only logged in thread 0.
    def video_summary(name, video):
        if not thread_id:
            indices = tf.lin_space(0.0, tf.cast(tf.constant(step_length-1), tf.float32), num=5)
            sampled_frame = tf.nn.embedding_lookup(video, tf.cast(indices, tf.int32))
            tf.summary.image(name, video, max_outputs=5)

    video = _load_and_decode_video_frms(frm_paths, step_length, image_format)
    #video_summary("original_video", video)
    #video = tf.image.convert_image_dtype(video, dtype=tf.float32)
    video_summary("original_video", video)


    # Resize video.
    assert (resize_height > 0) == (resize_width > 0)
    if resize_height:
        video = tf.image.resize_images(video,
                                       size=[resize_height, resize_width],
                                       method=tf.image.ResizeMethod.BILINEAR)
    video_summary("resized_video", video)


    # Central crop, assuming resize_height > height, resize_width > width.
    crop_list = []
    for i in range(step_length):
        frm = video[i, :, :, :]
        frm = tf.image.resize_image_with_crop_or_pad(frm, height, width)
        crop_list.append(frm)

    if mean == None:
        mean = np.load('/data1/common_datasets/vrsumm/PretrainedModules/c3d_mean.npy')
        mean = mean / 255.0
        video = tf.subtract(tf.stack(crop_list,0), mean)
    else:
        video = tf.subtract(tf.stack(crop_list,0), mean)
    #video_summary("resized_video", video)
    video_summary("mean_sub_video", video)
    # Rescale to [-1, 1] instead of [0, 1]
    #video = tf.subtract(video, 0.5)
    #video = tf.multiply(video, 2.0)
    return video

def vid_segment(segments, mean, video, stride=8, height=280, width=280, image_format="jpeg"):
    '''
    NOTE: Segments shorter than 16 frames (C3D input) don't get a prediction
    @param segments: list of segment tuples
    @param video: moviepy VideoFileClip
    @param stride: stride of the extraction (8=50% overlap, 16=no overlap)

    RETURN
    seg_vid : model input
    '''

    #snipplet_mean = np.load
    seg_vid = []
    seg_nr=0
    frames = []
    seg_num = []
    seg_img = []
    for frame_idx, f in enumerate(video.iter_frames(dtype="float32")):
        #Returns each frame of the clip as a HxWxN np.array, where N=1 for mask clips and N=3 for RGB clips.
        vf = scipy.misc.imresize(f,
                                size=[112,112],
                                interp='bilinear')
        vf = np.array(vf,dtype='f')
        vf = numexpr.evaluate('vf / 255.0')
        if frame_idx+1 > segments[seg_nr][1]:
            seg_nr+=1
            if seg_nr+1==len(segments):
                frames.append(vf)
                break
            frames=[]
        frames.append(vf)
        # incep 299 3  c3d 112 3
        if len(frames)==16:
            frm_list = []
            for i in range(16):#range(16):
                frm = frames[i] ### tf.decode(VideoFileClip frame type)
                frm_list.append(frm)

                #tf.convert_to_tensor(frm)
                #frm = tf.image.convert_image_dtype(frm, dtype=tf.float32)
                #video = tf.subtract(tf.stack(frm_list,0), mean)
            f = scipy.misc.imresize(f,size=[280,280]) # image version..;;
            f = numexpr.evaluate('((f - 255.0/2.0) / 255.0)*2.0')
            seg_img.append(f)
            seg_vid.append(np.array(frm_list, dtype='f') - mean)
            seg_num.append(seg_nr)
            frames = frames[stride:] # shift by 'stride' frames
    if not seg_num[-1] == seg_nr:
        remain_len = len(frames)
        seg_vid.append(np.stack([frames[i%remain_len] for i in range(16)])-mean )
        seg_img.append((scipy.misc.imresize(f,size=[width,height])/255.0 -0.5)*2.0 )
        seg_num.append(seg_nr)
    return seg_vid, seg_img, seg_num
