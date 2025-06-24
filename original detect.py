import os
import sys
import numpy as np
import tensorflow as tf
import numpy as np

import scipy
from scipy import misc

import skimage
from skimage import io,color


from multiprocessing import Pool

sys.path.append(".")
from utils import label_map_util
from PIL import Image #for update RWM


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        # print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

_errstr = "Mode is unknown or incompatible with input array shape."


def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    >>> from scipy.misc import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a numpy array and returns a PIL image.
    This function is only available if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image


def fast_overlay(input_image, segmentation, color=[0, 255, 0, 127]):
    """
    Overlay input_image with a hard segmentation result for two classes.
    Store the result with the same name as segmentation_image, but with
    `-overlay`.
    Parameters
    ----------
    input_image : numpy.array
        An image of shape [width, height, 3].
    segmentation : numpy.array
        Segmentation of shape [width, height].
    color: color for forground class
    Returns
    -------
    numpy.array
        The image overlayed with the segmenation
    """
    color = np.array(color).reshape(1, 4)
    shape = input_image.shape
    segmentation = segmentation.reshape(shape[0], shape[1], 1)

    output = np.dot(segmentation, color)
    # output = scipy.misc.toimage(output, mode="RGBA") #throwing errors for update
    # output=Image.fromarray(output, mode="RGBA")
    output = toimage(output, mode="RGBA") #throwing errors for update


    # background = scipy.misc.toimage(input_image)
    # background = Image.fromarray(input_image) #update RWM 
    background = toimage(input_image)

    background.paste(output, box=None, mask=output)

    return np.array(background)



def write_nanowell_info(fname, info_array):
    f = open(fname,'w')
    # Cell_ID	x	y	w	h	class	score
    for info in info_array:
        line = str(info[0]) + '\t' +str(info[1]) + '\t' + str(info[2]) + '\t' + str(info[3]) + '\t' +str(info[4]) + '\t' + str(info[5]) + '\t' + format(info[6],'.4f') + '\n'
        f.writelines(line)
    f.close()
    

def get_nanowell_images(parameter):
    PATH_TO_OUTPUT_DIR = parameter[0]
    BID = parameter[1]
    nanowell = parameter[2]
    frames = parameter[3]
    CH_MIX = parameter[4]

    # Initialize an empty array for the image stack
    Nanowell_Image_Stack = np.empty((0,))  # This will later be checked and reshaped

    for t in range(1, frames + 1):
        image = None  # Initialize image as None for conditional overlay
        for CH in ['CH0', 'CH1', 'CH2']:
            if CH in CH_MIX:
                TEST_IMAGE_PATH = "{}/{}/images/crops_8bit_s/imgNo{}{}/imgNo{}{}_t{}.tif".format(PATH_TO_OUTPUT_DIR, BID, nanowell, CH, nanowell, CH, t)
                if os.path.exists(TEST_IMAGE_PATH):
                    img_temp = io.imread(TEST_IMAGE_PATH)
                    if CH == 'CH0':
                        image = color.gray2rgb(img_temp)
                    elif CH == 'CH1' and image is not None:
                        image = fast_overlay(image, img_temp, [255,0,0,127])
                    elif CH == 'CH2' and image is not None:
                        image = fast_overlay(image, img_temp, [0,255,0,127])
                else:
                    print("Warning: File not found {}. Skipping.".format(TEST_IMAGE_PATH))

        if image is not None:
            image_np = np.expand_dims(image, axis=0)
            if Nanowell_Image_Stack.size == 0:
                Nanowell_Image_Stack = image_np
            else:
                Nanowell_Image_Stack = np.concatenate([Nanowell_Image_Stack, image_np], axis=0)

    return Nanowell_Image_Stack

def get_block_images_parallel(PATH_TO_OUTPUT_DIR, BID, nanowell_numbers, FRAMES, CH_MIX, CORES):
    from multiprocessing import Pool

    nanowell_list = []
    for nanowell in nanowell_numbers:
        nanowell_list.append([PATH_TO_OUTPUT_DIR, BID, nanowell, FRAMES, CH_MIX])
        
    with Pool(processes=CORES) as pool:
        block_image_list = pool.map(get_nanowell_images, nanowell_list)
    
    return block_image_list





def detect_cells(PATH_TO_CKPT, PATH_TO_LABELS, Block_Image_Stacks):
    NUM_CLASSES = 2
    
    # load frozen map
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # load label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    box_result = []
    score_result = []
    class_result = []
    num_detections_result = []

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            number_image = len(Block_Image_Stacks)
            for i in range(number_image):
                # image_np = np.expand_dims(Block_Image_Stacks[i], axis = 0)
                image_np = Block_Image_Stacks[i]
                (boxes1, scores1, classes1, num_detections1) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np})
                box_result.append(boxes1)
                score_result.append(scores1)
                class_result.append(classes1)
                num_detections_result.append(num_detections1)
                
            return box_result, score_result, class_result, num_detections_result
        

def get_nanowell_numbers(OUTPUT_PATH, DATASET, BLOCK):
    meta_all_clean_fname = OUTPUT_PATH + DATASET + '/' + BLOCK + '/meta/a_centers_clean_sorted.txt'
    with open(meta_all_clean_fname, 'r') as file:
        lines = file.readlines()
    
    # Extract nanowell numbers from the last element of each line
    nanowell_numbers = [int(line.split()[-1]) for line in lines]
    
    return nanowell_numbers

        

def parse_write(Args):
    # Unpack arguments
    PATH_TO_OUTPUT_DIR, BID, Nanowell, FRAMES, boxes, scores, classes, MAXIMUM_CELL_DETECTED, Detector_Type, NANOWELL_SIZE = Args

    # Directory path for metadata based on detector type
    meta_fname_prefix = os.path.join(PATH_TO_OUTPUT_DIR, BID, 'labels', 'DET', Detector_Type, 'raw')
    # Ensure the directory exists
    os.makedirs(meta_fname_prefix, exist_ok=True)
    
    meta_fname_prefix2 = os.path.join(meta_fname_prefix, 'imgNo' + str(Nanowell))
    # Ensure the sub-directory exists
    os.makedirs(meta_fname_prefix2, exist_ok=True)
    
    for t in range(FRAMES):
        meta_fname = os.path.join(meta_fname_prefix2, 'imgNo' + str(Nanowell) + '_t' + str(t+1) + '.txt')
        meta_array = []

        # Check if there are detected cells in the current frame to avoid IndexError
        if t < len(boxes) and len(boxes[t]) > 0:
            for cell_count in range(min(len(boxes[t]), MAXIMUM_CELL_DETECTED)):
                ymin, xmin, ymax, xmax = boxes[t][cell_count]
                x = int(xmin * NANOWELL_SIZE)
                y = int(ymin * NANOWELL_SIZE)
                w = int((xmax - xmin) * NANOWELL_SIZE)
                h = int((ymax - ymin) * NANOWELL_SIZE)
                cell_class = classes[t][cell_count] if cell_count < len(classes[t]) else -1
                cell_score = scores[t][cell_count] if cell_count < len(scores[t]) else -1
                
                temp_meta = [cell_count, x, y, w, h, cell_class, cell_score]
                meta_array.append(temp_meta)
            
        write_nanowell_info(meta_fname, meta_array)




def DT_Cell_Detector(DEEP_TIMING_HOME, OUTPUT_PATH, DATASET, BLOCKS, FRAMES, CH_MIX, Detector_Type, MAXIMUM_CELL_DETECTED, NANOWELL_SIZE, CORES):
    for BLOCK in BLOCKS:
        
        print("Detecting Cells in " + BLOCK + " ...... ")
        
        # Get Stack Image List of a Block [stack1, stack2, ....], stack1 shape:[72, 281, 281, 3]
        PATH_TO_OUTPUT_DIR = OUTPUT_PATH + DATASET + '/'
        Nanowells = get_nanowell_numbers(OUTPUT_PATH, DATASET, BLOCK)
        Block_Image_Stacks = get_block_images_parallel(PATH_TO_OUTPUT_DIR, BLOCK, Nanowells, FRAMES, CH_MIX, CORES)
        
        
        # detect Cells in Block_Image_Stacks
        if ('CH0' in CH_MIX) and ('CH1' in CH_MIX) and ('CH2' in CH_MIX):
            if Detector_Type == 'FRCNN-Fast':
                PATH_TO_CKPT = DEEP_TIMING_HOME + 'DT2-detector/Cell/checkpoints/FRCNN-Fast/MIX/frozen_inference_graph.pb'
                PATH_TO_LABELS = DEEP_TIMING_HOME + 'DT2-detector/Cell/checkpoints/FRCNN-Fast/TIMING-CELL-FRCNN.pbtxt'
                boxes, scores, classes, num_detections = detect_cells(PATH_TO_CKPT, PATH_TO_LABELS, Block_Image_Stacks)
                
            if Detector_Type == 'SSD':
                PATH_TO_CKPT = DEEP_TIMING_HOME + 'DT2-detector/Cell/checkpoints/SSD/MIX/frozen_inference_graph.pb'
                PATH_TO_LABELS = DEEP_TIMING_HOME + 'DT2-detector/Cell/checkpoints/SSD/TIMING-CELL-FRCNN.pbtxt'
                boxes, scores, classes, num_detections = detect_cells(PATH_TO_CKPT, PATH_TO_LABELS, Block_Image_Stacks)
        
        # parse and write cell detection
        parameter_list = []
        for i, nanowell in enumerate(Nanowells):
            temp = []
            temp.append(PATH_TO_OUTPUT_DIR)
            temp.append(BLOCK)
            temp.append(nanowell)
            temp.append(FRAMES)
            temp.append(boxes[i])
            temp.append(scores[i])
            temp.append(classes[i])
            temp.append(MAXIMUM_CELL_DETECTED)
            temp.append(Detector_Type)
            temp.append(NANOWELL_SIZE)
            
            parameter_list.append(temp)
            
        pool = Pool(processes = CORES)
        
        pool.map(parse_write, parameter_list)
        
        pool.close()
    
    
    
    
