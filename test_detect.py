from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util, config_util
from object_detection.builders import model_builder
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import requests, os
import shutil, sys

MODEL_NAME = 'SSD_MobileNet_V2_FPNLite_320x320'
LABEL_MAP_NAME = 'label_map.pbtxt'

@tf.function
def detect_image(image):
  image, shapes = detection_model.preprocess(image)
  detections = detection_model.predict(image, shapes)
  detections_dict = detection_model.postprocess(detections,shapes)
  return detections_dict

def download_image(url):
    ## Set up the image URL and filename
    image_url = url
    filename = image_url.split("/")[-1]

    # Open the url image, set stream to True, this will return the stream content.
    r = requests.get(image_url, stream = True)

    # Check if the image was retrieved successfully
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True
        
        # Open a local file with wb ( write binary ) permission.
        with open(filename,'wb') as f:
            shutil.copyfileobj(r.raw, f)
            
        print('Image sucessfully Downloaded: ',filename)
        return filename
    else:
        print('Image Couldn\'t be retreived')
        return None

paths = {
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',MODEL_NAME),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations')
}

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# Load pipeline config and build detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoints
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'],'ckpt-4')).expect_partial()

# TODO Change to argument from script
img_url = sys.argv[1]
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
TEST_IMAGE_PATH = download_image(img_url)

img = np.array(cv.imread(TEST_IMAGE_PATH))
input_tensor = tf.convert_to_tensor(np.expand_dims(img, 0), dtype=tf.float32)
# print(input_tensor)

result = detect_image(input_tensor)
num_detections = int(result.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy() for key, value in result.items()}
detections['num_detections'] = num_detections
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

img_with_detections = img.copy()
label_offset=1

viz_utils.visualize_boxes_and_labels_on_image_array(
    img_with_detections,
    detections['detection_boxes'],
    detections['detection_classes']+label_offset,
    detections['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    min_score_thresh = 0.4,
    agnostic_mode=False
)

SAVE_PATH = TEST_IMAGE_PATH.split('.')[0]+'_detections.png'
cv.imwrite(SAVE_PATH, cv.cvtColor(img_with_detections,cv.COLOR_BGR2RGB))

# plt.figure(figsize=(14,10))
# plt.imshow(cv.cvtColor(img_with_detections,cv.COLOR_BGR2RGB))
# plt.show()