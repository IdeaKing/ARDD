####################################################
#                                                  #
# ARDD Configurations for Object Detection.        #
# Created by Thomas Chia and Cindy Wu              #
# Medical Research by Sreya Devarakonda            #
# Created for the 2021 Congressional App Challenge #
# Winning "webapp" of Virginia's 10th District     #
#                                                  #
#################################################### 

YOLO_DARKNET_WEIGHTS        = "./object_detection_ai/configuration_files/yolov3.weights"
YOLO_TF_WEIGHTS             = "./object_detection_ai/checkpoints/coco_weights/coco_weights"
YOLO_COCO_CLASSES           = "./object_detection_ai/configuration_files/coco.names"
YOLO_CUSTOM_WEIGHTS         = False 
YOLO_STRIDES                = [8, 16, 32]
YOLO_IOU_LOSS_THRESH        = 0.5
YOLO_ANCHOR_PER_SCALE       = 3
YOLO_MAX_BBOX_PER_SCALE     = 100
YOLO_INPUT_SIZE             = 416
YOLO_ANCHORS                = [[[10,  13], [16,   30], [33,   23]],
                               [[30,  61], [62,   45], [59,  119]],
                               [[116, 90], [156, 198], [373, 326]]]
TRAIN_CLASSES               = "./configurations/TEMP_CLASSES.txt"
TRAIN_WEIGHTS               = "./configurations/v3_yolo_weights/A-EYE_weights_v3"
TRAIN_MODEL_NAME            = "yolov3_fundus_test_1"
TRAIN_SAVE_BEST_ONLY        = True 
TRAIN_SAVE_CHECKPOINT       = True 
TRAIN_LOAD_IMAGES_TO_RAM    = True 
TRAIN_DATA_AUG              = True 
TRAIN_TRANSFER              = True
TRAIN_FROM_CHECKPOINT       = False 
TRAIN_BATCH_SIZE            = 8
TRAIN_INPUT_SIZE            = 416
TRAIN_LR_INIT               = 1e-4
TRAIN_LR_END                = 1e-6
TRAIN_WARMUP_EPOCHS         = 2
TRAIN_EPOCHS                = 35
TEST_ANNOT_PATH             = "./object_detection_ai/image_dir/allweights_test.txt"
TEST_DECTECTED_IMAGE_PATH   = "./object_detection_ai/testing_objdet/test_output"
TEST_DATA_AUG               = True
TEST_BATCH_SIZE             = 4
TEST_INPUT_SIZE             = 416
TEST_SCORE_THRESHOLD        = 0.3
TEST_IOU_THRESHOLD          = 0.45

