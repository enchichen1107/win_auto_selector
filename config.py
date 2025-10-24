MODEL_NAME = "init"
FONT_PATH = "./assets/NotoSansTC-Medium.otf"

# Training and testing face data folders
DIR_FACE_TRAIN_0 = "./models/data/train/0/"
DIR_FACE_TRAIN_1 = "./models/data/train/1/"
DIR_FACE_TEST_0 = "./models/data/test/0/"
DIR_FACE_TEST_1 = "./models/data/test/1/"

NEW_DATA = [DIR_FACE_TRAIN_0, DIR_FACE_TRAIN_1, DIR_FACE_TEST_0, DIR_FACE_TEST_1]
NEW_SRCS = ["./models/data/train/", "./models/data/test/"]
NEW_TRAIN_SRCS = [DIR_FACE_TRAIN_0, DIR_FACE_TRAIN_1]

# Buffer folders
DIR_FACE_BUFFER_0 = "./models/buffer/data/0/"
DIR_FACE_BUFFER_1 = "./models/buffer/data/1/"
BUFFER_SRCS = [DIR_FACE_BUFFER_0, DIR_FACE_BUFFER_1]

# Retrieve folders
DIR_FACE_RETRIEVE_TRAIN_0 = "./models/retrieve/train/0/"
DIR_FACE_RETRIEVE_TRAIN_1 = "./models/retrieve/train/1/"
DIR_FACE_RETRIEVE_TEST_0 = "./models/retrieve/test/0/"
DIR_FACE_RETRIEVE_TEST_1 = "./models/retrieve/test/1/"
DESTS = [
    DIR_FACE_RETRIEVE_TRAIN_0,
    DIR_FACE_RETRIEVE_TEST_0,
    DIR_FACE_RETRIEVE_TRAIN_1,
    DIR_FACE_RETRIEVE_TEST_1
]
DIRS = ["./models/retrieve/train/", "./models/retrieve/test/"]