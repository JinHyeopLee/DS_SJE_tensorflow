import os
from glob import glob
import cv2


train = True
read_list = list()

with open("/home/jh/CUB/trainclasses.txt") as f:
    while True:
        line = f.readline()
        if not line: break
        read_list.append(line)

image_file_name_list = list()

for class_name in read_list:
    class_name = class_name.replace("\n", "")
    new_image_list = sorted(glob(os.path.join("/home/jh/CUB/images",
                                              class_name,
                                              "*.jpg")))
    image_file_name_list = image_file_name_list + new_image_list

print(image_file_name_list)

for image_file_name in image_file_name_list:
    image = cv2.imread(image_file_name)

    width = image.shape[1]
    height = image.shape[0]

    print(width, height)

    image_list = list()

    if train:
        if width > height:
            width = int(width * 256 / height)
            height = 256

            image = cv2.resize(image, (width, height))

            image_list.append(image[0:224, 0:224])  # left-up
            image_list.append(image[0:224, -224:])  # right-up
            image_list.append(image[-224:, 0:224])  # left-down
            image_list.append(image[-224:, -224:])  # right-down
            image_list.append(image[int(height / 2) - 112:int(height / 2) + 112,
                                    int(width / 2) - 112:int(width / 2) + 112])  # center

            image = cv2.flip(image, 0)

            image_list.append(image[0:224, 0:224])  # left-up
            image_list.append(image[0:224, -224:])  # right-up
            image_list.append(image[-224:, 0:224])  # left-down
            image_list.append(image[-224:, -224:])  # right-down
            image_list.append(image[int(height / 2) - 112:int(height / 2) + 112,
                                    int(width / 2) - 112:int(width / 2) + 112])  # center

        elif height >= width:
            height = int(height * 256 / width)
            width = 256

            image = cv2.resize(image, (width, height))

            image_list.append(image[0:224, 0:224])  # left-up
            image_list.append(image[0:224, -224:])  # right-up
            image_list.append(image[-224:, 0:224])  # left-down
            image_list.append(image[-224:, -224:])  # right-down
            image_list.append(image[int(height / 2) - 112:int(height / 2) + 112,
                                    int(width / 2) - 112:int(width / 2) + 112])  # center

            image = cv2.flip(image, 0)

            image_list.append(image[0:224, 0:224])  # left-up
            image_list.append(image[0:224, -224:])  # right-up
            image_list.append(image[-224:, 0:224])  # left-down
            image_list.append(image[-224:, -224:])  # right-down
            image_list.append(image[int(height / 2) - 112:int(height / 2) + 112,
                                    int(width / 2) - 112:int(width / 2) + 112])  # center

        print(width, height)
    else:
        if width > height:
            width = int(width * 224 / height)
            height = 224

            image = cv2.resize(image, (width, height))

            image_list.append(image[:, int(width / 2) - 112:int(width / 2) + 112])

        elif height >= width:
            height = int(height * 224 / width)
            width = 224

            image = cv2.resize(image, (width, height))

            image_list.append(image[int(height / 2) - 112:int(height / 2) + 112,:])

    i = 0

    for image in image_list:
        image_file_name = image_file_name.replace('.jpg', '')
        if not os.path.isdir(image_file_name):
            os.mkdir(image_file_name)
        write_file_name = os.path.join(image_file_name, str(i) + ".jpg")
        print(write_file_name)
        cv2.imwrite(write_file_name, image)
        i += 1