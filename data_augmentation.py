import os
from glob import glob
import cv2


read_list = list()

with open("/home/jh/CUB/trainvalclasses.txt") as f:
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

    if width > height:
        if height < 256:
            width = int(width * 256 / height)
            height = 256

            image = cv2.resize(image, (height, width))

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
        if width < 256:
            width = 256
            height = int(height * 256 / width)

            image = cv2.resize(image, (height, width))

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

    i = 0

    for image in image_list:
        image_file_name = image_file_name.replace('.jpg', '')
        if not os.path.isdir(image_file_name):
            os.mkdir(image_file_name)
        write_file_name = os.path.join(image_file_name, str(i) + ".jpg")
        print(write_file_name)
        cv2.imwrite(write_file_name, image)
        i += 1