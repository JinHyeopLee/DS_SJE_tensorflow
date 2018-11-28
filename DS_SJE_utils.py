import numpy as np


def append_nparr(arr1, arr2, axis=0):
    if arr1 is None:
        arr1 = arr2
    else:
        arr1 = np.append(arr1, arr2, axis=axis)

    return arr1


def random_select(class_num, num_entity_each_class):
    random_offset = np.random.randint(0, num_entity_each_class[class_num])
    class_base_num = 0
    for i in range(len(num_entity_each_class)):
        if i == class_num:
            break

        class_base_num += num_entity_each_class[i]

    return class_base_num + random_offset