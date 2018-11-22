import os
from glob import glob
import numpy as np


def alphabetToChar(alphabet):
    if alphabet == 'a':
        return 0
    elif alphabet == 'b':
        return 1
    elif alphabet == 'c':
        return 2
    elif alphabet == 'd':
        return 3
    elif alphabet == 'e':
        return 4
    elif alphabet == 'f':
        return 5
    elif alphabet == 'g':
        return 6
    elif alphabet == 'h':
        return 7
    elif alphabet == 'i':
        return 8
    elif alphabet == 'j':
        return 9
    elif alphabet == 'k':
        return 10
    elif alphabet == 'l':
        return 11
    elif alphabet == 'm':
        return 12
    elif alphabet == 'n':
        return 13
    elif alphabet == 'o':
        return 14
    elif alphabet == 'p':
        return 15
    elif alphabet == 'q':
        return 16
    elif alphabet == 'r':
        return 17
    elif alphabet == 's':
        return 18
    elif alphabet == 't':
        return 19
    elif alphabet == 'u':
        return 20
    elif alphabet == 'v':
        return 21
    elif alphabet == 'w':
        return 22
    elif alphabet == 'x':
        return 23
    elif alphabet == 'y':
        return 24
    elif alphabet == 'z':
        return 25
    elif alphabet == '0':
        return 26
    elif alphabet == '1':
        return 27
    elif alphabet == '2':
        return 28
    elif alphabet == '3':
        return 29
    elif alphabet == '4':
        return 30
    elif alphabet == '5':
        return 31
    elif alphabet == '6':
        return 32
    elif alphabet == '7':
        return 33
    elif alphabet == '8':
        return 34
    elif alphabet == '9':
        return 35
    elif alphabet == '-':
        return 36
    elif alphabet == ',':
        return 37
    elif alphabet == ';':
        return 38
    elif alphabet == '.':
        return 39
    elif alphabet == '!':
        return 40
    elif alphabet == '?':
        return 41
    elif alphabet == ':':
        return 42
    elif alphabet == "'":
        return 43
    elif alphabet == '"':
        return 44
    elif alphabet == '/':
        return 45
    elif alphabet == '\\':
        return 46
    elif alphabet == '|':
        return 47
    elif alphabet == '_':
        return 48
    elif alphabet == '@':
        return 49
    elif alphabet == '#':
        return 50
    elif alphabet == '$':
        return 51
    elif alphabet == '%':
        return 52
    elif alphabet == '^':
        return 53
    elif alphabet == '&':
        return 54
    elif alphabet == '*':
        return 55
    elif alphabet == '~':
        return 56
    elif alphabet == '`':
        return 57
    elif alphabet == '+':
        return 58
    elif alphabet == '-':
        return 59
    elif alphabet == '=':
        return 60
    elif alphabet == '<':
        return 61
    elif alphabet == '>':
        return 62
    elif alphabet == '(':
        return 63
    elif alphabet == ')':
        return 64
    elif alphabet == '[':
        return 65
    elif alphabet == ']':
        return 66
    elif alphabet == '{':
        return 67
    elif alphabet == '}':
        return 68
    elif alphabet == ' ':
        return 69
    else:
        return -1 # change this part before distribute


read_list = list()

with open("/home/jh/CUB/trainvalclasses.txt") as f: # before upload final code, replace here to load module
    while True:
        line = f.readline()
        if not line: break
        read_list.append(line)

    text_file_name_list = list()

    for class_name in read_list:
        class_name = class_name.replace("\n", "")
        new_text_list = sorted(glob(os.path.join("/home/jh/CUB/text_c10",
                                                 class_name,
                                                 "*.txt")))
        text_file_name_list = text_file_name_list + new_text_list

    for text_file_name in text_file_name_list:
        read_lines = list()

        with open(text_file_name, "r") as f:
            while True:
                line = f.readline()
                if not line: break
                line = line.replace("\n", "")
                read_lines.append(line)

        one_hot_representation = np.zeros((10, 201, 1, 70), dtype=int)

        i = 0
        for line in read_lines:
            j = 0
            for character in line:
                print(character)
                if j > 200:
                    break
                one_hot_position = alphabetToChar(character)
                if one_hot_position == -1:
                    break
                one_hot_representation[i, j, 0, one_hot_position] = 1
                j += 1
            i += 1

        text_file_name = text_file_name.replace('.txt', '.npy')
        np.save(text_file_name, one_hot_representation)