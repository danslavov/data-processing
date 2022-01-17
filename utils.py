from collections import OrderedDict

from cv2 import cv2 as cv2


def main():
    rgb_path = 'c:/users/admin/desktop/rgb.png'
    gray_path = 'c:/users/admin/desktop/gray.png'
    # print_times_and_improvements((0.0547, 0.8225, 0.0278, 0.9188))
    rgb_to_gray(rgb_path, gray_path)
    # tmp()


def print_times_and_improvements(values):
    loss_initial, mIoU_initial, loss_final, mIoU_final = values
    total_seconds_train = 0.0
    total_seconds_val = 0.0
    train_epoch_count = 0
    val_epoch_count = 0
    with open('log.txt', 'r') as file:
        for line in file:
            try:
                list_of_words = line.split(' ')
                first_word = list_of_words[0]
                if first_word == 'Epoch:':
                    seconds_per_train_epoch = float(list_of_words[-1])
                    total_seconds_train += seconds_per_train_epoch
                    train_epoch_count += 1
                elif first_word == 'VALIDATION':
                    seconds_per_val_epoch = float(list_of_words[-1])
                    total_seconds_val += seconds_per_val_epoch
                    val_epoch_count += 1
            except ValueError:
                continue
    total_hours = (total_seconds_train + total_seconds_val) / 3600
    loss_improvement_per_hour = (loss_initial - loss_final) / total_hours
    mIoU_improvement_per_hour = (mIoU_final - mIoU_initial) / total_hours
    print()
    print('TRAIN')
    print('total hours: ' + str(total_seconds_train / 3600))
    print('avg. seconds: ' + str(total_seconds_train / train_epoch_count))
    print()
    print('VAL')
    print('total hours: ' + str(total_seconds_val / 3600))
    print('avg. seconds: ' + str(total_seconds_val / val_epoch_count))
    print()
    print('TOTAL hours: ' + str(total_hours))
    print()
    print('loss improvement per hour: ' + str(loss_improvement_per_hour))
    print('mIoU improvement per hour: ' + str(mIoU_improvement_per_hour))


def tmp():
    color_encoding = OrderedDict([
        ('capacitor', (0, 0, 255)),  # 0
        ('capacitor-flat', (255, 0, 0)),  # 1
        ('resistor', (255, 255, 0)),  # 2
        ('background', (170, 170, 170)),  # 3
    ])

    colors = list(color_encoding.values())
    print(colors)


def rgb_to_gray(rgb_path, gray_path):
    image = cv2.imread(rgb_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(gray_path, gray)


if __name__ == '__main__':
    main()
