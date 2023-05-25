import cv2
import numpy as np
import datetime
import math
import os
import pandas as pd


class CreateClockDataset:

    def __init__(self):
        self.colors = {'blue': (0, 0, 0), 'white': (255, 255, 255), 
                       'black': (0, 0, 0),'gray': (125, 125, 125),
                       'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}

        self.hours_dest = np.array(
                        [(620, 320), (580, 470), 
                        (470, 580), (320, 620), 
                        (170, 580), (60, 470), 
                        (20, 320), (60, 170), 
                        (169, 61), (319, 20),
                        (469, 60), (579, 169)])

        self.hours_orig = np.array(
                        [(600, 320), (563, 460), 
                        (460, 562), (320, 600), 
                        (180, 563), (78, 460), 
                        (40, 320), (77, 180), 
                        (179, 78), (319, 40),
                        (459, 77), (562, 179)])

        self.image = np.zeros((640, 640, 3), dtype="uint8")

    def array_to_tuple(self, arr):
        return tuple(arr.reshape(1, -1)[0])


    def DrowClock(self, lines=False):
        self.image[:] = self.colors['white']
        if lines:
            for i in range(0, 12):
                cv2.line(self.image, self.array_to_tuple(self.hours_orig[i]), 
                        self.array_to_tuple(self.hours_dest[i]), self.colors['black'], 3)

        cv2.circle(self.image, (320, 320), 310, self.colors['dark_gray'], 8)
        return self.image


    def CreateImageClock(self):
        image_original = self.image.copy()

        times_labels = []
        for h in range(1, 13):
            for m in range(0, 60):
                times_labels.append([h, m])

        times_create = []
        for h, m in times_labels:
            times_create.append(datetime.time(hour=h, minute=m))
       
        episode = 0
        file_name = {}
        for time_now, target in zip(times_create, times_labels):
            hour = math.fmod(time_now.hour, 12)
            minute = time_now.minute
            second = time_now.second

            print(f"hour:'{hour}' minute:'{minute}' ")

            minute_angle = math.fmod(minute * 6 + 270, 360)
            hour_angle = math.fmod((hour * 30) + (minute / 2) + 270, 360)

            print(f"hour_angle:'{hour_angle}' minute_angle:'{minute_angle}'")

            minute_x = round(320 + 260 * math.cos(minute_angle * 3.14 / 180))
            minute_y = round(320 + 260 * math.sin(minute_angle * 3.14 / 180))
            cv2.line(self.image, (320, 320), (minute_x, minute_y), self.colors['blue'], 8)

            hour_x = round(320 + 220 * math.cos(hour_angle * 3.14 / 180))
            hour_y = round(320 + 220 * math.sin(hour_angle * 3.14 / 180))
            cv2.line(self.image, (320, 320), (hour_x, hour_y), self.colors['blue'], 10)

            cv2.circle(self.image, (320, 320), 10, self.colors['dark_gray'], -1)

            if os.path.exists("created_dataset"):
                cv2.imwrite(f"created_dataset/clock_{episode}.jpg", self.image)
                file_name[f"clock_{episode}.jpg"] = target

            else:
                os.mkdir("created_dataset")
                cv2.imwrite(f"created_dataset/clock_{episode}.jpg", self.image)
                if file_name.empty:
                    file_name['path_file'] = f"clock_{episode}.jpg"
                    file_name['target'] = target
                else:
                    file_name['path_file'].join(f"clock_{episode}.jpg")
                    file_name['target'].join(target)

            self.image = image_original.copy()
            episode += 1

        file_name = pd.Series(file_name)
        file_name.to_csv("labels.csv")
        

Clock = CreateClockDataset()
image = Clock.DrowClock(lines=False)
Clock.CreateImageClock()

