import numpy as np
import cv2
from typing import List
import os
import imutils

class FindSimilar:

    def __init__(self, shape, path_test):
        self.shape = shape
        self.img_test = cv2.imread(path_test)
        self.img_test = cv2.cvtColor(self.img_test, cv2.COLOR_BGR2GRAY)
        self.img_test = cv2.resize(self.img_test, (self.shape))
        self.img_test = self.FindCircle(self.img_test)

    def CalculateMatches(self, des1:List[cv2.KeyPoint], des2:List[cv2.KeyPoint]):
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)
        result = []
        for m, n in matches:
            if m.distance<0.7 * n.distance:
                result.append([m])

        return result


    def FindCircle(self, img):
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.3, 100)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                radius = r+20
                centerX = x - radius
                centerY = y - radius
                h = 2 * radius
                w = 2 * radius
            img = img[centerY:centerY + h, centerX:centerX + w]
        return img 

    def CompareImages(self, labels):
        accuracy = []
        images_list = []
        for name_file in labels['path'].tolist():
            images_list.append(os.path.join('created_dataset', name_file))

        for image in images_list:
            img1 = cv2.imread(image)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            kaze = cv2.KAZE_create()
            kp1, des1 = kaze.detectAndCompute(img1, None)
            kp2, des2 = kaze.detectAndCompute(self.img_test, None)
            matches = self.CalculateMatches(des1, des2)
            try:
                score = 100 * (len(matches) / min(len(kp1), len(kp2)))
            except ZeroDivisionError:
                score = 0
            accuracy.append([score, image])
            
        return accuracy
    

    def ShowResult(self, accuracy, labels):
        maximom = max(accuracy)[0]
        high_accuracy = []
        for val in accuracy:
            if val[0]==maximom and maximom!=0:
                high_accuracy.append(val[1].split('/')[1])

        for index in range(len(labels)-1):
            if labels['path'].iloc[index]==high_accuracy:
                print(f"We find the prediction time nears {labels['target'].iloc[index][1]}:{labels['target'].iloc[index][-2]}")

        img_accuracy = cv2.imread(f"created_dataset/{high_accuracy[-1]}")
        cv2.imshow("Predicted", img_accuracy)
        cv2.imshow("Real", self.img_test)
        cv2.waitKey(0)

