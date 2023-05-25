import pandas as pd
from ClockDetection import FindSimilar

labels = pd.read_csv("labels.csv")
file_path = "test/test_1.jpg"
SHAPES = (220, 420)

finder = FindSimilar(SHAPES, file_path)
accuracy = finder.CompareImages(labels)
maximom = max(accuracy)[0]
for val in accuracy:
    if val[0]==maximom and maximom!=0:
        print(val)

finder.ShowResult(accuracy, labels)

