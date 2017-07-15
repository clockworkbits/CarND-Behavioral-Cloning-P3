import csv
import cv2

lines = []
with open("data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurments = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG' + filename
    image = cv2.imread(current_path)
    images.append(image)

    # Add the measurement for the given image
    measurments.append(float(line[3]))

X_train = np.array(images)
y_train = np.array(measurments)