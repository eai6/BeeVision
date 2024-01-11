import pandas as pd
import cv2
import ast
import argparse
import numpy as np
import matplotlib.pyplot as plt


def extracDetections(data):
    detections = []
    for frame in data["detections_coordinates"].tolist():
        frame = ast.literal_eval(frame)
        for delection in frame:
            detections.append(delection)

    return detections

# update matrix according to detections
def updateMatrix(matrix, x_start, y_start, width, height):
    for x in range(int(x_start-(width/2)), int(x_start + (width/2))):
        for y in range(int(y_start-(height/2)), int(y_start + (height/2))):
            matrix[y][x] += 1


def filterMatrix(matrix, threshold):
    '''
    Input:
    matrix: matrix to be filtered
    threshold: threshold for filtering
    
    Output:
    filtered matrix
    '''
    matrix_copy = matrix.copy()
    image_shape = matrix.shape
    for x in range(image_shape[0]):
        for y in range(image_shape[1]):
            if matrix[x][y] < threshold:
                matrix_copy[x][y] = 0.0

    return matrix_copy


def getNestCords(matrix):
    # Convert the matrix to a binary image
    ret, binary_image = cv2.threshold(matrix.astype('uint8'), 1, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around the contours
    coordinates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        coordinates.append((x, y, w, h))


    return coordinates


def sortCoordinates(coordinates):
    sorted_coordinates = sorted(coordinates, key=lambda c: (c[1], c[0]))
    return sorted_coordinates

def createVisualisation(image, df):
    df_coordinates = df["coordinates"].tolist()
    df_nest_id = df["nest_id"].tolist()

    # Plot nest detection on image
    for i in range(len(df_coordinates)):
        cords = df_coordinates[i]
        id = df_nest_id[i]
        x, y, w, h = cords
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Save image_rgb as an image file
    cv2.imwrite("output_image.jpg", image_rgb)


def main(data, image):
    data = pd.read_csv(data)
    image = cv2.imread(image)
    image_shape = image.shape[:2]

    # create a matrix for image
    matrix = np.zeros((image_shape[0], image_shape[1]))

    # get all nest detections
    detections = extracDetections(data)


    # update matrix with nest detections
    for detection in detections:
        try:
            updateMatrix(matrix, int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3]))
        except:
            continue

    #print(matrix)

    # filter matrix
    filtered_matrix = filterMatrix(matrix, 50)

    #print(filtered_matrix)

    # get nest coordinates
    nest_coordinates = getNestCords(filtered_matrix)

    #print(nest_coordinates)

    sorted_coordinates = sortCoordinates(nest_coordinates)

    df = pd.DataFrame({"coordinates": sorted_coordinates, "nest_id": [i for i in range(len(sorted_coordinates))]})

    createVisualisation(image, df)

    df.to_csv("nest_output.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video and data to detect nest locations')
    parser.add_argument('--data', default="/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/nest_results.csv",type=str, help='data file', required=False)
    parser.add_argument('--image', default="/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/outputs/nest_frames/2023-05-29_14_20_01/frame_1.jpg",type=str, help='image file', required=False)
    args = parser.parse_args()

    main(args.data, args.image)


