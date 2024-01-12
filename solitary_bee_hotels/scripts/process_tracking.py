from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import ast
import argparse

FRAME_RATE = 30 # frames per second
SILENT_FRAME = '(0, 0, 0, 0)'

def getActivityRanges(lst, element):
    ''' 
    Input:
    lst: the list of motion coordinate detections on all frames
    element: seperator between activity detections (0,0,0,0)

    Output:
    list with ranges of frame indexes of all detections
    '''
    occurrence_ranges = []
    start_index = None

    for i, item in enumerate(lst):
        if item != element:
            if start_index is None:
                start_index = i
        else:
            if start_index is not None:
                end_index = i - 1
                occurrence_ranges.append((start_index, end_index))
                start_index = None

    if start_index is not None:
        # If the last element in the list is the target element
        end_index = len(lst) - 1
        occurrence_ranges.append((start_index, end_index))

    return occurrence_ranges


def processActivity(data, activity_range):

    '''
    Inputs: 
    data: csv output from video processing
    activity_range: indexes ranges of an activity period 

    Output:
    {
        "number_of_objects": 1, # number of objects being tracked or trigger the motion
        "classes": [class_1, ], 
        "trajectories": [[(x_start,y_start), (x_end,y_end)], ],
        "timestamp" : [frame_start, frame_end]
    }
    '''
    # extract activity information
    average_motion_coords = data["motions_coordinates"].tolist()[activity_range[0]: activity_range[1]]
    average_dectections_coords = data["detections_coordinates"].tolist()[activity_range[0]: activity_range[1]]
    detections_classes = data["detections_classes"].tolist()[activity_range[0]: activity_range[1]]
    all_motion_cords = data["frame_motions"].tolist()[activity_range[0]: activity_range[1]]
    all_detections_cords = data["frame_detections"].tolist()[activity_range[0]: activity_range[1]]
    
    
    def getClusterNumber(data:list) -> int:
        '''
        Input: List of [(x,y), (x,y)]

        output: number of clusters in list based on silhoutte scores
        '''
        #print(data)
        #print(len(data))

        if len(data) < 2:
            return 1

        #Calculate silhouette scores for different numbers of clusters
        max_clusters = len(data) - 1
        if max_clusters > 5: # No more than 5 clusters
            max_clusters = 5

        si_scores = {
            1: 0.1
        } # save scores
        # iterate over different k number and get scores
        for num_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            labels = kmeans.fit_predict(data)
            #print(labels)
            silhouette_avg = silhouette_score(data, labels)
            si_scores[num_clusters] = silhouette_avg


        max_key = max(si_scores, key=si_scores.get)


        return max_key

    def extractMiddlePoints(coordinates):
        xywh_cords = ast.literal_eval(coordinates)
        xy_cords = [(cord[0],cord[1]) for cord in xywh_cords]
        return xy_cords
    
    def getDetectedClasses(detections_classes):
        ''' 
        Input:
        detections_classes: List of all detected classes for every frame

        Output:
        Counter object of detected classes. 
        {
            "detected_class": int, - the most occuring class
            "classes": counter object of all detected classes
        }
        '''
        detected_classes_list = []
        detections_classes = [ast.literal_eval(frame) for frame in detections_classes]
        for frame in detections_classes:
            for detection in frame:
                detected_classes_list.append(detection)


        counter = Counter(detected_classes_list)
        try:
            most_occuring_class = counter.most_common(1)[0][0]
        except: # incase there was no object detection for the activity
            most_occuring_class = ""

        return {
            "detected_class": most_occuring_class,
            "classes": counter
        }

    

    df1 = pd.DataFrame({
    "motions": all_motion_cords,
    "detections": all_detections_cords, 
    "classes": detections_classes
    })

    # extrac xy cords for clustering algorithm
    #df1["motions_xy"] = df1["motions"].apply(extractMiddlePoints)
    df1["detections_xy"] = df1["detections"].apply(extractMiddlePoints)

    # apply clustering to detect best cluster k
    #df1["motions_k"] = df1["motions_xy"].apply(getClusterNumber)
    df1["detections_k"] = df1["detections_xy"].apply(getClusterNumber)

    k = df1.groupby("detections_k")['motions'].apply(lambda x: x.mode().iloc[0]).reset_index()["detections_k"].tolist()[0]

    if int(k) == 1: # there is only one object
        # get classes
        classes = getDetectedClasses(df1["classes"].tolist())
        classes = classes["detected_class"]

        return {
            "objects_num": k,
            "classes": classes,
            "trajectory": [average_motion_coords[0], average_motion_coords[-1]],
            "timestamp" : [activity_range[0]/FRAME_RATE, activity_range[-1]/FRAME_RATE]
        }
    else: # multiple objects --- need incorporate this component
        return "Multiple objects"


## process video data
def processVideo(data):

    ## get activity periods ##
    activity_periods = getActivityRanges(data["motions_coordinates"].tolist(), SILENT_FRAME)

    # filter activity_periods and remove noises
    activity_periods = [activity_period for activity_period in activity_periods if activity_period[1]- activity_period[0] > 1 ]

    # process activities to extract the activity type
    activity_types = []
    for activity_period in activity_periods:
        activity_types.append(processActivity(data, activity_period))

    # return as a dataframe
    return pd.DataFrame(activity_types), activity_periods

def main(file, output):
    data = pd.read_csv(file)
    print("Processing Video data")
    results, periods = processVideo(data)
    results["activity_period_id"] = results.index
    results["frames"] = periods
    results.to_csv(output, index=False)

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/results.csv", required=False, help="path to processed traking csv file")
    parser.add_argument("--output", default="/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/motion_output.csv", required=False, help="path to output csv file")   
    args = parser.parse_args()
    main(args.file, args.output)