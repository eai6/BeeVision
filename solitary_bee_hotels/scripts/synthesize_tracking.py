import pandas as pd
import ast
import argparse


def calculate_overlap(box1, box2):
    # Extract coordinates and dimensions from the boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate the coordinates of the overlapping region
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    
    # Calculate the area of the overlapping region
    overlap_area = x_overlap * y_overlap
    
    # Calculate the area of each bounding box
    area_box1 = w1 * h1
    area_box2 = w2 * h2
    
    # Calculate the total area covered by both boxes (union)
    total_area = area_box1 + area_box2 - overlap_area
    
    # Calculate the overlap ratio
    overlap_ratio = overlap_area / total_area
    
    return overlap_ratio




def trajectoryNest(trajectory):
    '''
    Input: 
    trajectory: [(x_start, y_start, w, h), (x_end, y_end, w, h)]

    Output:
    nest_id: nest id of the nest that the trajectory overlaps with
    '''
    trajectory = ast.literal_eval(trajectory)
    start_position = ast.literal_eval(trajectory[0])
    end_position = ast.literal_eval(trajectory[1])

    # get start and end nest if trajectory overlaps with nest
    start_nest = {}
    end_nest = {}

    # start position
    for i in range(len(nest_coordinates)):
        nest = ast.literal_eval(nest_coordinates[i])
        nest_id = nest_ids[i]
        overlap_ratio = calculate_overlap(start_position, nest)
        if overlap_ratio > 0.01:
            start_nest[nest_id] = overlap_ratio
        

    # end position
    for i in range(len(nest_coordinates)):
        nest = ast.literal_eval(nest_coordinates[i])
        nest_id = nest_ids[i]
        overlap_ratio = calculate_overlap(end_position, nest)
        if overlap_ratio > 0.01:
            end_nest[nest_id] = overlap_ratio

    # get nest id with highest overlap ratio
    if len(start_nest) == 0:
        start_nest_id = ""
    else:
        start_nest_id = max(start_nest, key=start_nest.get)

    if len(end_nest) == 0:
        end_nest_id = ""
    else:
        end_nest_id = max(end_nest, key=end_nest.get)

    return [start_nest_id, end_nest_id]


dict = {
    0: "carpenter_bee",
    1: "closed_nest",
    2: "leafcutting_bee",
    3: "open_nest",
    4: "wasp"
}
def decodeClass(id):
    try:
        return dict[id]
    except:
        return "Unknown"

nest_coordinates = []
nest_ids = []


def getAction(nest_ids):
    start_nest_id = nest_ids[0]
    end_nest_id = nest_ids[1]

    if start_nest_id == end_nest_id:
        return "In Nest"
    elif start_nest_id == "":
        return "Entering Nest"
    elif end_nest_id == "":
        return "Leaving Nest"
    elif start_nest_id != end_nest_id and start_nest_id != "" and end_nest_id != "":
        return "Moving Between Nests"
    else:
        return "Unknown"

def main(tracking, nests, output):
    global nest_coordinates, nest_ids
    
    # load data
    data = pd.read_csv(tracking)
    nest_data = pd.read_csv(nests)

    # get nest coordinates
    nest_coordinates = nest_data["coordinates"].tolist()
    nest_ids = nest_data["nest_id"].tolist()

    # get nest ids for each trajectory
    data["nest_ids"] = data["trajectory"].apply(trajectoryNest)

    # get classes
    data["species"] = data["classes"].apply(decodeClass)

    # get action
    data["action"] = data["nest_ids"].apply(getAction)

    data["activity_period_id"] = data.index

    # save dataframe
    #data[["trajectory", "timestamp", "nest_ids", "class", "action"]].to_csv(output, index=True)
    data.to_csv(output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process tracking data')
    parser.add_argument('--tracking', default="/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/motion_output.csv", help='tracking data file')
    parser.add_argument('--nests', default="/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/nest_output.csv", help='nests data file')
    parser.add_argument('--output', default="/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/final_output.csv", help='output file')
    args = parser.parse_args()

    main(args.tracking, args.nests, args.output)