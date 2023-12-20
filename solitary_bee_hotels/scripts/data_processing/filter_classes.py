import os
import argparse

def filterFiles(classes, folder):
    # open the txt file to read
    files = []
    for file_path in os.listdir(folder):
        #print(file)
        file = open(os.path.join(folder, file_path), 'r')
        lines = file.readlines()
        #print(li)
        for line in lines:
            #print(line)
            label = line.split()[0]
            #print(type(label))
            if int(label) in classes:
                files.append(file_path)
                break
    return files

        
def removeFiles(files, folder):
    for file in os.listdir(folder):
        if file not in files:
            os.remove(os.path.join(folder, file))
            print(f"Remvoed {file}")

def main(dataset, classes):
    print("Filtering classes")
    train_files = filterFiles(classes, os.path.join(dataset, "train/labels"))
    test_files = filterFiles(classes, os.path.join(dataset, "test/labels"))
    valid_files = filterFiles(classes, os.path.join(dataset, "valid/labels"))

    # remove labels
    print("Removing labels")
    removeFiles(train_files, os.path.join(dataset, "train/labels"))
    removeFiles(test_files, os.path.join(dataset, "test/labels"))
    removeFiles(valid_files, os.path.join(dataset, "valid/labels"))
    
    # get images
    print("Getting images")
    train_files = [file.split(".txt")[0]+".jpg" for file in train_files]
    test_files = [file.split(".txt")[0]+".jpg" for file in test_files]
    valid_files = [file.split(".txt")[0]+".jpg" for file in valid_files]

    # remove images
    print("Removing images")
    removeFiles(train_files, os.path.join(dataset, "train/images"))
    removeFiles(test_files, os.path.join(dataset, "test/images"))
    removeFiles(valid_files, os.path.join(dataset, "valid/images"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter classes from dataset")
    parser.add_argument("--dataset", type=str, help="Path to dataset folder")
    parser.add_argument("--classes", type=str, help="Classes to keep")
    args = parser.parse_args()
    classes = [int(c) for c in args.classes.split(",")]
    main(args.dataset, classes)