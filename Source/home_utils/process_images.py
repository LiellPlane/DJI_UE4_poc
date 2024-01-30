
import os
import json

IMAGES_FOLDER = "/Users/liell_p/GIT/DJI_UE4_poc"
OPERATION_JSON = "/Users/liell_p/GIT/DJI_UE4_poc/Source/home_utils/all_images.json"
PROCESSED_JSON = "/Users/liell_p/GIT/DJI_UE4_poc/Source/home_utils/processed.json"


def jpgs_in_folder(directory):
    allFiles = []
    for root, _, files in os.walk(directory):
        for name in files:
            if name[-4:len(name)].lower() == '.jpg':
                allFiles.append(os.path.join(root, name))
    return allFiles


def main():
    # get all images in a folder
    jpgs = jpgs_in_folder(IMAGES_FOLDER)
    # create the json if it doesn't exist
    print(f"{len(jpgs)} images (jpgs) found")
    all_images_list = None
    if not os.path.exists(OPERATION_JSON):
        print("creating json file")
        with open(OPERATION_JSON, 'w') as file:
            json.dump(jpgs, file, indent=4)
    if not os.path.exists(PROCESSED_JSON):
        print("creating processed file")
        with open(PROCESSED_JSON, 'w') as file:
            json.dump({}, file, indent=4)
    print("opening existing json file")
    with open(OPERATION_JSON, 'r') as file:
        all_images_list = json.load(file)
    print("opening existing processed file")
    with open(PROCESSED_JSON, 'r') as file:
        processed_list = json.load(file)
    # check that the local images match the file
    assert len(set(jpgs).symmetric_difference(set(all_images_list))) == 0


if __name__ == '__main__':
    main()
