
import os
import json
import cv2
import time
import random
import copy
import queue
current_file_path = os.path.abspath(__file__)
# Get the directory containing the current file
directory_of_current_file = os.path.dirname(current_file_path)


IMAGES_FOLDER = r"D:\temp_phoneimgs"
OPERATION_JSON = f"{directory_of_current_file}\\all_images.json"
PROCESSED_JSON = f"{directory_of_current_file}\\processed.json"
IMAGETYPES = [
    "haha",
    "art",
    "family",
    "liell",
    "xtra_sort",
    "delete",
    "undo"
]

def viewer(
        inputimage,
        move_windowx,
        move_windowy,
        pausetime_Secs=0,
        presskey=False,
        destroyWindow=True):
    try:
        cv2.imshow("img", inputimage)
        cv2.moveWindow("img", move_windowx, move_windowy)

        while True:
            presssedkey = cv2.waitKey(0)
            for _key in IMAGETYPES:
                if presssedkey == ord(_key[0]):
                    return _key
            print("please press first letter of function", IMAGETYPES)
        if presskey==False:
            if cv2.waitKey(20) & 0xFF == 27:
                    pass
        if pausetime_Secs>0:
            time.sleep(pausetime_Secs)
        if destroyWindow==True: cv2.destroyAllWindows()

    except Exception as e:
        print(e)


def jpgs_in_folder(directory):
    allFiles = []
    for root, _, files in os.walk(directory):
        for name in files:
            if name[-4:len(name)].lower() == '.jpg':
                allFiles.append(os.path.join(root, name))
    return allFiles


def main():
    # get all images in a folder
    jpgs = {x: None for x in jpgs_in_folder(IMAGES_FOLDER)}
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

    undo_fifo = queue.LifoQueue()
    while len(all_images_list) > 0:
        newfile = random.choice(all_images_list)
        img = cv2.imread(newfile)
        ratio = 1000/img.shape[0]
        img = cv2.resize(img, (int(img.shape[1]*ratio),int(img.shape[0]*ratio)), interpolation = cv2.INTER_AREA)

        cmd = viewer(
            img,
            None,
            None,
            0,
            True,
            False
        )
        if cmd == "undo":
            if not undo_fifo.empty():
              del processed_list[undo_fifo.get()]
        else:
            processed_list[newfile] = cmd
            undo_fifo.put(newfile)
            

        print(processed_list)
if __name__ == '__main__':
    main()
