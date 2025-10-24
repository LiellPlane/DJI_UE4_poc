
import os
import json
import cv2
import time
import random
import copy
import queue
import re
import shutil


current_file_path = os.path.abspath(__file__)
# Get the directory containing the current file
directory_of_current_file = os.path.dirname(current_file_path)

PROCESSED = "processed"
IMAGES_FOLDER = r"D:\phone_to_sort"
SORTED_IMAGES_FOLDER = r"D:\temp_phone_images_sorted"
OPERATION_JSON = f"{directory_of_current_file}\\all_images.json"
PROCESSED_JSON = f"{directory_of_current_file}\\{PROCESSED}.json"
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

def json_in_folder(directory):
    allFiles = []
    for root, _, files in os.walk(directory):
        for name in files:
            if name[-4:len(name)].lower() == 'json':
                if PROCESSED in name:
                    allFiles.append(os.path.join(root, name))
    return allFiles

def DeleteFiles_RecreateFolder(FolderPath):
    Deltree(FolderPath)
    os.mkdir(FolderPath)


def Deltree(Folderpath):
      # check if folder exists
    if len(Folderpath)<6:
        raise("Input:" + str(Folderpath),"too short - danger")
        raise ValueError("Deltree error - path too short warning might be root!")
        return
    if os.path.exists(Folderpath):
         # remove if exists
         shutil.rmtree(Folderpath)
    else:
         # throw your exception to handle this special scenario
         #raise Exception("Unknown Error trying to Deltree: " + Folderpath)
         pass
    return

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
    # if not os.path.exists(PROCESSED_JSON):
    #     print("creating processed file")
    #     with open(PROCESSED_JSON, 'w') as file:
    #         json.dump({}, file, indent=4)
    print("opening existing json file")
    with open(OPERATION_JSON, 'r') as file:
        all_images_list = json.load(file)


    # with open(PROCESSED_JSON, 'r') as file:
    #     processed_list = json.load(file)
    # check that the local images match the file
    assert len(set(jpgs).symmetric_difference(set(all_images_list))) == 0
    processed_list = {}
    undo_fifo = queue.LifoQueue()



    list_batch_jsons  = {x: None for x in json_in_folder(directory_of_current_file)}
    print("total img len", len(all_images_list))
    all_processed_files = []
    for batch_file in list_batch_jsons:
        print(batch_file)
        with open(batch_file, 'r') as file:
            proceseed_imgs = json.load(file)
            for img in proceseed_imgs.keys():
                all_processed_files.append(img)
                del all_images_list[img]
    print("total img len after removing processed", len(all_images_list))
    assert len(set(all_processed_files)) == len(all_processed_files)
    pattern = re.compile(r'processed_(\d+)\.json')
    max_number = -1
    file_with_max_number = None
    
    for filename in list_batch_jsons:
        basename = os.path.basename(filename)
        match = pattern.match(basename)
        if match:
            # Extract the numerical part
            number = int(match.group(1))
            
            # Check if this number is the largest we've seen so far
            if number > max_number:
                max_number = number
                file_with_max_number = filename

    batch_id = max_number + 1
    while len(all_images_list) > 0:
        
        newfile = random.choice(list(all_images_list.keys()))
        print(newfile)
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
            del all_images_list[newfile]
            undo_fifo.put(newfile)

        if len(processed_list) > 30 or len(all_images_list) < 1:
            # save batched file process
            batch_file = PROCESSED_JSON.replace(PROCESSED, f"{PROCESSED}_{batch_id}")
            with open(batch_file, 'w') as file:
                json.dump(processed_list, file, indent=4)
            batch_id += 1
            processed_list = {}

        print(batch_id)
        print("total to_be_processed len", len(all_images_list))
        print("total processed_list len", len(processed_list))

    fart = input("press y (in termimal) to distribute into folders")
    if fart == "y":
        fart = input(f"type {SORTED_IMAGES_FOLDER} to continue")
        if fart != SORTED_IMAGES_FOLDER:
            raise Exception("process terminated by user")
        DeleteFiles_RecreateFolder(SORTED_IMAGES_FOLDER)
        #os.mkdir(f"{SORTED_IMAGES_FOLDER}\\BACKUP")
        list_batch_jsons  = {x: None for x in json_in_folder(directory_of_current_file)}
        all_processed_files = {}
        for batch_file in list_batch_jsons:
            with open(batch_file, 'r') as file:
                proceseed_imgs = json.load(file)
                for img, folder in proceseed_imgs.items():
                    all_processed_files[img] = folder

        for folder in list(set(list(all_processed_files.values()))):
            os.mkdir(f"{SORTED_IMAGES_FOLDER}\\{folder}")
        for img_file, folder in all_processed_files.items():
            print("copying", img_file, "to", folder)
            destination_directory = f"{SORTED_IMAGES_FOLDER}\\{folder}"
            shutil.move(img_file, destination_directory)

if __name__ == '__main__':
    main()
