import cv2
import json
import sys
import os

# Get the absolute path of the directory containing the module
module_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the module's parent directory to sys.path
sys.path.insert(0, module_parent_dir)
import lumotag
import numpy as np
print(np.__version__)
# List to store click positions
import pickle
def overlay_warped_image(background, warped):
    # Ensure the images have the same size and are mono
    assert background.shape == warped.shape, "Images must have the same dimensions"
    assert len(background.shape) == 2 and len(warped.shape) == 2, "Images must be mono (single channel)"
    
    # Ensure images are 8-bit unsigned integer type
    background = background.astype(np.uint8)
    warped = warped.astype(np.uint8)

    # Create a mask based on non-black pixels in the warped image
    _, mask = cv2.threshold(warped, 1, 255, cv2.THRESH_BINARY)

    # Black-out the area of warped image in background
    background_masked = cv2.bitwise_and(background, cv2.bitwise_not(mask))

    # Combine the background and warped image
    result = cv2.add(background_masked, warped)

    return result

def compute_and_apply_transform(src_img, dst_img, src_points, dst_points):
    # Convert points to numpy arrays
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective transformation to the source image
    height, width = dst_img.shape[:2]
    result = cv2.warpPerspective(src_img, matrix, (width, height))

    return result, matrix

def apply_transform(matrix, src_img, dst_img):

    # Apply the perspective transformation to the source image
    height, width = dst_img.shape[:2]
    result = cv2.warpPerspective(src_img, matrix, (width, height))

    return result


current_dir = os.path.dirname(os.path.abspath(__file__))
# Mouse callback function
class ClickRecorder:
    def __init__(self):
        self.click_positions = []

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_positions.append((x, y))
            print(f"Click recorded at position: ({x}, {y})")



def show_image_until_keypress(image, window_name='Image'): 
    cv2.imshow(window_name, image)
    cv2.moveWindow(window_name, 100, 100)   
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key != 255:  # Any key press
            break
    
    #cv2.destroyWindow(window_name)
# load long range then close range
images = [
    
    r"C:\Working\GIT\DJI_UE4_poc\Source\lumotag\typescript-server-app\packages\server\uploads\LID19028hXtsvLID.jpg",
    r"C:\Working\GIT\DJI_UE4_poc\Source\lumotag\typescript-server-app\packages\server\uploads\LID19048hXtsvLID.jpg"
    ]

images = [cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY) for image in images]

clicked_positions = []
for _image in images:
    click_positions = []
    image = _image

    if image is None:
        print(f"Error: Could not read the image at {image_path}")
        exit()

    # Create a window and set the callback function
    cv2.namedWindow('Image')
    recorder = ClickRecorder()
    cv2.setMouseCallback('Image', recorder.mouse_callback)

    # Display the image and wait for clicks
    while len(recorder.click_positions) < 4:
        cv2.imshow('Image', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    clicked_positions.append(recorder.click_positions)
    cv2.destroyAllWindows()


warped_img, warpmatrix = compute_and_apply_transform(
    images[0],
    images[1],
    clicked_positions[0],
    clicked_positions[1]
    )
# show_image_until_keypress(warped_img,"warped_img")
script_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(script_path))
pickle_file_path = os.path.join(parent_dir, lumotag.get_perspectivewarp_filename())

perp_dic = {}
perp_dic[lumotag.get_perspectivewarp_dictkey()] = warpmatrix
perp_dic["longrange_positions"] = clicked_positions[0]
perp_dic["closerange_positions"] = clicked_positions[1]
perp_dic["info"] = "longrange position clicked in image, and associated closerange positions to create a warp from longrange into closerange"

with open(pickle_file_path, 'wb') as f:
    pickle.dump(perp_dic, f)
# Load the list from the JSON file
with open(pickle_file_path, 'rb') as f:
    warp_matrix = pickle.load(f)["warpmatrix"] 

wraped_img = apply_transform(warp_matrix,images[0],images[1])
show_image_until_keypress(wraped_img,"wraped_img")
combo_image = overlay_warped_image(images[1], wraped_img)
while True:
    show_image_until_keypress(images[1],"combo_image")
    show_image_until_keypress(combo_image,"combo_image")