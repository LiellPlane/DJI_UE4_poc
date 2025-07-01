import cv2
import numpy as np
import os
import glob

script_dir = os.path.dirname(os.path.abspath(__file__))
jpg_files = glob.glob(os.path.join(script_dir, "*.jpg"))
print(f"Found {len(jpg_files)} jpg files: {jpg_files}")

for i, jpg_file in enumerate(jpg_files):
    print(f"\nProcessing {jpg_file} ({i+1}/{len(jpg_files)})")
    
    # Just open the image with OpenCV
    yuv420 = cv2.imread(jpg_file)
    GRAY = cv2.cvtColor(yuv420, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(GRAY, cv2.COLOR_YUV2RGB_I420)
    
    print(f"RGB shape: {rgb.shape}")
    print(f"RGB min/max: {rgb.min()}/{rgb.max()}")
    
    cv2.imshow(f"YUV420 to RGB - {jpg_file}", cv2.resize(rgb, (500,500)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Done processing all files")
