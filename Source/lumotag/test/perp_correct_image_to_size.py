import cv2
import numpy as np
import os

class PerspectiveCorrector:
    def __init__(self, image_path, target_width=616, target_height=1000):
        self.image_path = image_path
        self.target_width = target_width
        self.target_height = target_height
        self.corners = []
        self.image = None
        self.display_image = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to select corners"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.corners) < 4:
                self.corners.append((x, y))
                print(f"Corner {len(self.corners)}: ({x}, {y})")
                
                # Draw the point on the image
                cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(self.display_image, f"{len(self.corners)}", 
                           (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Select 4 corners (click in order: top-left, top-right, bottom-right, bottom-left)', self.display_image)
                
                # If we have 4 corners, perform perspective correction
                if len(self.corners) == 4:
                    self.perform_perspective_correction()
    
    def load_image(self):
        """Load the image and create a copy for display"""
        if not os.path.exists(self.image_path):
            print(f"Error: Image file '{self.image_path}' not found!")
            return False
            
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            print(f"Error: Could not load image '{self.image_path}'!")
            return False
            
        # Scale the image for better visibility (make it much bigger)
        height, width = self.image.shape[:2]
        # Make the image much larger - scale to at least 1200px on the longer side
        scale_factor = max(1200 / width, 1200 / height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        self.scaled_image = cv2.resize(self.image, (new_width, new_height))
        self.scale_factor = scale_factor
        
        # Create a copy for display (we'll draw on this)
        self.display_image = self.scaled_image.copy()
        return True
    
    def perform_perspective_correction(self):
        """Apply perspective transformation to the selected corners"""
        print("Performing perspective correction...")
        
        # Convert scaled corners back to original image coordinates
        original_corners = []
        for corner in self.corners:
            original_x = int(corner[0] / self.scale_factor)
            original_y = int(corner[1] / self.scale_factor)
            original_corners.append((original_x, original_y))
        
        # Define the destination points (rectangle)
        dst_points = np.array([
            [0, 0],                           # top-left
            [self.target_width - 1, 0],       # top-right
            [self.target_width - 1, self.target_height - 1],  # bottom-right
            [0, self.target_height - 1]       # bottom-left
        ], dtype=np.float32)
        
        # Convert corners to numpy array
        src_points = np.array(original_corners, dtype=np.float32)
        
        # Calculate the perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply the transformation
        corrected_image = cv2.warpPerspective(self.image, matrix, 
                                            (self.target_width, self.target_height))
        
        # Save the corrected image
        self.save_corrected_image(corrected_image)
        
        # Show the result
        cv2.imshow('Perspective Corrected Image', corrected_image)
        print("Perspective correction complete! Press any key to close windows.")
        
    def save_corrected_image(self, corrected_image):
        """Save the corrected image with _perp_corrected suffix"""
        # Get the base filename and extension
        base_name = os.path.splitext(self.image_path)[0]
        extension = os.path.splitext(self.image_path)[1]
        
        # Create the output filename
        output_path = f"{base_name}_perp_corrected{extension}"
        
        # Save the image
        success = cv2.imwrite(output_path, corrected_image)
        if success:
            print(f"Corrected image saved as: {output_path}")
        else:
            print(f"Error: Failed to save corrected image to {output_path}")
    
    def run(self):
        """Main function to run the perspective correction tool"""
        if not self.load_image():
            return
            
        print("Instructions:")
        print("1. Click on the 4 corners of the rectangle you want to correct")
        print("2. Click in this order: top-left, top-right, bottom-right, bottom-left")
        print("3. The corrected image will be saved automatically")
        print("4. Press 'q' to quit or any key after correction to close")
        
        # Create window and set mouse callback
        window_name = 'Select 4 corners (click in order: top-left, top-right, bottom-right, bottom-left)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        # Resize window to show the image at its actual scaled size
        cv2.resizeWindow(window_name, self.scaled_image.shape[1], self.scaled_image.shape[0])
        
        # Display the image
        cv2.imshow(window_name, self.display_image)
        
        # Wait for user input
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC to quit
                break
            if len(self.corners) == 4:
                # Wait for any key after correction
                cv2.waitKey(0)
                break
        
        cv2.destroyAllWindows()

def main():
    # Find the image file using relative discovery
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "voltagereg.jpg")
    
    # Check if file exists, if not try current directory
    if not os.path.exists(image_path):
        image_path = "voltagereg.jpg"
        if not os.path.exists(image_path):
            print(f"Error: Could not find 'voltagereg.jpg' in {script_dir} or current directory")
            return
    
    print(f"Using image: {image_path}")
    
    # Create and run the perspective corrector
    corrector = PerspectiveCorrector(image_path, target_width=616, target_height=1000)
    corrector.run()

if __name__ == "__main__":
    main()
