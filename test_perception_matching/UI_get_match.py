import tkinter as tk
from tkinter import filedialog, Button, Label, Frame
import cv2
import numpy as np
import os
import get_sequence_images_qdrant
import generate_embeddings
import json
client = get_sequence_images_qdrant.get_qdrant_client()
vector, random_item, closest_matches, payload = get_sequence_images_qdrant.get_random_item_with_closest_match(
    client,
    collection_name="everything",
    limit=1
    )
# ensure using same embeddings by grabbing it straight from qdrant collection
GRABBED_EMBEDDING_PARAMS = generate_embeddings.ImageEmbeddingParams(**json.loads(payload["params"]))

class ImageProcessorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.root.geometry("800x600")
        
        # Create main frame
        self.main_frame = Frame(root, bg="#f0f0f0")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Instructions for users
        self.instruction_label = Label(
            self.main_frame, 
            text="Drag an image from your file explorer and drop it onto the button below",
            bg="#f0f0f0",
            font=("Arial", 12)
        )
        self.instruction_label.pack(pady=10)
        
        # Button to browse for image (alternative to drag-drop)
        self.browse_button = Button(
            self.main_frame, 
            text="Select Image", 
            command=self.browse_image,
            height=6,
            width=30,
            bg="#e0e0e0",
            font=("Arial", 12)
        )
        self.browse_button.pack(pady=20)
        
        # Label to show selected image path
        self.path_label = Label(self.main_frame, text="No image selected", bg="#f0f0f0")
        self.path_label.pack(pady=5)
        
        # Result section
        self.result_label = Label(self.main_frame, text="Results will appear here", bg="#f0f0f0")
        self.result_label.pack(pady=20)
        
        # Configure file drop event
        self.root.drop_target_register(self.root)
        self.root.dnd_bind('<<Drop>>', self.on_drop)
        
        # Store the current image path
        self.current_image_path = None
        
    def browse_image(self):
        """Open file dialog to select an image"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.process_image_file(file_path)
    
    def on_drop(self, event):
        """Handle file drop event"""
        file_path = event.data
        
        # Clean up the path based on platform
        import sys
        if sys.platform == 'win32':
            file_path = file_path.replace('{', '').replace('}', '')
        else:
            file_path = file_path.strip('{}').replace('\\', '')
        
        self.process_image_file(file_path)
    
    def process_image_file(self, file_path):
        """Process the selected image file"""
        # Check if it's an image file
        if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            self.path_label.config(text="Not an image file")
            return
            
        self.current_image_path = file_path
        filename = os.path.basename(file_path)
        self.path_label.config(text=f"Selected: {filename}")
        
        # Display the image using OpenCV
        self.display_input_image(file_path)
        
        # Process the image (placeholder)
        self.process_image()
    
    def display_input_image(self, image_path):
        """Display the input image using OpenCV"""
        try:
            img = cv2.imread(image_path)
            cv2.imshow("Input Image", img)
            cv2.waitKey(1)  # Non-blocking wait
        except Exception as e:
            print(f"Error displaying image: {e}")
    
    def process_image(self):
        """Process the current image and display results"""
        if not self.current_image_path:
            return
            
        # PLACEHOLDER - Replace with your actual image processing code
        # Example: result = your_processing_function(self.current_image_path)
        
        # Simulate processing
        self.result_label.config(text="Prowcessing...")
        self.root.update()
        # For demonstration, just display the image in a new window
        img = cv2.imread(self.current_image_path)
        embedding = generate_embeddings.create_image_embedding(
            img, 
            params=GRABBED_EMBEDDING_PARAMS,
            mask=None
        )
                
        self.result_label.config(text=f"length of embedding: {len(embedding)}")
        self.root.update()
        
        # Add a simple effect to simulate processing
        # (Replace this with your actual processing)
        search_result = client.search(
            collection_name="everything",
            query_vector=embedding,
            limit=1,
            with_payload=True,
            with_vectors=True
        )
        # Display result
        cv2.imshow("Processing Result", processed_img)
        cv2.waitKey(1)  # Non-blocking wait
        
        # Update the result information
        filename = os.path.basename(self.current_image_path)
        filepath = os.path.dirname(self.current_image_path)
        self.result_label.config(text=f"Processed: {filename}\nFrom: {filepath}")

# Define a minimal drop target register function for Tkinter
def dnd_bind(widget, event, callback):
    """Bind a drop event to a callback"""
    widget.bind(event, callback)

def drop_target_register(widget, *args):
    """Register a widget as a drop target"""
    try:
        widget.tk.call('package', 'require', 'tkdnd')
    except:
        # TkDND is not available, use a simpler approach
        print("TkDND not available, drag-and-drop may not work")
        print("Use the 'Select Image' button instead")
    
    try:
        widget.tk.eval('package require tkdnd')
        widget.tk.call('tkdnd::drop_target', 'register', widget, 'DND_Files')
    except:
        pass  # Silently continue if tkdnd is not available
        
# Patch Tkinter to add the drag and drop functions
tk.Tk.drop_target_register = drop_target_register
tk.Tk.dnd_bind = dnd_bind

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorUI(root)
    root.mainloop()
