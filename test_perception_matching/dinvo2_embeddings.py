import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import cv2

# Initialize model and device globally on module load
print("Initializing DINOv2 model...")
print(f"CUDA available: {torch.cuda.is_available()}")

dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
dinov2_vits14.to(device)
dinov2_vits14.eval()

transform_image = T.Compose([
    T.ToTensor(), 
    T.Resize(244), 
    T.CenterCrop(224), 
    T.Normalize([0.5], [0.5])
])

print(f"DINOv2 model loaded on {device}")


def create_image_embedding(image: np.ndarray) -> np.ndarray:
    """
    Create a DINOv2 embedding from a CV2/numpy image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image as numpy array (BGR format from CV2)
        
    Returns:
    --------
    numpy.ndarray
        Feature vector (embedding) as 1D numpy array
    """
    # Convert BGR (CV2) to RGB (PIL)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Transform image
    transformed_img = transform_image(pil_image)[:3].unsqueeze(0)
    
    # Generate embedding
    with torch.no_grad():
        embedding = dinov2_vits14(transformed_img.to(device))
    
    # Convert to numpy and flatten
    embedding_np = embedding[0].cpu().numpy().flatten()
    
    return embedding_np


def main():
    """Test function to generate 100 images and time embedding generation"""
    import time
    import random
    
    print("\n=== Testing DINOv2 Embedding Performance ===\n")
    
    num_images = 100
    embeddings = []
    
    print(f"Generating {num_images} random images and computing embeddings...\n")
    
    start_time = time.time()
    
    for i in range(num_images):
        # Create a 500x500 image with random background color
        bg_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
        img = np.ones((500, 500, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
        
        # Draw a circle with random color and position
        center = (random.randint(150, 350), random.randint(150, 350))
        radius = random.randint(50, 150)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.circle(img, center, radius, color, -1)
        
        # Generate embedding
        embedding = create_image_embedding(img)
        embeddings.append(embedding)
        
        # Progress update every 10 images
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (num_images - (i + 1)) / rate if rate > 0 else 0
            print(f"Progress: {i+1}/{num_images} | Rate: {rate:.2f} img/s | ETA: {eta:.1f}s")
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    print(f"\n=== Results ===")
    print(f"Total images processed: {num_images}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per image: {(total_time/num_images)*1000:.2f} ms")
    print(f"Images per second: {num_images/total_time:.2f}")
    print(f"\nEmbedding shape: {embeddings[0].shape}")
    print(f"Embedding size: {embeddings[0].shape[0]} dimensions")
    
    # Show the last image as example
    print("\nShowing last generated image (press any key to close)...")
    cv2.imshow("Last Test Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return embeddings


if __name__ == "__main__":
    main()