import generate_embeddings
import dinvo2_embeddings
import qdrant_utils
import get_sequence_images_qdrant
import json
import cv2
import numpy as np

def get_image_embedding(image_path: str, img_in_memory:np.ndarray = None):
    """
    TODO lazy implementation of dino v2 - refactor this to make it all nicer
    """

    if img_in_memory is None:
        img = cv2.imread(image_path)
    else:
        img = img_in_memory

    # TODO dino probably doesnt need this flipping
    img_flipped = cv2.flip(img, 1)
    embedding = dinvo2_embeddings.create_image_embedding(
        img, 
    )
    embedding_flipped = dinvo2_embeddings.create_image_embedding(
        img_flipped
    )
    return embedding, embedding_flipped
