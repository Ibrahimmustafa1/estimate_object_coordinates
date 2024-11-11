import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ast
import pandas as pd
from PIL import Image
import json

def draw_bounding_boxes(image, objects):
    """
    Draw bounding boxes on the image using Matplotlib.

    Parameters:
    - image: The image array.
    - objects: List of objects with bounding box coordinates.
    """
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.imshow(image)
    
    # Define colors to cycle through for bounding boxes
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    
    # Iterate over each object and draw its bounding box
    for i, obj in enumerate(objects):
        for bbox in obj["coordinates"]:
            xmin = bbox["xmin"]
            ymin = bbox["ymin"]
            xmax = bbox["xmax"]
            ymax = bbox["ymax"]

            # Create a rectangle patch with a unique color
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, 
                linewidth=2, edgecolor=colors[i % len(colors)], facecolor='none'
            )
            ax.add_patch(rect)

    plt.axis('off')  # Hide axes
    plt.title('Image with Bounding Boxes')
    plt.show()

def visualize_image_with_boxes(image_path, object_coordinates):
    """aa
    Visualize the image with bounding boxes drawn using Matplotlib.

    Parameters:
    - image_path: Path to the image file.
    - object_coordinates: List of object coordinates with bounding boxes.
    """
    # Load the image using PIL and convert to numpy array
    image = Image.open(image_path)
    image_np = np.array(image)

    # Draw bounding boxes
    draw_bounding_boxes(image_np, object_coordinates)

# Set the row index to process (assuming you are continuing from row 219)

df = pd.read_csv("path_of_dataframe")
i = df.sample(n=1).index[0]
# Extract relevant information for the specified row
predicted_coordinates = df.iloc[i]['Predicted Coordinates']
predicted_depth = df.iloc[i]['Predicted Depth']
predicted_bbox = df.iloc[i]['Predicted Bounding Box']
error = df.iloc[i]['error']
predicted_bbox = ast.literal_eval(predicted_bbox)
image_path = df.iloc[i]['image_path']
json_path = df.iloc[i]['json_path']
print(json_path)
# Display extracted information for debugging
print(f"Image Path: {image_path}")
print(f"Predicted Coordinates: {predicted_coordinates}")
print(f"Predicted Depth: {predicted_depth}")
print(f"Error :" ,{error})
# Construct full paths for the image and JSON file
img_path = f"{image_path}"
json_file_path = f"{json_path}"

# Read and print data from the corresponding JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)
    print("Origin:", data['latitude_origin'], data['longitude_origin'])


visualize_image_with_boxes(img_path, [predicted_bbox])
visualize_image_with_boxes(img_path, data['result'])
