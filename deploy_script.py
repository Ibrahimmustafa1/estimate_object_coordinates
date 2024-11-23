import matplotlib.pyplot as plt
from PIL import Image
import torch
import cv2
import torchvision.transforms as transforms
import numpy as np
from transformers import AutoImageProcessor, DPTForDepthEstimation
import math
from exif import Image as meta
import json
import matplotlib.patches as patches
import math
import ast
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# intialise and download depth model
image_processor = AutoImageProcessor.from_pretrained("facebook/dpt-dinov2-large-kitti")
model = DPTForDepthEstimation.from_pretrained("facebook/dpt-dinov2-large-kitti")
model.config.depth_estimation_type = "metric"  # Metric depth for real-world coordinates
# add these configs to the model
model = model.to(device)


def estimate_depth_map(image):
    # Prepare image for the model
    inputs = image_processor(images=image, return_tensors="pt")
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    depth_map = prediction.squeeze().cpu().numpy()
    
    plt.imshow(depth_map, cmap='inferno')  # Use 'inferno' colormap or any other colormap you like
    plt.colorbar(label='Depth Values')  # Add color bar with label
    plt.title('Depth Map')  # Add a title to the plot
    plt.show()
    return depth_map

# Function to calculate field of view (FoV)
def calculate_fov(focal_length_mm, sensor_width_mm, sensor_height_mm):
    """
    Calculate the horizontal and vertical field of view (FoV) based on focal length and sensor size.
    """
    hfov = 2 * np.arctan(sensor_width_mm / (2 * focal_length_mm)) * (180 / np.pi)
    vfov = 2 * np.arctan(sensor_height_mm / (2 * focal_length_mm)) * (180 / np.pi)
    return hfov, vfov



def estimate_nearest_object_within_distance(
    depth_map, object_bboxes, image_width_px, image_height_px, latitude_origin, 
    longitude_origin, hfov, vfov, bearing, distance_threshold=30, 
    width_scaling_factor=1
):
    """
    Estimate the coordinates of the nearest object within a given distance threshold after merging overlapping boxes.
    Returns the closest object based on the reduced depth after adjusting bounding boxes.

    Parameters:
        depth_map (np.array): Depth map of the image.
        object_bboxes (list): List of bounding boxes with object coordinates.
        image_width_px (int): Width of the image in pixels.
        image_height_px (int): Height of the image in pixels.
        latitude_origin (float): Latitude of the camera's origin.
        longitude_origin (float): Longitude of the camera's origin.
        hfov (float): Horizontal field of view in degrees.
        vfov (float): Vertical field of view in degrees.
        bearing (float): Bearing angle of the camera.
        distance_threshold (float): Maximum distance (meters) to include an object.
        width_scaling_factor (float): Factor by which to reduce the bounding box width for depth calculation.

    Returns:
        tuple: Tuple with nearest coordinates (latitude, longitude), bounding box, and reduced depth of the closest object.
    """

    nearby_objects = []

    # Calculate angle per pixel
    angle_per_pixel_horizontal = hfov / image_width_px
    angle_per_pixel_vertical = vfov / image_height_px

    # Earth radius in meters
    R = 6371000

    # Find objects within the distance threshold
    for bbox in object_bboxes:
        xmin = bbox['coordinates'][0]['xmin']
        xmax = bbox['coordinates'][0]['xmax']
        ymin = bbox['coordinates'][0]['ymin']
        ymax = bbox['coordinates'][0]['ymax']
        
        # Adjust width by scaling factor
        center_x = (xmin + xmax) / 2
        new_width = int((xmax - xmin) * width_scaling_factor)
        
        new_xmin = max(int(center_x - new_width / 2), 0)
        new_xmax = min(int(center_x + new_width / 2), depth_map.shape[1])
        
        # Adjust height by reducing from the top by half, no scaling applied
        new_height = ymax - ymin  # Original height of the bounding box
        
        # Skip boxes with new_height < 90
        if new_height <= 120:
            continue  # Skip this bounding box if height is too small
        
        new_ymin = max(int(ymin + new_height / 2), 0)  # Shift ymin down by half the height
        new_ymax = ymax  # Keep ymax the same, so we only reduce from the top

        # Calculate the reduced depth within the adjusted bounding box
        reduced_depth = np.median(depth_map[new_ymin:new_ymax, new_xmin:new_xmax])
        
        # Only process objects that are within the distance threshold
        if reduced_depth <= 30:
            adjusted_latitude, adjusted_longitude = latitude_origin, longitude_origin
            # Calculate angle offsets
            center_x_reduced = (new_xmin + new_xmax) / 2
            center_y_reduced = (new_ymin + new_ymax) / 2
            angle_offset_horizontal = (center_x_reduced - (image_width_px / 2)) * angle_per_pixel_horizontal
            angle_offset_vertical = (center_y_reduced - (image_height_px / 2)) * angle_per_pixel_vertical

            total_bearing = bearing + angle_offset_horizontal
            adjusted_distance = reduced_depth * math.cos(math.radians(angle_offset_vertical))

            # Calculate coordinates based on adjusted distance and bearing
            lat_rad = math.radians(adjusted_latitude)
            lon_rad = math.radians(adjusted_longitude)
            bearing_rad = math.radians(total_bearing)

            new_lat = math.asin(math.sin(lat_rad) * math.cos(adjusted_distance / R) +
                                math.cos(lat_rad) * math.sin(adjusted_distance / R) * math.cos(bearing_rad))
            
            new_lon = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(adjusted_distance / R) * math.cos(lat_rad),
                                           math.cos(adjusted_distance / R) - math.sin(lat_rad) * math.sin(new_lat))

            nearest_coordinates = (math.degrees(new_lat), math.degrees(new_lon))
            nearby_objects.append((nearest_coordinates, bbox, reduced_depth))

    # Sort by reduced depth (ascending order) to get the closest object
    nearby_objects_sorted = sorted(nearby_objects, key=lambda x: x[2])

    # Return only the nearest object (if no object is found within threshold, return None)
    return nearby_objects_sorted[0] if nearby_objects_sorted else None


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


def main(image_path, depth_map, focal_length_mm, latitude_origin, longitude_origin, bearing, sensor_width_mm, sensor_height_mm, bbox, distance_threshold=25):
    """
    Main function to estimate object coordinates within a distance threshold.

    Parameters:
        image_path (str): Path to the image file.
        depth_map (np.array): Depth map of the image.
        focal_length_mm (float): Focal length of the camera in millimeters.
        latitude_origin (float): Latitude of the camera's origin.
        longitude_origin (float): Longitude of the camera's origin.
        bearing (float): Bearing angle of the camera.
        sensor_width_mm (float): Sensor width in millimeters.
        sensor_height_mm (float): Sensor height in millimeters.
        bbox (list): List of bounding boxes for objects.
        distance_threshold (float): Maximum distance (meters) to include an object.

    Returns:
        list: List of tuples with coordinates, bounding boxes, and reduced depth of objects within the distance threshold.
    """

    # Load the image to get its dimensions
    image = Image.open(image_path)
    image_width_px = image.width   
    image_height_px = image.height  

    # Calculate field of view (FoV)
    hfov, vfov = calculate_fov(focal_length_mm, sensor_width_mm, sensor_height_mm)

    # Estimate object coordinates within the specified distance threshold
    nearby_objects = estimate_nearest_object_within_distance(
        depth_map, bbox, image_width_px, image_height_px, latitude_origin, longitude_origin,
        hfov, vfov, bearing
    )

    # Return the list of nearby objects with their coordinates, bounding boxes, and depths
    return nearby_objects



def prepare_json_file(file_path ,image_path):
    """
    Prepare and update JSON data with additional metadata from an image.

    This function loads JSON data from a specified file, adds metadata from an associated image (such as bearing, focal length, and model),
    and removes unnecessary fields from each object in the result. Raises an error if no result is found in the JSON file.

    Parameters:
        file_path (str): Path to the JSON file containing object data.
        image_path (str): Path to the image file, used to extract additional metadata.

    Returns:
        dict: Updated JSON data with additional metadata and cleaned object data.
        
    Raises:
        ValueError: If no results are found in the JSON file.
    """
    
    with open(file_path) as f:
        data = json.load(f)
        if data['result']:
            with open(image_path, 'rb') as image_file:
                image = meta(image_file)
                data['image_path'] = image_path
                data['bearing'] = image['gps_img_direction']
                data['focal_length'] = image['focal_length']
                data['model'] = image['model']
                data["latitude_origin"] = float(data['result'][0]['latitude'])
                data["longitude_origin"] = float(data['result'][0]['longitude'])
                for obj in data['result']:
                    obj.pop('latitude', None)
                    obj.pop('longitude', None)  
                    obj.pop('dms_id', None)
                    obj.pop('classification_id', None)
                    obj.pop('type', None)
        else:
             print('No result found in the json file')
             return
    return data


js_file = "json_file"
im_path = "image_path"
sensor_width_mm = 12.35
sensor_height_mm = 9.63
image = Image.open(im_path)
config = prepare_json_file(js_file,im_path)
depth_map = estimate_depth_map(image)
light_poles_coordinates = []
near_objects = []
reduced_depths = []
if config:
    nearest_object = main(
        config["image_path"],
        depth_map,
        config["focal_length"],
        float(config["latitude_origin"]),
        float(config["longitude_origin"]),
        config["bearing"],
        12.35,
        9.63,
        config['result']
    )
    print(nearest_object)
    if nearest_object:
        nearest_coordinates, bbox, reduced_depth = nearest_object
        visualize_image_with_boxes(im_path,config['result'])
        visualize_image_with_boxes(im_path, [bbox])
        print("Light pole predicted Cordinates : " ,nearest_coordinates)
    else:
        print("No nearby object found within the distance threshold.")
