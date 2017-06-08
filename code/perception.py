import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # TODO:
    # Apply a scaling and a translation
    xpix_translated = np.int_(xpos + (xpix_rot / scale))
    ypix_translated = np.int_(ypos + (ypix_rot / scale))
    # Return the result  
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped

# Define a function to detect rock
def colorDetection(image, color = None):
    assert color != None, 'A target color must be defined!'
    
    boundaries = {
        'red':([17, 15, 100], [50, 56, 200]),
        'blue':([86, 31, 4], [220, 88, 30]),
        'yellow':([150, 100, 0], [255, 200, 60]),
        'gray':([103, 86, 65], [145, 133, 128])
    }
    
    # find the colors within the specified boundaries and apply
    # the mask
    targetColor = boundaries.get(color)
    lower = np.array(targetColor[0])
    upper = np.array(targetColor[1])
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)
    gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
    output_binary = np.zeros(output.shape[:2])
    output_binary[gray > 0] = 255
    return output_binary

def distance2D(point1_x, point1_y, point2_x, point2_y):
    point1 = np.array((point1_y, point1_x))
    point2 = np.array((point2_y, point2_x))
    distance = np.linalg.norm(point1 - point2)
    return distance

def is_clear(Rover):
    clear = (np.sum(Rover.terrain_navigable[140:150,150:170]) > 130) & (np.sum(Rover.terrain_navigable[110:120,150:170]) > 100) & (np.sum(Rover.terrain_navigable[150:153,155:165]) > 20)
    # clear = (np.sum(Rover.terrain_navigable[150:153,155:165]) > 20)
    return clear



# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    
    dst_size = 5
    bottom_offset = 6
    src = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    dst = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                      [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                      [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset], 
                      [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                      ])
    # 2) Apply perspective transform
    warped = perspect_transform(Rover.img, src, dst)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    terrain_navigable = color_thresh(warped)
    terrain_nonNavigable = (terrain_navigable == 0)
    Rover.terrain_navigable = terrain_navigable
    Rover.clear_forward = is_clear(Rover)
    
    rock_featuremap = colorDetection(Rover.img, color='yellow')
    isRockFound = np.max(rock_featuremap)


    right_ahead = terrain_navigable[:, terrain_navigable.shape[1]//2:]
    Rover.right_ahead_pixels = np.sum(right_ahead)

    # # Calculate the left steering angle
    # left_ahead_nonZeros = left_ahead.nonzero()
    # left_ahead_distances, left_ahead_angles = to_polar_coords(left_ahead_nonZeros[1], left_ahead_nonZeros[0]) # Convert to polar coords
    # left_ahead_avg_angle = np.mean(left_ahead_angles)  # Compute the average angle
    # left_ahead_vg_angle_degrees = left_ahead_avg_angle * 180 / np.pi
    # left_ahead_steering = np.clip(left_ahead_vg_angle_degrees, -15, 15)
    # Rover.left_steer = left_ahead_steering
    #
    # # Calculate the right steering angle
    # right_ahead_nonZeros = right_ahead.nonzero()
    # right_ahead_distances, right_ahead_angles = to_polar_coords(right_ahead_nonZeros[1],
    #                                                             right_ahead_nonZeros[0])  # Convert to polar coords
    # right_ahead_avg_angle = np.mean(right_ahead_angles)  # Compute the average angle
    # right_ahead_vg_angle_degrees = right_ahead_avg_angle * 180 / np.pi
    # right_ahead_steering = np.clip(right_ahead_vg_angle_degrees, -15, 15)
    # Rover.right_steer = right_ahead_steering
    
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    #gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    #zozzero_y, nonzero_x = gray.nonzero()
    #Rover.vision_image[zozzero_y, nonzero_x, 0] = 255
    #Rover.vision_image[terrain_navigable] = np.array([0, 0, 255])
    Rover.vision_image[:,:,0] = terrain_nonNavigable*255
    Rover.vision_image[:,:,2] = terrain_navigable*255
    # 5) Convert map image pixel values to rover-centric coords
    xpix, ypix = rover_coords(terrain_navigable)
    # 6) Convert rover-centric pixel values to world coordinates
    navigable_x_world, navigable_y_world = pix_to_world(xpix, ypix, Rover.pos[0], Rover.pos[1], Rover.yaw, 200, 10)
    #print(xpix, ypix, Rover.pos[0], Rover.pos[1])
    #print(navigable_x_world, navigable_y_world, 'Those are the world coord!')
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    Rover.worldmap[navigable_y_world, navigable_x_world, 0] += 1
    if isRockFound:
        warped_rock_featuremap = perspect_transform(rock_featuremap, src, dst)
        rock_xpix, rock_ypix = rover_coords(warped_rock_featuremap)
        rock_x_world, rock_y_world = pix_to_world(rock_xpix, rock_ypix, Rover.pos[0], Rover.pos[1], Rover.yaw, 200, 10)
        Rover.worldmap[rock_y_world, rock_x_world, 2] = 255
        
        rock_centric_pixel_distances, rock_centric_angles = to_polar_coords(rock_xpix, rock_ypix)
        Rover.nav_dists = rock_centric_pixel_distances
        Rover.nav_angles = rock_centric_angles
        avg_angle_degrees = np.mean(rock_centric_angles)
        steer_ = np.clip(np.mean(avg_angle_degrees * 180/np.pi), -15, 15)
        Rover.rock_in_view = True
        
        rock_nonzero_y, rock_nonzero_x = warped_rock_featuremap.nonzero()
        index_nearest = np.argmax(rock_nonzero_y)
        center_y = Rover.img.shape[0] - bottom_offset
        center_x = Rover.img.shape[1]/2
        rock_distance = distance2D(center_x, center_y, rock_nonzero_x[index_nearest], rock_nonzero_y[index_nearest])
        Rover.rock_in_distance = rock_distance
        # print('Rock is {} meters away! Turn {} degrees.'.format(rock_distance, steer_))
    
    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    else:
        Rover.rock_in_view = False
        Rover.rock_in_distance = None
        rover_centric_pixel_distances, rover_centric_angles = to_polar_coords(xpix, ypix)
        Rover.nav_dists = rover_centric_pixel_distances
        Rover.nav_angles = rover_centric_angles
    
 
    
    
    return Rover