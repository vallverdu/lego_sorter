import bpy
# from bpycv import render_segmentation, object_to_img
# from bpycv.transform_utils import resize, random_scale, random_translation, random_rotation
import bpycv
import cv2

import random
import os
import math
import mathutils
import numpy as np
import csv
import uuid

import argparse
import sys


class Balancer:
    def __init__(self, categories, desired_samples_per_category):
        self.categories = categories
        self.desired_samples_per_category = desired_samples_per_category
        self.current_samples = {category: 0 for category in categories}

    def get_next_category(self):
        # Return the category that has the least samples
        return min(self.current_samples, key=self.current_samples.get)

    def update_category(self, category):
        if category in self.current_samples:
            self.current_samples[category] += 1

    def is_balanced(self):
        # Check if the dataset is balanced
        return all(count == self.desired_samples_per_category for count in self.current_samples.values())

def hide_children(obj, hide_status):
    """Recursively hide an object and all of its children."""
    obj.hide_render = hide_status
    for child in obj.children:
        hide_children(child, hide_status)


def generate_random_rotation():
    '''
    rot_x will have a high chance of being 0 or 180.
    rot_y can take the values 0, 90, 180, or 270 degrees but only if rot_x is 0.
    rot_z is always applied and can take any value between 0 and 360 degrees.
    
    
    Modifying the generate_random_rotation function to prioritize rotations on x axis as 0 or 180
    '''

    # Z-axis rotation: any value between 0 and 360 degrees
    rot_z = random.uniform(0, 360)

    # Prioritize x-axis rotations: 0 or 180 degrees
    rot_x = random.choice([0, 180, 0,0,0, 180,0,0, 90, 270])  # Higher probability for 0 and 180

    # Y rotations: 0, 90, 180, or 270 degrees but only if rot_x is 0
    if rot_x == 0:
        rot_y = random.choice([0, 0,0,90,0, 180,0,0,0, 180, 270,0, 180,0, 180])
    else:
        rot_y = 0


    return rot_x, rot_y, rot_z



def ensure_piece_has_mass(piece):
    """
    Ensure that the given piece and its children (sub-pieces) have a non-zero mass.
    """
    # Check the main piece
    if not piece.rigid_body:
        bpy.ops.rigidbody.object_add({'object': piece})
        piece.rigid_body.mass = 0.1
    elif piece.rigid_body.mass == 0:
        piece.rigid_body.mass = 0.1
        
    # Check all its children (sub-pieces)
    for child in piece.children:
        if not child.rigid_body:
            bpy.ops.rigidbody.object_add({'object': child})
            child.rigid_body.mass = 0.1
        elif child.rigid_body.mass == 0:
            child.rigid_body.mass = 0.1



def apply_gravity_to_piece(piece):
    """
    Apply gravity to the LEGO piece and let it settle on the ground plane.
    Returns the final rotation of the piece after settling.
    """
    # Ensure there's a ground plane (a large flat mesh) named 'Ground' in your Blender file

    ensure_piece_has_mass(piece)
    
    ground = bpy.data.objects.get('Ground')
    if not ground:
        # If not, you can create one
        # Create a plane for the piece to fall onto and enable rigid body physics for the plane:
        bpy.ops.mesh.primitive_plane_add(size=10,  location=(0, 0, -0.05))
        ground = bpy.context.object
        ground.name = 'Ground'
    


    # Set the ground as a passive rigid body so pieces can collide with it
    ground.select_set(True)
    bpy.ops.rigidbody.object_add(type='PASSIVE')
    
    # Set the piece as an active rigid body
    piece.select_set(True)
    bpy.ops.rigidbody.object_add(type='ACTIVE')
    
    bpy.context.scene.frame_end = 250  # or another appropriate value
    
    # Run the physics simulation
    bpy.ops.ptcache.bake_all(bake=True)
    
    # Capture the final rotation
    final_rotation = piece.rotation_euler.copy()
    
    # Apply the transformation after simulation
    bpy.ops.object.visual_transform_apply()
    
    # Remove rigid body physics from the piece so it won't be affected in further simulations
    # bpy.ops.rigidbody.object_remove()
    
    # # Reset the piece's location to (0,0,0)
    # piece.location = (0, 0, 0)
    
    # # Return the piece to original layer and deselect
    # piece.select_set(False)
    
    # Clean up
    bpy.data.objects.remove(ground)

    return final_rotation

# Define a function to move the lights randomly
def move_lights_randomly(lights, max_translation=1.0, max_energy= 50, max_color = 0.2):
    original_positions = {}
    original_light_power = {}
    original_light_color = {}
    
    for light in lights:
        # Store the original position
        original_positions[light.name] = light.location.copy()
        original_positions[light.name] = light.location.copy()
        original_positions[light.name] = light.location.copy()
        
        original_light_power[light.name] = light.data.energy
        original_light_color[light.name] = light.data.color.copy()

        # Apply random translation
        light.location.x += random.uniform(-max_translation, max_translation)
        light.location.y += random.uniform(-max_translation, max_translation)
        light.location.z += random.uniform(-max_translation, max_translation)

        light.data.energy += random.uniform(-max_energy, max_energy)

        light.data.color[0] += random.uniform(-max_color, max_color)
        light.data.color[1] += random.uniform(-max_color, max_color)
        light.data.color[2] += random.uniform(-max_color, max_color)
    
    return original_positions, original_light_power, original_light_color

# Define a function to reset the lights to their original positions
def reset_lights_to_original(lights, original_positions, original_light_power, original_light_color):
    for light in lights:
        light.location = original_positions[light.name]
        light.data.energy = original_light_power[light.name]
        light.data.color = original_light_color[light.name]

def create_synthetic_images(output_folder, num_images, csv_filename, engine, debug, visualize):

    # Set the Engine for rendering
    bpy.context.scene.render.engine = engine

    os.makedirs(output_folder, exist_ok=True)
    

    # Set the desired resolution
    bpy.context.scene.render.resolution_x = 256
    bpy.context.scene.render.resolution_y = 256

    # Assuming you're working with the active camera in the scene
    camera = bpy.context.scene.camera.data

    # Image dimensions
    image_width = bpy.context.scene.render.resolution_x
    image_height = bpy.context.scene.render.resolution_y

    # Calculate fx and fy
    fx = camera.lens * image_width / camera.sensor_width
    fy = camera.lens * image_height / camera.sensor_height

    # Principal point
    cx = image_width / 2
    cy = image_height / 2

    # Print the intrinsics matrix
    print(f"Intrinsics matrix K:\n[[{fx}, 0, {cx}],\n [0, {fy}, {cy}],\n [0, 0, 1]]")
    
    # Adjust camera's clip start for small objects
    bpy.data.cameras['Camera'].clip_start = 0.001

    # Make the world dark
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    bg_node = world.node_tree.nodes["Background"]
    bg_node.inputs["Strength"].default_value = 0
    
    scene = bpy.context.scene
    scene.view_settings.exposure = 0


    # Create lights
    locations = [[0,0,5], [0,-2,5], [2,0,5]]
    lights = []

    for idx, location in enumerate(locations):
        bpy.ops.object.light_add(type='POINT', align='WORLD', location=location)
        light = bpy.context.active_object
        light.name = f"MyPointLight{idx+1}"  # Name the light based on its index
        light.data.energy = 1000
        lights.append(light)

    

    # Filter out child objects and exclude Camera and Light
    pieces = [obj for obj in bpy.data.objects if obj.parent is None and obj.type == 'MESH' and obj.name not in ['Camera', 'Light']]

    # Create a balancer to ensure a balanced dataset
    unique_piece_types = list(set(pieces))
    balancer = Balancer(unique_piece_types, num_images)

    for piece in pieces:
        hide_children(piece, True)  # hide all pieces initially for rendering
    
     # Open CSV file for writing
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        csvwriter.writerow(["filename", "brick_type", "rotation_x", "rotation_y", "rotation_z", "color_r", "color_g", "color_b"])

        for i in range(num_images * len(pieces)):

            # Randomly select a piece model (from objects without parents)
            # piece = random.choice(pieces)

            # Decide the next category using the balancer
            piece = balancer.get_next_category()

            # Update balancer after rendering the piece
            balancer.update_category(piece)

            hide_children(piece, False)  # unhide the selected piece for rendering


            # Store the original rotation and location
            original_rotation = piece.rotation_euler.copy()
            original_location = piece.location.copy()
            
            # Move the piece to the origin
            piece.location = (0, 0, 0)
            
            # Randomly rotate the piece
            # piece.rotation_euler = (random.uniform(0,6.28), random.uniform(0,6.28), random.uniform(0,6.28))
            rot_x, rot_y, rot_z = generate_random_rotation()
            
            rot_x_rad = math.radians(rot_x)
            rot_y_rad = math.radians(rot_y)
            rot_z_rad = math.radians(rot_z)

            # eul = mathutils.Euler((rot_x_rad, rot_y_rad, rot_z_rad), 'XYZ')
            eul = mathutils.Euler((rot_x_rad, rot_y_rad, rot_z_rad), 'XYZ')
            piece.rotation_euler = eul
            print('initial rotation x: ',piece.rotation_euler.x,' y: ', piece.rotation_euler.y, ' z: ',piece.rotation_euler.z)
            
        
            # final_rotation =  apply_gravity_to_piece(piece)
            # piece.rotation_euler = apply_gravity_to_piece(piece)
            # print('final_rotation x: ',piece.rotation_euler.x,' y: ', piece.rotation_euler.y, ' z: ',piece.rotation_euler.z)
            
            # Move the lights randomly
            original_positions, original_light_power, original_light_color = move_lights_randomly(lights,0.3)
            
            # Change the piece's color
            color = (random.random(), random.random(), random.random(), 1)

            if piece.material_slots:
                mat = piece.material_slots[0].material
                nodes = mat.node_tree.nodes
                principled = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')
                principled.inputs[0].default_value = color  # Base Color
                
            original_scale = piece.scale.copy()  # Store the original scale
            piece.scale *= 10  # Scale the object

            # Render the RGB 
            # Generate a short UUID for filename
            short_uuid = str(uuid.uuid4())[:8]
            rgb_filename = f"brick_{short_uuid}"
            # bpy.context.scene.render.filepath = os.path.join(output_folder, f"{rgb_filename}_rgb.png")
            # bpy.ops.render.render(write_still=True)

            # Render the instance segmentation and convert it to a binary mask
            mask_filename = f"mask_{short_uuid}"
            
            piece["inst_id"] = 2 * 1000 + i

            result = bpycv.render_data()

            # print(result["inst"])

            # print(result["depth"])

            if visualize :
                # Visualize the instance image with a colormap
                # Normalize the instance image for better visualization
                norm_instance = cv2.normalize(result["inst"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                # Use a colormap to visualize distinct values
                colored_instance = cv2.applyColorMap(norm_instance, cv2.COLORMAP_JET)

                cv2.imshow('Instance Visualization', colored_instance)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


            cv2.imwrite(
               os.path.join(output_folder, f"{rgb_filename}_rgb.png"), result["image"][..., ::-1]
            )  # transfer RGB image to opencv's BGR


            binary_mask = np.where(result["inst"] > 0, 255, 0).astype(np.uint8)
            cv2.imwrite(os.path.join(output_folder, f"{rgb_filename}_mask.png"), binary_mask)


            if debug :

                # save instance map as 16 bit png
                cv2.imwrite(os.path.join(output_folder, f"{rgb_filename}_inst.png"), np.uint16(result["inst"]))
                # the value of each pixel represents the inst_id of the object

                # convert depth units from meters to millimeters
                depth_in_mm = result["depth"] * 10000
                cv2.imwrite(os.path.join(output_folder, f"{rgb_filename}_depth.png"), np.uint16(depth_in_mm))  # save as 16bit png

                # visualization instance mask, RGB, depth for human
                cv2.imwrite(os.path.join(output_folder, f"{rgb_filename}_vis.jpg"), result.vis()[..., ::-1])

           

            csvwriter.writerow([rgb_filename, piece.name, piece.rotation_euler.x, piece.rotation_euler.y, piece.rotation_euler.z, color[0], color[1], color[2]])


            # Restore the original rotation, location & scale
            piece.rotation_euler = original_rotation
            piece.location = original_location
            piece.scale = original_scale

            # Restore the original Light values
            reset_lights_to_original(lights, original_positions, original_light_power, original_light_color)
            
            hide_children(piece, True)  # hide the piece again after rendering


if __name__ == "__main__":

    # Filter out Blender's default arguments

    # Check if '--' is in sys.argv, if not, then set argv to an empty list
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


    parser = argparse.ArgumentParser(description="Generate synthetic images in Blender.")
    parser.add_argument("--output_folder",    type=str, default="./dataset/", help="Path to the folder where images will be saved.")
    parser.add_argument("--num_images",       type=int, default=2, help="Number of synthetic images per category to generate.")
    parser.add_argument("--csv_filename",     type=str, default="./dataset.csv", help="Path to the CSV file to store metadata.")
    parser.add_argument("--engine",           type=str, default="BLENDER_EEVEE", choices=["CYCLES", "BLENDER_EEVEE"], help="Blender rendering engine to use.")
    parser.add_argument('--debug',            action='store_true', help='Set this flag to debug')
    parser.add_argument('--visualize',        action='store_true', help='Set this flag to visualize')

    args = parser.parse_args(argv)

    create_synthetic_images(args.output_folder, args.num_images, args.csv_filename, args.engine,args.debug, args.visualize)


'''
To execute from terminal
/Applications/Blender.app/Contents/MacOS/Blender -b lego_bricks_base2.blend -P dataset_creator.py -- --debug --visualize --output-folder /path/to/output/folder --num-images 1000 --csv-filename /path/to/output/folder/data.csv --engine CYCLES

Install bpycv
https://github.com/vallverdu/bpycv

# Ensure pip: equl to /blender-path/3.xx/python/bin/python3.10 -m ensurepip
/Applications/Blender.app/Contents/MacOS/Blender -b --python-expr "from subprocess import sys,call;call([sys.executable,'-m','ensurepip'])"
# Update pip toolchain
/Applications/Blender.app/Contents/MacOS/Blender -b --python-expr "from subprocess import sys,call;call([sys.executable]+'-m pip install -U pip setuptools wheel'.split())"
# pip install bpycv
/Applications/Blender.app/Contents/MacOS/Blender -b --python-expr "from subprocess import sys,call;call([sys.executable]+'-m pip install -U bpycv'.split())"
# Check bpycv ready
/Applications/Blender.app/Contents/MacOS/Blender -b -E CYCLES --python-expr "import bpycv,cv2;d=bpycv.render_data();bpycv.tree(d);cv2.imwrite('/tmp/try_bpycv_vis(inst-rgb-depth).jpg', d.vis()[...,::-1])"

'''
