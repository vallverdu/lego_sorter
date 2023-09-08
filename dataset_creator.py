import bpy
import random
import os
import math
import numpy as np
import csv
import uuid

import argparse
import sys


def hide_children(obj, hide_status):
    """Recursively hide an object and all of its children."""
    obj.hide_render = hide_status
    for child in obj.children:
        hide_children(child, hide_status)

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

def create_synthetic_images(output_folder, num_images, csv_filename, engine):

    # Set the Engine for rendering
    bpy.context.scene.render.engine = engine

    
    # Set the desired resolution
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    
    # Adjust camera's clip start for small objects
    bpy.data.cameras['Camera'].clip_start = 0.001
    
    # Filter out child objects and exclude Camera and Light
    pieces = [obj for obj in bpy.data.objects if obj.parent is None and obj.type == 'MESH' and obj.name not in ['Camera', 'Light']]

    for piece in pieces:
        hide_children(piece, True)  # hide all pieces initially for rendering
    
     # Open CSV file for writing
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        csvwriter.writerow(["filename", "brick_type", "rotation_x", "rotation_y", "rotation_z", "color_r", "color_g", "color_b"])


        for i in range(num_images):
            # Randomly select a piece model (from objects without parents)
            piece = random.choice(pieces)
            hide_children(piece, False)  # unhide the selected piece for rendering
            
            # Store the original rotation and location
            original_rotation = piece.rotation_euler.copy()
            original_location = piece.location.copy()
            
            # Move the piece to the origin
            piece.location = (0, 0, 0)
            
            # Randomly rotate the piece
            piece.rotation_euler = (random.uniform(0,6.28), random.uniform(0,6.28), random.uniform(0,6.28))
            print('initial rotation x: ',piece.rotation_euler.x,' y: ', piece.rotation_euler.y, ' z: ',piece.rotation_euler.z)
            
        
            # final_rotation =  apply_gravity_to_piece(piece)
            # piece.rotation_euler = apply_gravity_to_piece(piece)
            # print('final_rotation x: ',piece.rotation_euler.x,' y: ', piece.rotation_euler.y, ' z: ',piece.rotation_euler.z)
            
            # Adjust light values 
            light = bpy.data.objects['Light']  # Assuming the light source is named 'Light'
            original_light_location = light.location.copy()
            original_light_power = light.data.energy
            original_light_color = light.data.color.copy()

            # Random adjustments for each render:
            light.location = (
                original_light_location.x + random.uniform(-0.2, 0.2),
                original_light_location.y + random.uniform(-0.2, 0.2),
                original_light_location.z + random.uniform(-0.2, 0.2)
            )
            light.data.energy = original_light_power + random.uniform(-10, 10)
            light.data.color = (
                original_light_color[0] + random.uniform(-0.1, 0.1),
                original_light_color[1] + random.uniform(-0.1, 0.1),
                original_light_color[2] + random.uniform(-0.1, 0.1)
            )
            
            # Change the piece's color
            color = (random.random(), random.random(), random.random(), 1)
            if piece.material_slots:
                piece.material_slots[0].material.diffuse_color = color

            # Render the brick 
            # Generate a short UUID for filename
            short_uuid = str(uuid.uuid4())[:8]
            rgb_filename = f"brick_{short_uuid}"
            bpy.context.scene.render.filepath = os.path.join(output_folder, f"{rgb_filename}.png")
            bpy.ops.render.render(write_still=True)


            csvwriter.writerow([rgb_filename, piece.name, piece.rotation_euler.x, piece.rotation_euler.y, piece.rotation_euler.z, color[0], color[1], color[2]])


            # Restore the original rotation and location
            piece.rotation_euler = original_rotation
            piece.location = original_location
            
            # Restore the original Light values
            light.location = original_light_location
            light.data.energy = original_light_power
            light.data.color = original_light_color
            
            hide_children(piece, True)  # hide the piece again after rendering


if __name__ == "__main__":

    # Filter out Blender's default arguments

    # Check if '--' is in sys.argv, if not, then set argv to an empty list
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


    parser = argparse.ArgumentParser(description="Generate synthetic images in Blender.")
    parser.add_argument("--output_folder",    type=str, default="./dataset/", help="Path to the folder where images will be saved.")
    parser.add_argument("--num_images",       type=int, default=4000, help="Number of synthetic images to generate.")
    parser.add_argument("--csv_filename",     type=str, default="./dataset.csv", help="Path to the CSV file to store metadata.")
    parser.add_argument("--engine",           type=str, default="BLENDER_EEVEE", choices=["CYCLES", "BLENDER_EEVEE"], help="Blender rendering engine to use.")

    args = parser.parse_args(argv)

    create_synthetic_images(args.output_folder, args.num_images, args.csv_filename, args.engine)


'''
To execute from terminal
/Applications/Blender.app/Contents/MacOS/Blender -b lego_bricks_base1.blend -P dataset_creator.py -- --output-folder /path/to/output/folder --num-images 1000 --csv-filename /path/to/output/folder/data.csv --engine CYCLES
'''
