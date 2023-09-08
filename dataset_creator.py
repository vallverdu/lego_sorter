import bpy
import random
import os
import math
import numpy as np
import csv
import uuid



def hide_children(obj, hide_status):
    """Recursively hide an object and all of its children."""
    obj.hide_render = hide_status
    for child in obj.children:
        hide_children(child, hide_status)

def create_synthetic_images(output_folder, num_images, csv_filename):
    
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

            
            '''
            # Realistic Rotations using Physics:
            piece.rigid_body_type = 'ACTIVE' 
            
            # Create a plane for the piece to fall onto and enable rigid body physics for the plane:
            bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, -0.1))
            plane = bpy.context.active_object
            
            plane.rigid_body_type = 'PASSIVE'  
            
            
            # Run the physics simulation for a short period
            bpy.context.scene.frame_set(0)
            bpy.ops.screen.animation_play()
            bpy.context.scene.frame_set(10)
            bpy.ops.screen.animation_play()
            
            # Capture the rotation of the piece after it has fallen
            final_rotation = piece.rotation_euler.copy()
            
            # Clean up
            bpy.data.objects.remove(plane)
            '''
        

            
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


# To call the function:
create_synthetic_images("/Users/jordivallverdu/Documents/360code/apps/lego_sorter/dataset/", 5)

'''
To execute from terminal
blender -b lego_bricks_base.blend -P dataset_creator.py
'''
