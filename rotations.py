import random

def generate_random_rotation():
    # Z-axis rotation: any value between 0 and 360 degrees
    rot_z = random.uniform(0, 360)

    # X and Y rotations: 0, 90, 180, or 270 degrees
    possible_rotations = [0, 90, 180, 270]
    rot_x = random.choice(possible_rotations)
    rot_y = random.choice(possible_rotations)

    return rot_x, rot_y, rot_z

# Test the function
for _ in range(5):
    print(generate_random_rotation())