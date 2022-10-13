import util

# Extract the average angle between 3 random vertices of the shape
def A3(mesh):
    # Get 3 random vertices
    v1, v2, v3 = util.random_vertices(mesh, 3)
    # Calculate the angle between the 3 vertices
    angle = util.angle_between(v1, v2, v3)
    # Return the angle
    return angle
