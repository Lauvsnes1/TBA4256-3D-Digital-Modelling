import laspy
import numpy as np
from sklearn import linear_model

def fit_plane(points):
    """
    Fit a plane to a set of 3D points.
    Returns the plane coefficients a, b, c, d.
    """
    centroid = np.mean(points, axis=0)
    _, _, V = np.linalg.svd(points - centroid)
    a, b, c = V[-1]
    d = -(centroid @ np.array([a, b, c]))
    
    # Normalize the plane coefficients
    norm = np.sqrt(a**2 + b**2 + c**2)
    return a/norm, b/norm, c/norm, d/norm


def distance_to_plane(a, b, c, d, point):
    """
    Compute the distance from a point to a plane.
    """
    x, y, z = point
    return abs(a*x + b*y + c*z + d)


def ransac_plane_fit(points, num_points=5, num_iterations=100, threshold=0.5):
    """
    Fit a plane to 3D points using the RANSAC algorithm.
    """
    best_inliers = []
    best_plane = (0, 0, 0, 0)

    for _ in range(num_iterations):
        # Randomly sample points
        sampled_points = points[np.random.choice(points.shape[0], num_points, replace=False)]
        
        # Fit a plane to the sampled points
        a, b, c, d = fit_plane(sampled_points)
        
        # Compute distances to the plane for all points
        distances = np.array([distance_to_plane(a, b, c, d, pt) for pt in points])
        
        # Determine inliers
        inliers = points[distances < threshold]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (a, b, c, d)

    # Refit the plane using the best inliers
    a, b, c, d = fit_plane(best_inliers)
    
    return a, b, c, d



def main():
    # Load the LAZ file
    in_file = laspy.read("data.laz")

    # Extract the points
    points = in_file.points

    # Extract the X, Y, and Z coordinates
    x_coords = in_file.points['X']
    y_coords = in_file.points['Y']
    z_coords = in_file.points['Z']

    # Convert them into a 2D numpy array
    points_np = np.vstack((x_coords, y_coords, z_coords)).T

    # Fit the plane using RANSAC
    a, b, c, d = ransac_plane_fit(points_np)
    print(a,b,c,d)

if __name__ == '__main__':
    main()
