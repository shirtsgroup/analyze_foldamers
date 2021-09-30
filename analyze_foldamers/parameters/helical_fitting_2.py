import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize, basinhopping
from mpl_toolkits import mplot3d


def fit_helix_to_points(data_points, x0):
    """
    Given a series of 3D data points fits a helix to those points
    and returns fitting stats
    """
    # Fit cylinder of projection of points

    def cylinder_error(coeffs, data_points):
        # coeffs: 0:r_sq, 1,2,3:C[0,1,2], 4,5,6:W[0,1,2]
        r_sq = coeffs[0] # 1 parameter
        C = np.array([coeffs[1], coeffs[2], coeffs[3]]) # 3 parameters
        W = np.array([coeffs[4], coeffs[5], coeffs[6]]) # 2/3 parameters?
        # residuals = np.zeros(data_points.shape[0])
        error = 0

        for i in range(data_points.shape[0]):
            error += (np.dot(np.dot(data_points[i, :] - C, np.eye(3) - np.outer(W,W)).transpose(),data_points[i, :] - C) - r_sq)**2
        
        return error

    xmin = [0, -10, -10, -10, -1, -1, -1]
    xmax = [np.inf, 10, 10, 10, 1, 1, 1]

    bounds = [(low, high) for low, high in zip(xmin, xmax)]

    cyl_results = minimize(cylinder_error, x0, args=(data_points), method="L-BFGS-B", bounds=bounds)

    center = cyl_results.x[1:4]
    axis = cyl_results.x[4:]
    axis_norm = axis/np.sqrt(np.dot(axis, axis))

    # Fit Helix rotate coordinate system

    centered = data_points - center

    cart_axis = np.array([0,0,1])
    ortho_norm = np.cross(axis_norm, cart_axis)
    ortho_norm = ortho_norm / np.sqrt(np.dot(ortho_norm, ortho_norm))

    rotation_matrix = np.array([ortho_norm, np.cross(axis_norm, ortho_norm), axis_norm])

    transformed_points = np.dot(data_points, rotation_matrix.transpose())

    def helix_error(coeffs, data_points):
        # coeffs: 0:radius, 1: w, 2:phi
        radius = coeffs[0] # 1 parameter
        w = coeffs[1] # 1 parameter
        phi = coeffs[2] # 1 parameter

        error = 0
        for i in range(data_points.shape[0]):
            error += np.square(data_points[i,0] - radius*np.cos(w*data_points[i,2] + phi)) + np.square(data_points[i,1] - radius*np.sin(w*data_points[i,2] + phi))
        
        return error

    helix_results = minimize(helix_error, x0, args=(data_points), method="L-BFGS-B", bounds=bounds)



    return(*helix_results.x, np.linalg.inv(rotation_matrix), center, axis_norm)



def generate_test_helix(radius, pitch, res_per_turn, noise, n_points, phase_shift = 0, rotation = [0, 0, 0], centered = True):
    """
    Generate a test points to fit with fit_helix_to_points function
    """

    z_height = (n_points-1)*pitch/res_per_turn
    z_line = np.linspace(0, z_height, n_points).reshape(n_points, 1)
    x_line = radius*np.sin(2*np.pi*z_line/pitch + phase_shift)
    y_line = radius*np.cos(2*np.pi*z_line/pitch + phase_shift)

    z_data = z_line + np.random.normal(scale=noise, size = z_line.shape)
    x_data = x_line + np.random.normal(scale=noise, size = z_line.shape)
    y_data = y_line + np.random.normal(scale=noise, size = z_line.shape)

    data_coordinates = np.concatenate((x_data, y_data, z_data), axis=1)

    rotation_matrix = lambda a, b, c: np.array([[np.cos(a)*np.cos(b), np.cos(a)*np.sin(b)*np.sin(c) - np.sin(a)*np.cos(c), np.cos(a)*np.sin(b)*np.cos(c) + np.sin(a)*np.sin(c)],[np.sin(a)*np.cos(b), np.sin(a)*np.sin(b)*np.sin(c)+np.cos(a)*np.cos(c), np.sin(a)*np.sin(b)*np.cos(c) - np.cos(a)*np.sin(c)],[-np.sin(b), np.cos(b)*np.sin(c), np.cos(b)*np.cos(c)]])

    data_coordinates = np.dot(data_coordinates, rotation_matrix(*rotation).transpose())

    if centered is True:
        mean_helix_center = np.array([np.mean(data_coordinates[:,0]), np.mean(data_coordinates[:, 1]), np.mean(data_coordinates[:,2])])
        data_coordinates = data_coordinates - mean_helix_center

    return data_coordinates


def main():
    # Generate sample helix and center at origin
    test_helix = generate_test_helix(4, 4, 4.5, 0.05, 15, phase_shift = 0, rotation=[0,0,0.3])
    
    # Plot helix points
    fig = plt.figure()
    ax = fig.gca(projection='3d',autoscale_on=False)
    ax.scatter3D(test_helix[:, 0], test_helix[:, 1], test_helix[:, 2], s=125)
    ax.plot3D(test_helix[:, 0], test_helix[:, 1], test_helix[:, 2])

    # Fit helix points to helix equation
    x0 = np.array([1, 0, 0, 0, 0, -0.6, 1])
    radius, w, phi, rotation = fit_helix_to_points(test_helix, x0)

    # Show projection of helix on identified plane
    point = np.array(x_solved[1:4])
    ax.scatter3D(point[0], point[1], point[2], c="black")
    normal = np.array(x_solved[4:])
    normal = normal / np.sqrt(np.dot(normal, normal))
    print("Center:", point)
    print("Average Point:", mean_helix_center)
    print("Normal:", normal)
    ax.plot3D(np.array([0, normal[0]])+point[0], np.array([0, normal[1]])+point[1], np.array([0, normal[2]])+point[2], "black")

    # Plot project of helix onto plane
    centered = test_helix - point
    dist = np.dot(centered, normal)
    projected_points = test_helix - dist[:, np.newaxis] * normal[np.newaxis, :]
    ax.scatter3D(projected_points[:, 0], projected_points[:, 1], projected_points[:, 2], c="black")
    ax.plot3D(projected_points[:, 0], projected_points[:, 1], projected_points[:, 2], "black")


    plt.show()



if __name__ == "__main__":
    main()