import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize, basinhopping
from mpl_toolkits import mplot3d
from scipy.optimize._trustregion_constr.minimize_trustregion_constr import TERMINATION_MESSAGES
from analyze_foldamers.parameters.helical_fitting_2 import * 
import mdtraj as md


def main():
    helix_traj = md.load("LHH.pdb")
    top = helix_traj.topology
    bb_helix = helix_traj.atom_slice(top.select("name BB1 BB2 BB3"))
    
    # Scaling helix to help with fitting. Computers are bad at small numbers
    test_helix = 100*bb_helix.xyz[0]

    # Fit helix points to helix equation
    entries = []
    RMSEs = []
    for i in range(20):
        x0 = np.array([1, 0, 0, 0, 0, 0, 0])
        radius, w, phi, z_tot, rotation, center, normal, sse_helix, sse_cylinder = fit_helix_to_points(test_helix, x0)
        RMSE_tot = np.sqrt(sse_cylinder/test_helix.shape[0]) + np.sqrt(sse_helix/test_helix.shape[0])
        entries.append([radius, w, phi, z_tot, rotation, center, normal, sse_helix, sse_cylinder])
        RMSEs.append(RMSE_tot)

    i_min = RMSEs.index(np.min(RMSEs))

    radius, w, phi, z_tot, rotation, center, normal, sse_helix, sse_cylinder = entries[i_min]
    
    # Plot helix points
    fig = plt.figure()
    ax = fig.gca(projection='3d',autoscale_on=False)
    ax.scatter3D(test_helix[:, 0], test_helix[:, 1], test_helix[:, 2], s=125)
    ax.plot3D(test_helix[:, 0], test_helix[:, 1], test_helix[:, 2])


    # Plot fitted helix
    t = np.linspace(0, z_tot, 100)
    print(z_tot)
    fitted_helix = np.zeros([len(t), 3])
    fitted_helix[:, 0] = radius * np.cos(w*t + phi)
    fitted_helix[:, 1] = radius * np.sin(w*t + phi)
    fitted_helix[:, 2] = t
    fitted_helix = np.dot(fitted_helix, rotation.transpose())
    fitted_helix = fitted_helix -  np.mean(fitted_helix, axis = 0)
    ax.plot3D(fitted_helix[:, 0], fitted_helix[:, 1], fitted_helix[:, 2])



    print("Helix Fit Summary")
    print("-----------------")    
    print("Radius:", radius/100)
    print("Angular Frequency:", w)
    print("Phase Shift:", phi)
    print("RMSE Cylinder:", np.sqrt(sse_cylinder/test_helix.shape[0]))
    print("RMSE Helix:", np.sqrt(sse_helix/test_helix.shape[0]))
    

    # Show projection of helix on identified plane
    point = center
    ax.scatter3D(point[0], point[1], point[2], c="black")
    normal = normal / np.sqrt(np.dot(normal, normal))
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