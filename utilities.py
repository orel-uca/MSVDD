__author__ = 'raulpaez' 
__docs__ = 'Utility functions for MSVDD models'

def Kernel(a, b, kernel="linear", sigma=1, degree=2):
    """
    Compute the kernel matrix between two sets of vectors.
    This function calculates various kernel transformations between input matrices,
    commonly used in machine learning algorithms like Support Vector Machines.
    
    Parameters:
    ----------.
        a: (array-like, shape (n_samples_a, n_features)) First input matrix where each row represents a sample.
        b: (array-like, shape (n_samples_b, n_features)) Second input matrix where each row represents a sample.
        kernel: (str) optional (default="linear"). Type of kernel to compute. Options are:
            - "linear": Linear kernel (dot product)
            - "rbf": Radial Basis Function (Gaussian) kernel
            - "polynomial": Polynomial kernel
        sigma: (float) optional (default=1). Bandwidth parameter for the RBF kernel. Controls the influence radius
            of samples. Only used when kernel="rbf".
        degree: (int) optional (default=2). Degree of the polynomial kernel. Only used when kernel="polynomial".
    
    Returns:
    --------
        (ndarray, shape (n_samples_a, n_samples_b)) Kernel matrix representing the similarity between samples in a and b.
    """
    
    import numpy as np
    from scipy.spatial.distance import cdist    
    
    if kernel == "linear":
        return np.dot(a, b.T)
    elif kernel == "rbf":
        sq_dists = cdist(a, b, 'sqeuclidean')
        return np.exp(-sq_dists / sigma)
    elif kernel == "polynomial":
        return (1 + np.dot(a, b.T)) ** degree
    else:
        raise ValueError(f"Kernel '{kernel}' unknown. Valid values are 'linear', 'rbf', 'polynomial'.")

def predict_scores(sol, new_points):
    """
    Predicts the scores of new points with respect to clusters defined by a solution object.
    
    Parameters:
    -----------
        sol: An object containing the solution and parameters of the model and the solution of the model solved.
        new_points: Array of new data points to predict scores for (size: m, d).
    
    Returns:
    --------
        scores: Minimum score for each new point with respect to the clusters (size: m,).
        scores_j: Scores of each new point with respect to each cluster (size: m, p).
        cindex: Index of the cluster with the minimum score for each new point (size: m,).
    """
    
    import numpy as np
    from scipy.spatial.distance import cdist
    
    # Drop spheres with zero radius or with no assigned points.
    x = sol.x
    alpha = sol.alpha  # Size (n, p)
    RR = sol.R  # Size (p,)
    c = sol.c  # Size (p, d)
    p = sol.p
    kernel = sol.kernel
    sigma = sol.sigma
    degree = sol.degree

    if c is None: # Kernelized model does not store centers
        # Kernel matrix between training points
        K_x_x = Kernel(x, x, kernel=kernel, sigma=sigma, degree=degree)  # (n, n)
        
        # Per-cluster quadratic term Cj
        Cj_array = np.array([alpha[:, j].T @ K_x_x @ alpha[:, j] for j in range(p)])  # (p,)

        # Kernel between new points and training points
        K_pt_x = Kernel(new_points, x, kernel=kernel, sigma=sigma, degree=degree)  # (len(new_points), n)

        # For RBF, K[a,a] = 1 so initialize diagonal to ones
        K_pt_pt = np.ones(len(new_points))  # (len(new_points),)

        # Compute scores for new points
        K_pt_x_alpha = K_pt_x @ alpha  # (len(new_points), p)
        scores_j = K_pt_pt[:, None] - 2 * K_pt_x_alpha + Cj_array[None, :] - RR[None, :]  # (len(new_points), p)

        # Obtain best *score* and col index (cluster) corresponding to each row
        scores = np.min(scores_j, axis=1)
        cindex = np.argmin(scores_j, axis=1)
    else:
        valid_centers = []
        valid_radii = []

        # Check for small radii first
        for j in range(len(c)):
            if RR[j] > 0.0001:
                # Calculate the minimum distance from this center to all new_points
                if len(new_points) > 0:
                    distances = np.sqrt(np.sum((new_points - c[j])**2, axis=1))
                    min_distance = np.min(distances) if len(distances) > 0 else float('inf')
                    # Keep this center only if its radius is not smaller than min distance
                    if RR[j] >= min_distance**2 or min_distance == float('inf'):
                        valid_centers.append(c[j])
                        valid_radii.append(RR[j])
                else:
                    # If there are no new_points, just keep centers with radius > threshold
                    valid_centers.append(c[j])
                    valid_radii.append(RR[j])

        # If we have valid centers, use them; otherwise, use all centers
        if valid_centers:
            c = np.array(valid_centers)
            RR = np.array(valid_radii)
        
        # Calculate scores based on Euclidean distance to centers
        scores_j = cdist(new_points, c)**2 - RR
        scores = np.min(scores_j, axis=1)
        cindex = np.argmin(scores_j, axis=1)

    # Return the minimum *score* of each point with respect to each cluster, the score with respect to the j clusters, and the index (cluster j) to which the point belongs
    return scores, scores_j, cindex

def DrawCurves(sol, ax=None):
    """
    DrawCurves(sol, ax=None, show_plot_details=True)
    Visualizes the decision boundaries and classification results of a model based on the provided solution object drom dual MSVDD model version.
    
    Parameters:
    -----------
        sol: An object containing the solution and parameters of the model and the solution of the model solved.
        ax: A matplotlib axes object (optional). If provided, the plot will be drawn on this axes; otherwise, a new figure will be created.
    
    Returns:
    --------
        The function produces a matplotlib visualization and does not return a value.
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    ### Leer variables de sol:
    x = sol.x

    P = range(sol.p)
    n = x.shape[0]
    
    if isinstance(sol.C, np.ndarray) or isinstance(sol.C, list):
        C_str = '[' + ', '.join([f"{ci:.4f}" for ci in sol.C]) + ']'
    else:
        C_str = f"{sol.C:.4f}"

    m = 100  # Higher grid resolution for smoother plot

    # Create grid based on training data bounds
    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, m), np.linspace(y_min, y_max, m))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    scores_grid, scores_j, _ = predict_scores(sol, grid)
    scores_grid = scores_grid.reshape(xx.shape)
    
    # Split train points by score sign
    scores_train, _, _ = predict_scores(sol, x)
    In = np.where(scores_train <= 0.001)[0]
    Out = np.where(scores_train > 0.001)[0]
    
    # Plot
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    
    # Draw decision region first
    contourf = ax.contourf(xx, yy, scores_grid, levels=50, cmap='RdYlBu_r', alpha=0.3)
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('Scores')
    
    # Plot decision contours
    colores = ['chocolate', 'lightgreen', 'darkcyan', 'dimgrey', 'lightsalmon', 'purple', 'orange']
    for j in P:
        if j < len(colores):
            color = colores[j]
        else:
            color = 'black'
        scores_xy = scores_j[:, j].reshape(xx.shape)
        contour_line = ax.contour(xx, yy, scores_xy, levels=[0], colors=color, linewidths=2)
        ax.clabel(contour_line, fmt='%.3f', inline=True, fontsize=8)
    
    # Plot inliers/outliers
    ax.scatter(x[In, 0], x[In, 1], s=4, color='b')  # Training inliers
    ax.scatter(x[Out, 0], x[Out, 1], s=4, color='r')  # Training outliers
    
    # Set explicit limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
    
    ax.set_title(f'MSVDD (train) kernel - V.O.= {sol.obj:.4f}\nk={sol.p:d}, C={C_str}, sigma={sol.sigma:.3f}')
    
    plt.tight_layout()
    
    plt.show()

def DrawBalls(sol, ax=None):
    """
    DrawBalls(sol, ax=None, show_plot_details=True)
    Visualizes clusters and outliers in a 2D space based on the provided solution object from primal MSVDD model version.
    
    Parameters:
    -----------
        sol: An object containing the solution and parameters of the model and the solution of the model solved.
        ax: A matplotlib Axes object to draw the plot on. If None, a new Axes object will be created.
    
    Returns:
    --------
        The function produces a matplotlib visualization and does not return a value.
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Read sol:
    x = sol.x
    c = sol.c
    RR = sol.R

    P = range(sol.p)
    n = x.shape[0]

    R = [np.sqrt(RR[j]) for j in P]
    
    if isinstance(sol.C, np.ndarray) or isinstance(sol.C, list):
        C_str = '[' + ', '.join([f"{ci:.4f}" for ci in sol.C]) + ']'
    else:
        C_str = f"{sol.C:.4f}"
    
    if ax is None:
        _, ax = plt.subplots()
    ax.axis('equal')
    
    for j in P:
        plt.plot(c[j,0], c[j,1], '*', color='c')
        circle = plt.Circle((c[j,0], c[j,1]), R[j], color='b', fill=True, alpha=0.05)
        ax.add_patch(circle)
    
    # Classify by score sign
    scores_train, _, _ = predict_scores(sol, x)
    In= np.where(scores_train <= 0)[0]  # Indices of inliers
    Out= np.where(scores_train > 0)[0]  # Indices of outliers

    # Plot inliers/outliers
    plt.scatter(x[In, 0], x[In, 1], s=4, color='b')  # Training inliers
    plt.scatter(x[Out, 0], x[Out, 1], s=4, color='r')  # Training outliers

    plt.title(f'MSVDD (train) - V.O.= {sol.obj:.4f}\nk={sol.p:d}, C={C_str}')

    plt.show()

def DrawBalls_test(sol, data_test, y_test, ntest, anom_frac):
    """
    Visualize MSVDD test results in 2D by plotting test samples, marking detected outliers,
    and drawing the model's decision spheres.
    The function:
    - Splits test samples by their ground-truth labels (inliers=+1, outliers=-1).
    - Plots inliers (blue) and outliers (red).
    - Predicts outliers using `predict_scores(sol, data_test.T)` and marks detected outliers (black "x").
    - Draws decision spheres for each component j with R[j] > 0, centered at c[j] with radius sqrt(R[j]).
    - Adds a title summarizing k, ntest, anomaly fraction, and counts of detected vs. real outliers.
    
    Parameters:
    -----------
        sol: An object containing the solution and parameters of the model and the solution of the model solved.
        data_test: (ndarray of shape (2, n)) 2D test samples, one sample per column.
        y_test: (array-like of shape (n,)) Ground-truth labels for the test set, expected in {+1 (inlier), -1 (outlier)}.
        ntest: (int) Number of test samples (used for display in the figure title).
        anom_frac: (float) Anomaly fraction (used for display in the figure title).
    
    Returns:
    --------
        The function produces a matplotlib visualization and does not return a value.
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    _, ax = plt.subplots(figsize=(5, 5))
    
    # Split test points by true label
    regulares = data_test[:, y_test == 1]
    outliers = data_test[:, y_test == -1]
    
    # Plot test points
    ax.scatter(regulares[0, :], regulares[1, :], s=20, color='blue', alpha=0.3, label='Regulares')
    ax.scatter(outliers[0, :], outliers[1, :], s=25, color='red', alpha=0.3, label='Outliers')
    
    # Predict outliers with the model
    scores, _, _ = predict_scores(sol, data_test.T)
    y_pred = scores > 0  # True = detected outlier
    
    # Mark detected outliers with black crosses
    detected_outliers = data_test[:, y_pred]
    ax.scatter(detected_outliers[0, :], detected_outliers[1, :], 
                s=20, marker='x', color='black', linewidths=1, 
                label='Outliers detected', zorder=5)
    
    # Draw decision spheres
    for j in range(len(sol.R)):
        if sol.R[j] > 0:  # Only plot spheres with positive radius
            # Sphere center
            ax.plot(sol.c[j, 0], sol.c[j, 1], '*', color='cyan', markersize=6, 
                    markeredgecolor='black', markeredgewidth=0.1)
            # Sphere outline
            circle = plt.Circle((sol.c[j, 0], sol.c[j, 1]), np.sqrt(sol.R[j]), 
                                color='blue', fill=False, linewidth=1, alpha=0.7)
            ax.add_patch(circle)

    ax.axis('equal')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.legend()
    plt.tight_layout()

    # Title with model info
    k = sol.p
    n_detected = y_pred.sum()
    n_real = (y_test == -1).sum()
    plt.title(f'MSVDD (Test) - k={k}, ntest={ntest}, %anom={anom_frac:.2f}\nOut detected: {n_detected}/{len(y_test)} | Out reals: {n_real}/{len(y_test)}')

    plt.show()

def DrawCurves_test(sol, data_test, y_test, ntest, anom_frac, sigma):
    """
    Visualize the MSVDD decision region, boundary, and 2D test samples, highlighting
    true labels and detected outliers.
    This function:
    - Builds a 2D grid over the test data span and uses `predict_scores` to obtain
        anomaly scores on the grid.
    - Draws a filled contour map of the scores and the decision boundary at score = 0.
    - Plots test samples labeled as regular (1) and outlier (-1) in different colors.
    - Marks detected outliers (where score > 0) with black crosses.
    - Overlays the active sphere centers (where corresponding radius R > 0) as cyan stars.
    - Adds a title summarizing model and detection statistics and optionally shows the figure.
    
    Parameters:
    -----------
        sol: An object containing the solution and parameters of the model and the solution of the model solved.
        data_test: (array-like of shape (2, n_test)) 2D test samples arranged as rows [x1; x2]. Only the first two features are visualized.
        y_test: (array-like of shape (n_test,)) Ground-truth labels for test samples. Expected values are 1 for regular and -1 for outlier.
        ntest: (int) Number of test samples (displayed in the plot title).
        anom_frac: (float) Fraction of anomalies in the test set (displayed in the plot title).
        sigma: (float) Kernel width (or related hyperparameter) displayed in the plot title; not used in computation here.
    
    Returns:
    --------
        The function produces a matplotlib visualization and does not return a value.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    
    _, ax = plt.subplots(figsize=(9, 5))
        
    # Split test points by true label
    regulares = data_test[:, y_test == 1]
    outliers = data_test[:, y_test == -1]
    
    # Build grid to visualize decision boundary
    m = 100
    
    x_min, x_max = data_test[0, :].min() - 0.5, data_test[0, :].max() + 0.5
    y_min, y_max = data_test[1, :].min() - 0.5, data_test[1, :].max() + 0.5
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, m), np.linspace(y_min, y_max, m))
    grid = np.c_[xx.ravel(), yy.ravel()]  # (m², 2)
    
    # Score the grid
    scores_grid, _, _ = predict_scores(sol, grid)
    scores_grid = scores_grid.reshape(xx.shape)
        
    # Draw decision region
    contourf = ax.contourf(xx, yy, scores_grid, levels=50, cmap='RdYlBu_r', alpha=0.3)
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('Scores')
    
    # Draw decision boundary (score = 0)
    contour_line = ax.contour(xx, yy, scores_grid, levels=[0], colors='red', linewidths=2, linestyles='--')
    ax.clabel(contour_line, fmt='%.3f', inline=True, fontsize=8)
    
    # Plot test points
    ax.scatter(regulares[0, :], regulares[1, :], s=20, color='blue', alpha=0.3, 
                linewidths=0.5, label='Regulares (test)')
    ax.scatter(outliers[0, :], outliers[1, :], s=25, color='red', alpha=0.3,
                linewidths=0.5, label='Outliers (test)')
    
    # Predict outliers with the model
    scores_test, _, _ = predict_scores(sol, data_test.T)
    y_pred = scores_test > 0  # True = detected outlier
        
    # Mark detected outliers with black crosses
    detected_outliers = data_test[:, y_pred]
    ax.scatter(detected_outliers[0, :], detected_outliers[1, :], 
                s=20, marker='x', color='black', linewidths=0.5, 
                label='Outliers detected', zorder=5)

    # Title with model info
    k = len([r for r in sol.R if r > 0])
    n_detected = y_pred.sum()
    n_real = (y_test == -1).sum()
    plt.title(f'MSVDD (Test) Kernel - k={sol.p}, σ={sigma:.2f}, ntest={ntest}, %anom={anom_frac:.2f}\nDetected: {n_detected}/{len(y_test)} | True outliers: {n_real}/{len(y_test)}')
    
    # Set explicit limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
    
    ax.legend()
    plt.tight_layout()
    
    plt.show()

def plot_cross_val(fname):
    """
    Plot cross-validated MSVDD metrics from a NumPy .npz file and append a summary row to a table file.
    This function:
    - Loads metric arrays from an .npz archive.
    - Draws error-bar curves across the regularization values of C for MSVDD.
    - Sets labels, legend, grid, and a descriptive title based on model/metric encoded in the filename.
    - Saves the figure as an PNG alongside the input file and displays it.
    - Computes, per method, the maximum mean metric and its corresponding std and regularization value, and appends these to a text table.
    
    Parameters:
    -----------
        fname: (str or os.PathLike) Path to the .npz file containing results.
    
    Returns:
    --------
        The function performs plotting and file writing as side effects.
    """
    
    import re, os
    import numpy as np
    import matplotlib.pyplot as plt
    
    foo = np.load(fname)
    maucs = foo['maucs']
    saucs = foo['saucs']
    Cs = foo['Cs']
    anom_frac = foo['outlier_frac']
    ks = foo['ks']
    reps = foo['reps']
    anom_frac = foo['outlier_frac']
    
    DorP = 'Dual' if 'Dual' in fname else 'Prim'  # Extract Dual/Prim from filename
    numbers_in_fname = re.findall(r'_(\d+)', fname)
    ntest = int(numbers_in_fname[-6])
    nrep = int(numbers_in_fname[-4])
    
    plt.figure()
    
    # Colors for different k values
    cols = np.array([[0.77132064, 0.02075195, 0.63364823],
                    [0.74880388, 0.49850701, 0.22479665],
                    [0.19806286, 0.76053071, 0.16911084],
                    [0.08833981, 0.68535982, 0.95339335],
                    [0.00394827, 0.51219226, 0.81262096]])
    
    fmts = ['-x', '--o', '--D', '--s', '--H']
    
    for i in range(maucs.shape[1]):
        plt.errorbar(Cs, maucs[:, i], saucs[:, i]/np.sqrt(reps), fmt=fmts[i], color=cols[i, :], ecolor=cols[i, :], linewidth=2.0, elinewidth=1.0, alpha=0.8)
    plt.xlim((min(Cs)-0.005, max(Cs)+0.005))
    plt.ylim((-0.05, 1.05))
    plt.xticks(Cs, Cs)
    
    plt.grid()

    plt.xlabel('regularization parameter C', fontsize=10)
    
    plt.ylabel('AUC-ROC', fontsize=10)
    
    # Construct and set the title
    title = f'{DorP} - (nTest={ntest}, reps={nrep}, %anom={anom_frac})'
    plt.title(title)
    
    names = ['SVDD']
    for i in range(1, maucs.shape[1]):
        names.append(f'MSVDD ($p$={ks[i]})')
    
    plt.legend(names, loc='lower right', fontsize=10)
    plt.show()
    
    # Get max values of means per column (each row corresponds to a C value) for each metric array, and theirs corresponding std deviations.
    max_maucs = np.max(maucs, axis=0)
    idx_max_maucs = np.argmax(maucs, axis=0)
    max_saucs = np.array([saucs[idx_max_maucs[i], i] for i in range(len(idx_max_maucs))])
    max_Cs = np.array([Cs[idx_max_maucs[i]] for i in range(len(idx_max_maucs))])
    
    # Convert arrays to strings with space-separated values
    vector_str_m = "  ".join([f"{m:5.4f}" for m in max_maucs])
    vector_str_s = "  ".join([f"{s:5.4f}" for s in max_saucs])
    vector_str_nc = "  ".join([f"{nc:5.4f}" for nc in max_Cs])
    
    # Write values to table.txt
    output_dir = os.path.dirname(fname)

    with open(os.path.join(output_dir, 'table.txt'), 'a') as f:
        f.write(f"{ntest}   {anom_frac}    {vector_str_m:{5*max(ks)+1}s}    -    {vector_str_s:{5*max(ks)+1}s}    -    {vector_str_nc:{5*max(ks)+1}s} \n")

def generate_data(datapoints, outlier_frac=0.1, dims=2):
    """
    Generates synthetic data for a model using two Gaussian distributions and noise around the clusters.
    
    Parameters:
    -----------
        datapoints: (int) The total number of data points to generate.
        outlier_frac: (float) optional. The fraction of data points that should be outliers. Default is 0.1.
        dims: (int) optional. The number of dimensions for the data points. Default is 2.
    
    Returns:
    --------
        A tuple containing:
            - X (numpy.ndarray): A 2D array of shape (dims, datapoints) containing the generated data points.
            - y (numpy.ndarray): A 1D array of shape (datapoints,) containing labels for the data points, where 
                regular points are labeled as 1 and outliers as -1.
    """
    
    import numpy as np
    
    X = np.zeros((dims, datapoints))
    y = np.zeros(datapoints)
    
    # Calculate number of points for each distribution
    num_regular = datapoints - int(np.floor(datapoints * outlier_frac))
    num_per_cluster = int(num_regular / 2)
    num_noise = datapoints - 2 * num_per_cluster
    
    # Generate two regular gaussian distributions more separation (-2, -2) and (2, 2) or less separation (-1, -1) and (1, 1)
    # First gaussian cluster - left bottom
    X[:, :num_per_cluster] = 0.5 * np.random.randn(dims, num_per_cluster) + \
                            np.array([-2.0, -2.0]).reshape((2, 1)).dot(np.ones((1, num_per_cluster)))
    y[:num_per_cluster] = 1
    
    # Second gaussian cluster - right top
    X[:, num_per_cluster:2*num_per_cluster] = 0.6 * np.random.randn(dims, num_per_cluster) + \
                                            np.array([2.0, 2.0]).reshape((2, 1)).dot(np.ones((1, num_per_cluster)))
    y[num_per_cluster:2*num_per_cluster] = 1
    
    # Generate noise points avoiding the cluster areas but distributed all around
    noise = np.zeros((dims, num_noise))
    for i in range(num_noise):
        while True:
            # Generate candidate noise point in wider range (-4 to 4)
            point = 8.0 * (np.random.rand(dims) - 0.5)  # Range from -4 to 4 (si -5 a 5, 10.0)
            # Check if point is far enough from both clusters
            dist1 = np.sqrt(np.sum((point - [-2.0, -2.0])**2))
            dist2 = np.sqrt(np.sum((point - [2.0, 2.0])**2))
            # Accept point if outside minimum distance from both clusters
            if dist1 > 1.5 and dist2 > 1.5 and (point[0] - point[1] < 4.0) and (point[0] - point[1] > -4.0): # Si mas cerca de l as esquinas 6.0 y -6.0
                noise[:, i] = point
                break
                
    X[:, 2*num_per_cluster:] = noise
    y[2*num_per_cluster:] = -1
    
    return X, y

def calculate_single_metric(scores, y_true, val_idx, test_idx, current_max_val_score, current_test_score_at_max_val, current_sigma_at_max_val, candidate_sigma):
    """
    Calculates a single performance metric on validation and test sets.
    If the validation score improves, it updates the best validation score,
    the corresponding test score, and the sigma value.

    Parameters:
    -----------
        scores: (np.array) Raw scores from the model for all data points.
        y_true: (np.array) True labels (0 or 1) for all data points.
        val_idx: (np.array) Indices for the validation set.
        test_idx: (np.array) Indices for the test set.
        current_max_val_score: (float) The best validation score found so far for this metric.
        current_test_score_at_max_val: (float) The test score corresponding to current_max_val_score.
        current_sigma_at_max_val: (float) The sigma value corresponding to current_max_val_score.
        candidate_sigma: (float) The current sigma value being evaluated.
    
    Returns:
    --------
        tuple: (updated_max_val_score, updated_test_score_at_max_val, updated_sigma_at_max_val)
    """
    
    from sklearn import metrics
    
    val_metric_value = 0.0
    test_metric_value = 0.0

    # Initialize return values to current bests
    updated_max_val_score = current_max_val_score
    updated_test_score_at_max_val = current_test_score_at_max_val
    updated_sigma_at_max_val = current_sigma_at_max_val

    val_score_source = scores[val_idx]
    test_score_source = scores[test_idx]
    
    fpr_val, tpr_val, _ = metrics.roc_curve(y_true[val_idx], val_score_source, pos_label=1)
    val_metric_value = metrics.auc(fpr_val, tpr_val)
    
    if val_metric_value >= current_max_val_score:
        fpr_test, tpr_test, _ = metrics.roc_curve(y_true[test_idx], test_score_source, pos_label=1)
        test_metric_value = metrics.auc(fpr_test, tpr_test)
        
        updated_max_val_score = val_metric_value
        updated_test_score_at_max_val = test_metric_value
        updated_sigma_at_max_val = candidate_sigma

    return updated_max_val_score, updated_test_score_at_max_val, updated_sigma_at_max_val

def gen_new_data(dir_data, ntrain, nval, ntest, anom_frac, reps):
    """
    Generates new synthetic data for training, validation, and testing, ensuring a specified fraction of anomalies in the validation set.
    
    Parameters:
    -----------
        dir_data (str): The directory path where the generated data and plots will be saved.
        ntrain (int): The number of training samples to generate.
        nval (int): The number of validation samples to generate.
        ntest (int): The number of testing samples to generate.
        anom_frac (float): The fraction of anomalies to include in the dataset.
        reps (iterable): An iterable containing the number of repetitions for data generation.
    
    Returns:
    --------
        The function produces a matplotlib visualization and does not return a value.
    """
    
    import numpy as np
    
    for n in reps:
        train = np.array(range(ntrain), dtype='i')
        val = np.array(range(ntrain, ntrain+nval), dtype='i')
        test = np.array(range(ntrain+nval, ntrain+nval+ntest), dtype='i')
        
        # generate new gaussians.
        data, y = generate_data(ntrain+nval+ntest, outlier_frac=anom_frac)

        inds = np.random.permutation(range(ntrain+nval+ntest))
        data = data[:, inds]
        y = y[inds]
        
        # Ensure at least 10% outliers in validation split so AUC is computable.
        while np.array(y[val]<0., dtype='i').sum() <= int(np.ceil(0.1 * nval)):
            inds = np.random.permutation(range(ntrain+nval+ntest))
            data = data[:, inds]
            y = y[inds]

        # Persist generated splits for downstream models
        np.save(f'{dir_data}/data_train_{ntrain}_{anom_frac}_{n}.npy', data[:, train].copy())
        np.save(f'{dir_data}/clust_train_{ntrain}_{anom_frac}_{n}.npy', y[train].copy())
        np.save(f'{dir_data}/data_val_{nval}_{anom_frac}_{n}.npy', data[:, val].copy())
        np.save(f'{dir_data}/clust_val_{nval}_{anom_frac}_{n}.npy', y[val].copy())
        np.save(f'{dir_data}/data_test_{ntest}_{anom_frac}_{n}.npy', data[:, test].copy())
        np.save(f'{dir_data}/clust_test_{ntest}_{anom_frac}_{n}.npy', y[test].copy())

def evaluate(Cs, sigmas, ks, reps, ntrain, ntest, nval, anom_frac, use_kernel, dir_data, data_DorP, output_DorP):
    """
    Evaluate Multi-Sphere Support Vector Data Description (MSVDD) models with various hyperparameters.
    This function performs a comprehensive evaluation of MSVDD models by iterating through different
    combinations of regularization parameter (C), kernel bandwidth (sigma), and number of spheres (k).
    For each configuration, it trains the model, computes anomaly detection scores, and tracks performance
    metrics using ROC analysis. Results and summary are saved to file.
    
    Parameters:
    -----------
        Cs (list floats): Regularization parameter values to evaluate. Controls the trade-off between training
            error and margin maximization.
        sigmas (list floats): Kernel bandwidth parameter values (used only when use_kernel=True). Controls the
            influence range of each support vector in RBF kernel.
        ks (list ints): Number of spheres to use in the MSVDD model. Each sphere represents a different
            cluster or region in the data.
        reps (list ints): Repetition/trial indices for which to evaluate the models. Allows running multiple
            experiments with different random seeds or data splits.
        ntrain (int): Number of training samples used in the dataset split. This list must have the same length as ntest and nval.
        ntest (int): Number of test samples used in the dataset split. This list must have the same length as ntrain and nval.
        nval (int): Number of validation samples used in the dataset split. This list must have the same length as ntrain and ntest.
        anom_frac (floats): Fraction of anomalies/outliers in the dataset. Used in file naming conventions.
        use_kernel (bool):
            If True, uses the Dualized, kernelized version of MSVDD.
            If False, uses the Primal (linear) version of MSVDD.
        dir_data (str): Directory path where input data files (.npy) are located containing training,
            validation, and test data and their cluster assignments.
        data_DorP (str): Directory path where computed solutions (pickled model objects) are saved.
        output_DorP (str): Directory path where summary output files and logs are written.
    
    Returns:
    --------
        Results are written directly to files:
        - Pickled solution objects saved to data_DorP
        - Summary statistics logged to output_DorP/summary_{DorP}.txt (DorP = 'Dual' or 'Prim' based on use_kernel)
    """
    
    from msvdd_models import Primal_MSVDD, Dualized_MSVDD
    from msvdd_class import Instance
    import numpy as np
    import os, pickle
    
    if use_kernel:
        DorP = 'Dual'
        kernel = 'rbf'
        # kernel = 'polynomial' # If using polynomial kernel, uncomment this line and comment the previous one. Indicate the degree of the polynomial in the Instance constructor below (l. 789).
    else:
        DorP = 'Prim'
        kernel = 'linear'
    
    train = np.array(range(ntrain), dtype='i')
    val = np.array(range(ntrain, ntrain+nval), dtype='i')
        
    for r in reps:
        # Load data
        data_train = np.load(f'{dir_data}/data_train_{ntrain}_{anom_frac}_{r}.npy')
        y_train = np.load(f'{dir_data}/clust_train_{ntrain}_{anom_frac}_{r}.npy')
        data_val = np.load(f'{dir_data}/data_val_{nval}_{anom_frac}_{r}.npy')
        y_val = np.load(f'{dir_data}/clust_val_{nval}_{anom_frac}_{r}.npy')
        data_test = np.load(f'{dir_data}/data_test_{ntest}_{anom_frac}_{r}.npy')
        y_test = np.load(f'{dir_data}/clust_test_{ntest}_{anom_frac}_{r}.npy')

        data = np.concatenate((data_train, data_val, data_test), axis=1)
        y = np.concatenate((y_train, y_val, y_test))
        
        # Relabel outliers as 1 (positive) and normals as 0
        y_true = np.where(y == -1, 1, 0)
        
        for k in ks:
            for C in Cs:
                max_roc = -1.0
                max_val_roc = -1.0
                sigma_roc = -1.0

                for sigma in sigmas:
                    ###############################################################
                    #                       Solve our model                       #
                    ###############################################################
                    instance = Instance(data.T, ntrain, nval, ntest, y, k, C, kernel, sigma, None) # If kernel polynomial, the last None must be the degree of the polynomial, otherwise None.
                    
                    solution_file = f'{data_DorP}/solucion_{ntrain}_{anom_frac}_{r}_{k}_{C}_{sigma}.pkl'
                    
                    if os.path.exists(solution_file):
                        with open(f'{output_DorP}/summary_{DorP}.txt', 'a') as outfile:
                            print(f'{solution_file} - Done', file=outfile)
                        continue
                    else:
                        if not use_kernel:
                            sol = Primal_MSVDD(instance)
                        else:
                            sol = Dualized_MSVDD(instance)
                            
                    # Save the solution in another
                    with open(f'{solution_file}', 'wb') as f:
                        pickle.dump(sol, f)
                    
                    if sol.obj:
                        scores, _, _ = predict_scores(sol, data.T)

                        # Update ROC scores
                        max_val_roc, max_roc, sigma_roc = calculate_single_metric(scores, y_true, val, train, max_val_roc, max_roc, sigma_roc, sigma)
                        
                    ###############################################################
                    #                      Save output data                       #
                    ###############################################################
                    with open(f'{output_DorP}/summary_{DorP}.txt', 'a') as outfile:
                        output_sufix = output_DorP.split('_')[-1]
                        
                        if not sol.obj:
                            status = "             Error"
                            print(f'{DorP}      {output_sufix}   {ntrain:4d}    {anom_frac:4.2f}   {r:3d}  {k:3d}    {C:6.3f}    {sigma:6.3f}  {sol.runtime:7.1f}                              {status}', file=outfile)
                        else:
                            vector_R_str = ", ".join([f"{R_val:9.6f}" for R_val in sol.R])
                            status = ""
                            sigmas_choosen = f"sigmas = {sigma_roc:.2f}   " if use_kernel else " "
                            print(f'{DorP}      {output_sufix}   {ntrain:4d}    {anom_frac:4.2f}   {r:3d}  {k:3d}    {C:6.3f}    {sigma:6.3f}  {sol.runtime:7.1f}    {sol.obj:11.6f}   {sol.gap*100:7.3f}   {int(sol.nodes):8d}   {status}   [{vector_R_str:{10*max(ks)+2}s}]     {max_roc:5.6f}    {sigmas_choosen}', file=outfile)

def metrics(res_filename, Cs, sigmas, ks, reps, ntrain, ntest, nval, anom_frac, use_kernel, show_plots, dir_data, data_DorP, output_DorP):
    """
    Compute and evaluate MSVDD (Multi-Sphere Support Vector Data Description) metrics.
    This function loads pre-computed MSVDD solutions, generates predictions, calculates
    ROC-AUC scores, and generates visualizations. Results are saved to output files and
    aggregated statistics are stored in a compressed NumPy archive.
    
    Parameters:
    -----------
        res_filename (str): Path to the output .npz file where aggregated results will be saved.
        Cs (list floats): List of regularization parameter C values to evaluate.
        sigmas (list floats): List of RBF kernel sigma (bandwidth) values to evaluate.
        ks (list ints): List of numbers of spheres (k) to evaluate.
        reps (list ints): List of repetition/run indices to process.
        ntrain (int): Number of training samples.
        ntest (int): Number of test samples.
        nval (int): Number of validation samples.
        anom_frac (float): Fraction/percentage of anomalies in the dataset.
        use_kernel (bool): Flag to indicate whether to use kernel (Dual) methods or not (Primal).
        show_plots (bool): Flag to indicate whether to display plots.
        dir_data (str): Directory path containing the input data files.
        data_DorP (str): Directory path containing the pre-computed solution (.pkl) files.
        output_DorP (str): Directory path where output summary files will be written.
    
    Returns:
    --------
        Results are written to summary file at '{output_DorP}/summary_{DorP}.txt'.
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    import os, pickle
    
    if use_kernel:
        DorP = 'Dual'
        sig_choos = 'sigma choosen'
    else:
        DorP = 'Prim'
        sig_choos = ''
        
    val = np.array(range(ntrain, ntrain+nval), dtype='i')
    test = np.array(range(ntrain+nval, ntrain+nval+ntest), dtype='i')
    
    rocs = np.zeros((len(reps), len(Cs), len(ks)))
    
    with open(f'{output_DorP}/summary_{DorP}.txt', 'a') as outfile:
        print(f' Model    output    ntrain   %anom   rep   k       C       sigma     CPU          V.O.        GAP      Nodes         Radii {" "*(10*max(ks)+2)}  ROC      {sig_choos}', file=outfile)
                        
    for r in reps:
        # Load data
        data_train = np.load(f'{dir_data}/data_train_{ntrain}_{anom_frac}_{r}.npy')    # Cargo los datos de train en un archivo para cargarlos en nuestro modelo.
        y_train = np.load(f'{dir_data}/clust_train_{ntrain}_{anom_frac}_{r}.npy')    # Cargo la asignacion de cluster de los datos train
        data_val = np.load(f'{dir_data}/data_val_{nval}_{anom_frac}_{r}.npy')    # Cargo los datos de val en un archivo para cargarlos en nuestro modelo.
        y_val = np.load(f'{dir_data}/clust_val_{nval}_{anom_frac}_{r}.npy')    # Cargo la asignacion de cluster de los datos val
        data_test = np.load(f'{dir_data}/data_test_{ntest}_{anom_frac}_{r}.npy')    # Cargo los datos de test en un archivo para cargarlos en nuestro modelo.
        y_test = np.load(f'{dir_data}/clust_test_{ntest}_{anom_frac}_{r}.npy')    # Cargo la asignacion de cluster de los datos test

        data = np.concatenate((data_train, data_val, data_test), axis=1)
        y = np.concatenate((y_train, y_val, y_test))
        
        # Relabel outliers as 1 (positive) and normals as 0
        y_true = np.where(y == -1, 1, 0)
            
        for k in ks:
            for C in Cs:
                max_roc = -1.0
                max_val_roc = -1.0
                sigma_roc = -1.0
                
                for sigma in sigmas:
                    
                    ###############################################################
                    #                          Predictions                        #
                    ###############################################################
                    
                    # Check if solution file exists
                    solution_file = f'{data_DorP}/solucion_{ntrain}_{anom_frac}_{r}_{k}_{C}_{sigma}.pkl'
                    if os.path.exists(solution_file):
                        # Load the results obtained previously and saved in *.pkl files to make the predictions
                        with open(solution_file, 'rb') as f:
                            sol = pickle.load(f)
                    else:
                        # Skip to next iteration if the solution file doesn't exist
                        with open(f'{output_DorP}/summary_{DorP}.txt', 'a') as outfile:
                            print(f'{solution_file} - Not evaluated', file=outfile)
                        continue
                    
                    if sol.obj:
                        scores, _, _ = predict_scores(sol, data.T)

                        # Update ROC scores
                        max_val_roc, max_roc, sigma_roc = calculate_single_metric(scores, y_true, val, test, max_val_roc, max_roc, sigma_roc, sigma)
                        
                    ###############################################################
                    #                      Save output data                       #
                    ###############################################################
                    with open(f'{output_DorP}/summary_{DorP}.txt', 'a') as outfile:
                        output_sufix = output_DorP.split('_')[-1]
                            
                        if not sol.obj:
                            status = "             Error"
                            print(f'{DorP}    {output_sufix:10s}   {ntrain:4d}    {anom_frac:4.2f}   {r:3d}  {k:3d}    {C:6.3f}    {sigma:6.3f}  {sol.runtime:7.1f}                              {status}', file=outfile)
                        else:
                            vector_R_str = ", ".join([f"{R_val:9.6f}" for R_val in sol.R])
                            status = ""
                            sigmas_choosen = f"sigmas = {sigma_roc:.2f}   " if use_kernel else " "
                            print(f'{DorP}    {output_sufix:10s}   {ntrain:4d}    {anom_frac:4.2f}   {r:3d}  {k:3d}    {C:6.3f}    {sigma:6.3f}  {sol.runtime:7.1f}    {sol.obj:11.6f}   {sol.gap*100:7.3f}   {int(sol.nodes):8d}   {status}   [{vector_R_str:{10*max(ks)+2}s}]     {max_roc:5.6f}    {sigmas_choosen}', file=outfile)
                    
                    ###############################################################
                    #                       Draw the results                      #
                    ###############################################################
                                        
                    # Draw the datapoints by colors according to whether they are regular or outliers                    
                    if show_plots: # If there exists a solution, draw our results
                        if sol.obj:
                            reg_plot = [i for i in range(data.shape[1]) if y[i] == 1]
                            out_plot = [i for i in range(data.shape[1]) if y[i] == -1]
                            fig, ax = plt.subplots()
                            ax.scatter(data[0, reg_plot], data[1, reg_plot], s=2, color='cyan', alpha=0.3)
                            ax.scatter(data[0, out_plot], data[1, out_plot], s=2, color='orange', alpha=0.3)
                            
                            if not use_kernel:
                                DrawBalls(sol, ax) # Draw Balls for train data
                                DrawBalls_test(sol, data_test, y_test, ntest, anom_frac) # Draw Balls for test data
                            else:
                                DrawCurves(sol, ax) # Draw Curves for train data
                                DrawCurves_test(sol, data_test, y_test, ntest, anom_frac, sigma) # Draw Curves for test data
                            
                        plt.close(fig)

                rocs[reps.index(r), Cs.index(C), ks.index(k)] = max_roc

    # AUC-ROC
    mrocsRep = np.mean(rocs, axis=0) # means of metrics for each (C, k) over all instances (reps)
    srocsRep = np.std(rocs, axis=0) # stds of metrics for each (C, k) over all instances (reps)
    
    # save results
    np.savez(res_filename, maucs=mrocsRep, saucs=srocsRep, ntrain=ntrain, ntest=ntest, nval=nval, 
            outlier_frac=anom_frac, reps=len(reps), Cs=Cs, ks=ks, sigmas=sigmas)
