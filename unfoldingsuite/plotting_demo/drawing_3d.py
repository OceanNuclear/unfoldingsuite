import numpy as np
from numpy import pi; tau = pi*2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from numpy import array as ary
from numpy import sqrt

# logic for converting between 3d pose representations.
def cartesian_spherical(x, y, z):
    """
    convert a cartesian unit vector into theta-phi representation
    """
    x,y,z = ary(np.clip([x,y,z],-1,1), dtype=float) #change the data type to the desired format
    Theta = np.arccos(z)
    Phi = np.arctan(np.divide(y,x))    #This division is going to give nan if (x,y,z) = (0,0,1)
    Phi = np.nan_to_num(Phi)    #Therefore assert phi = 0 if (x,y) = (0,0)
    Phi+= ary( (np.sign(x)-1), dtype=bool)*pi #if x is positive, then phi is on the RHS of the circle; vice versa.
    return ary([Theta, Phi])

def QuatToR(q):
    theta = 2 * np.arccos(np.clip(q[0],-1,1))

    R = np.identity(3)
    # if theta>2E-5 or abs(theta-pi)>2E-5: # theta_prime not approaching 0 or pi
    R[0][0] -= 2*( q[2]**2  +q[3]**2  )
    R[1][1] -= 2*( q[1]**2  +q[3]**2  )
    R[2][2] -= 2*( q[1]**2  +q[2]**2  )
    R[0][1] -= 2*( q[0]*q[3]-q[1]*q[2])
    R[0][2] -= 2*(-q[0]*q[2]-q[1]*q[3])
    R[1][0] -= 2*(-q[0]*q[3]-q[1]*q[2])
    R[1][2] -= 2*( q[0]*q[1]-q[2]*q[3])
    R[2][0] -= 2*( q[0]*q[2]-q[1]*q[3])
    R[2][1] -= 2*(-q[0]*q[1]-q[2]*q[3])

    return R

# basic functions
def unit_vec(vec):
    return vec/np.linalg.norm(vec)

def get_singular_dir(R):
    """
    Given a 2x3 response matrix (consisting of two non-zero, non-degenerate responses),
    when measuring a 3-bin spectrum, there will always exist at least one singular direction,
    so that any changes in that direction is not detected by the system possessing this response matrix.

    Parameters
    ----------
    R: response matrix, shape must be 2x3

    Output
    ------
    unit vector pointing in the singular direction.
    """
    assert R.shape==(2, 3), "This program can only be used to analyze response matrix of shape (2, 3)."
    return unit_vec(np.cross(R[0], R[1]))

# determining the radius of the chi2 ellipse at that angle
def rotate_around_axis(axis, covar, R, num_points=120, initial_dir = [1,0,0]):
    """
    Draw out the chi^2 ellipse by rotating around a user-specified axis.
    Specifically generated for the case of reponse matrix with dimension m=2, n=3.

    Parameters
    ----------
    axis:   unnormalized axis that the ellipse will be perpendicular to.
    R:      response matrix
    num_points: resolution of the ellipse.

    Output
    ------
    an array of points 3D points, shape = (num_points, 3)
    """
    unit_axis = unit_vec(axis)
    initial_dir -= (unit_axis @ initial_dir) * unit_axis
    list_of_points = []
    for theta in np.linspace(0, tau, num_points):
        Rot = QuatToR([np.cos(theta/2), *(unit_axis * np.sin(theta/2))])
        vector = how_far_for_1_chi2(Rot @ initial_dir, covar, R)
        list_of_points.append(vector)
    return ary(list_of_points)

def how_far_for_1_chi2(direction, covar, R):
    """
    How far to walk in this direction in order to get one chi^2 deviation
    Parameters
    ----------
    direction: direction to walk
    covar: the covariance matrix
    R: the response matrix

    Output
    ------
    distance in phi space to walk (starting at the chi^2=0 line)
        in the direction specified such that we reach chi^2=1.
    """
    v = ary(direction)
    inv_n_squared = ((R@v) @ np.linalg.inv(covar) @ (R@v)) # should be a scalar
    n = 1/sqrt(inv_n_squared)
    return n*v

def parametric_within_bounds(anchor, unit_dir, lower_bounds, upper_bounds):
    r"""
    get the coordinates where a parametric line goes into and out-of a cube
    Parameters
    ----------
    anchor:     a point on the line
    unit_dir:   direction where the line points
    lower_bounds: left-bottom-closest vertex of the cube
    lower_bounds: right-top-furthest vertex of the cube

    Output
    ------
    (when travelling in the direction of the unit_dir vector,)
    min_lambda_point: the point where the line enters the box
    max_lambda_point: the point where the line enters the box

    The maths
    ---------
    [l] = vector denoting line,
    [a] = anchor vector
    [d] = unit vector pointing // to the line
    l = parametric variable
    equation: 
        [l]:[a]+ \lambda*[d]
    
    x_min <= l[0] <= x_max
    y_min <= l[1] <= y_max
    z_min <= l[2] <= z_max
    """
    lambda_min, lambda_max = [], []
    for basis in range(3):
        if np.sign(unit_dir[basis])>0:
            lambda_min.append((lower_bounds[basis]-anchor[basis])/unit_dir[basis])
            lambda_max.append((upper_bounds[basis]-anchor[basis])/unit_dir[basis])
        elif np.sign(unit_dir[basis])<0:
            lambda_max.append((lower_bounds[basis]-anchor[basis])/unit_dir[basis])
            lambda_min.append((upper_bounds[basis]-anchor[basis])/unit_dir[basis])

    min_lambda_point = anchor + max(lambda_min) * unit_dir
    max_lambda_point = anchor + min(lambda_max) * unit_dir
    return min_lambda_point, max_lambda_point

def interpolate(lower_limit, upper_limit, fraction):
    """
    find the location of the interpolated value within a range,
    when the lowest and highest values of that range is known.
    Parameters
    ----------
    lower_limit:array of the lowest values of that range
    upper_limit:array of the highest values of that range, same shape as above.
    fraction:   the location where I want my outputted interpolated value to be at.
    """
    full_range = upper_limit - lower_limit
    return lower_limit + full_range * fraction

def plot_chi2_line(ax, R, phi_true, ellipse_heights=[], chi2_mark=[1], covar_N=None):
    """
    Plot the chi^2=0 line within the viewing cube,
    And draw chi^2 = chi2_mark lines around it.

    Parameters
    ----------
    ax: axes on which the line is plotted
    R:  response matrix.
    phi_true:   true spectrum
    ellipse_heights:    the list of heights where the ellipses needs to be drawn.
                        This is expressed in terms of fraction of height of the chi^2 = 0 line.
    chi2_mark: the list of chi^2 values where ellipses need to be drawn.
    covar_N:    The covariance matrix for the measured reaction rates..
                Not needed if ellipse_heights = empty list or chi2_mark = empty list.

    Output
    ------
    ax: the originally inputted axes.
    """
    lims = ary(ax.get_w_lims()) # save the current limits before doing anything else to the plot
    lower_bounds, upper_bounds = lims[::2], lims[1::2]

    singular_dir = get_singular_dir(R)
    min_lambda_point, max_lambda_point = parametric_within_bounds(phi_true, unit_vec(singular_dir), lower_bounds, upper_bounds)
    ax.plot(*ary([min_lambda_point, max_lambda_point]).T, label='chi^2=0 line')
    # set the limits back to the original, before the line was plotted.
    ax.set_xlim3d(lims[0], lims[1]); ax.set_ylim3d(lims[2], lims[3]); ax.set_zlim3d(lims[4], lims[5])
    for chi2_ind, chi2_value in enumerate(chi2_mark):
        for ellipse_ind, height in enumerate(ellipse_heights):
            assert 0<height<1, "Parameter 'ellipse_height' is used to define the fraction of length (within the visualized bounds) along which the chi^2=chi2_mark should be drawn, and should not be out of bounds."
            center = interpolate(min_lambda_point, max_lambda_point, height)
            equi_chi_ellipse = rotate_around_axis(singular_dir, covar_N, R, num_points=120)
            if ellipse_ind==0:
                ax.plot(*(equi_chi_ellipse*sqrt(chi2_value) + center).T, color="C"+str(chi2_ind+2), label='chi^2='+str(chi2_value)+' cylinder')
            else:
                ax.plot(*(equi_chi_ellipse*sqrt(chi2_value) + center).T, color="C"+str(chi2_ind+2))
    return ax

# generating a graph 
def prepare_graph(ap_list, phi_true):
    """
    Create a 3d plot of the apriori points and the solution phi points.
    Parameters
    ----------
    ap_list: list of a priori spectra (2D array)
    phi_true: the true spectrum (1D array)

    Output
    ------
    fig: matplotlib figure
    ax: 3d axes on which the data points are scattered
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*ary(ap_list).T, label='a priori')
    ax.scatter(*phi_true, color='black', marker='x', label='true spectrum')
    return fig, ax