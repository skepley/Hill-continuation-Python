from hill_model import *
from scipy.interpolate import griddata

def sampler():
    """Sample parameters for the toggle switch other than the hill coefficient. This is a nondimensionalized sampler
     so it assumes that theta_1 = theta_2 = gamma_1 = 1 and returns a vector in R^5 of the form:
     (ell_1, delta_1, gamma_1, ell_2, delta_2).
     Takes a sample anywhere."""

    # pick ell_2, delta_2
    ell_1 = 1.5 * np.random.random_sample()  # sample in (0, 1.5)
    delta_1 = 1.5 * np.random.random_sample()  # sample in (0, 1.5)

    # pick gamma_2
    gammaScale = 1 + 9 * np.random.random_sample()  # gamma scale in [1, 10]
    g = lambda x: x if np.random.randint(2) else 1 / x
    gamma_2 = g(gammaScale)  # sample in (.1, 10) to preserve symmetry between genes

    # pick ell_2, delta_2
    ellByGamma_2 = 1.5 * np.random.random_sample()  # sample in (0, 1.5)
    deltaByGamma_2 = 1.5 * np.random.random_sample()  # sample in (0, 1.5)
    ell_2 = ellByGamma_2 * gamma_2
    delta_2 = deltaByGamma_2 * gamma_2
    return ezcat(ell_1, delta_1, gamma_2, ell_2, delta_2)


def fiber_sampler(u, v, alpha_bar=10):
    """Samples the fiber defined by (u,v) according to the formulas presented in the -temporary- Section 4.4.1"""

    # the first component: u, relates to gamma_2, ell_2, delta_2
    if u < 1:
        ell_2 = np.random.random_sample()
        delta_2 = np.random.random_sample()
        gamma_2 = u / (ell_2 + delta_2)
    elif u < 2:
        ell_2 = np.random.random_sample()
        gamma_2 = ell_2 + np.random.random_sample()
        delta_2 = (gamma_2 - ell_2) / (u - 1)
    else:
        ell_2 = np.random.random_sample()
        delta_2 = np.random.random_sample()
        gamma_2 = (u - 2) * (alpha_bar - 1) + 1

    # the second component: v, relates to ell_1, delta_1
    if v < 1:
        ell_1 = v * np.random.random_sample()  # sample in (0, v)
        delta_1 = v - ell_1
    elif v < 2:
        ell_1 = np.random.random_sample()
        delta_1 = (1 - ell_1) / (v - 1)
    else:
        ell_1 = (v - 2) * (alpha_bar - 1) + 1
        delta_1 = np.random.random_sample()

    return ezcat(ell_1, delta_1, gamma_2, ell_2, delta_2)


def heat_coordinate(alpha, beta, alphaMax):
    """Returns the DSGRN heat map coordinates For a parameter of the form (alpha, beta) where
    alpha = ell / gamma and beta = (ell + delta) / gamma"""

    if beta < 1:  # (ell + delta)/gamma < theta
        x = beta

    elif alpha > 1:  # theta < ell/gamma
        x = 2 + (alpha - 1) / (alphaMax - 1)

    else:  # ell/gamma < theta < (ell + delta)/gamma
        x = 1 + (1 - alpha) / (beta - alpha)
    return x


def heat_coordinates(alpha1, beta1, alpha2, beta2, alphaMax):
    """ take vectors of 4D coordinates and return vectors of x-coordinates and y-coordinates"""
    x = np.array(
        [heat_coordinate(alpha1[j], beta1[j], alphaMax) for j in range(len(alpha1))])
    y = np.array(
        [heat_coordinate(alpha2[j], beta2[j], alphaMax) for j in range(len(alpha2))])
    return x, y


def grid_lines():
    """Add grid lines to a dsgrn coordinate plot"""
    for i in range(1, 3):
        plt.plot([i, i], [0, 3], 'k')
        plt.plot([0, 3], [i, i], 'k')
    return


def dsgrn_plot(parameterArray, alphaMax=None):
    """A scatter plot in DSGRN coordinates of a M-by-5 dimensional array. These are nondimensional parameters with rows
    of the form: (ell_1, delta_1, gamma_2, ell_2, delta_2)."""

    alpha1 = parameterArray[:, 0]
    beta1 = parameterArray[:, 0] + parameterArray[:, 1]
    alpha2 = parameterArray[:, 3] / parameterArray[:, 2]
    beta2 = (parameterArray[:, 3] + parameterArray[:, 4]) / parameterArray[:, 2]

    if alphaMax is None:
        alphaMax = np.max(np.max(alpha1), np.max(alpha2))

    x, y = heat_coordinates(alpha1, beta1, alpha2, beta2, alphaMax)

    plt.scatter(x, y, marker='o', c='k', s=2)
    grid_lines()


def dsgrn_heat_plot(parameterData, colorData, heatMin=1000):
    """Produce a heat map plot of a given choice of toggle switch parameters using the specified color map data.
    ParameterData is a N-by-5 matrix where each row is a parameter of the form (ell_1, delta_1, gamma_2, ell_2, delta_2)"""

    nParameter = len(parameterData)
    alpha_1 = parameterData[:, 0]
    beta_1 = parameterData[:, 1]
    alpha_2 = parameterData[:, 3] / parameterData[:, 2]
    beta_2 = (parameterData[:, 3] + parameterData[:, 4]) / parameterData[:, 2]

    alphaMax = np.maximum(np.max(alpha_1), np.max(alpha_2))

    x = np.array([heat_coordinate(alpha_1[j], beta_1[j], alphaMax) for j in range(nParameter)])
    y = np.array([heat_coordinate(alpha_2[j], beta_2[j], alphaMax) for j in range(nParameter)])
    z = np.array([np.min([val, heatMin]) for val in colorData])

    # define grid.
    plt.figure()
    xGrid = np.linspace(0, 3, 100)
    yGrid = np.linspace(0, 3, 100)
    # grid the data.
    zGrid = griddata((x, y), z, (xGrid[None, :], yGrid[:, None]), method='linear')
    # contour the gridded data, plotting dots at the randomly spaced data points.
    zmin = 0
    zmax = np.nanmax(zGrid)
    palette = plt.matplotlib.colors.LinearSegmentedColormap('jet3', plt.cm.datad['jet'], 2048)
    palette.set_under(alpha=0.0)
    plt.imshow(zGrid, extent=(0, 3, 0, 3), cmap=palette, origin='lower', vmin=zmin, vmax=zmax, aspect='auto',
               interpolation='bilinear')
    # CS = plt.contour(xGrid, yGrid, zGrid, 15, linewidths=0.5, colors='k')
    # CS = plt.contourf(xGrid, yGrid, zGrid, 15, cmap=plt.cm.jet)
    plt.colorbar()  # draw colorbar
    # plot data points.
    plt.scatter(x, y, marker='o', c='k', s=2)
    plt.xlim(0, 3)
    plt.ylim(0, 3)
    # plt.title('griddata test (%d points)' % npts)
    plt.show()
    for i in range(1, 3):
        plt.plot([i, i], [0, 3], 'k')
        plt.plot([0, 3], [i, i], 'k')



number_of_samples = 100
a = np.random.random_sample()