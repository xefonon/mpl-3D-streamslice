from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)

def plot_contour(u, x, y, ax=None, tex=False, colormax=None, colormap='YlOrRd',
                 zdir='z', offset=0.):
    if tex:
        plt.rc('text', usetex=True)
    # plot quiver
    if ax is None:
        _, ax = plt.subplots()
    xx = np.linspace(np.min(x), np.max(x), 100)
    yy = np.linspace(np.min(y), np.max(y), 100)
    X, Y = np.meshgrid(xx, yy)
    uu_x = griddata((x, y), u[..., 0], (X, Y), method='cubic')
    uu_y = griddata((x, y), u[..., 1], (X, Y), method='cubic')
    uu = np.stack((uu_x, uu_y), axis=-1)
    dx = uu[..., 0]
    dy = uu[..., 1]
    vmin, vmax = colormax
    if zdir == 'x':
        q = ax.contourf(np.hypot(dy, dx), Y, X, zdir=zdir, offset=offset + 0.03, cmap=colormap,
                        alpha=0.5, vmin=vmin, vmax=vmax)
    elif zdir == 'y':
        q = ax.contourf(X, np.hypot(dx, dy), Y, zdir=zdir, offset=offset + 0.03, cmap=colormap,
                        alpha=0.5, vmin=vmin, vmax=vmax)
    elif zdir == 'z':
        q = ax.contourf(X, Y, np.hypot(dx, dy), zdir=zdir, offset=offset + 0.03, cmap=colormap,
                        alpha=0.5, vmin=vmin, vmax=vmax)

    # ax.set_ylim([offset, offset + 0.06])
    ax.set_box_aspect((1, 1, 1))
    return ax, q

def plot_stream(u, x, y, ax=None, tex=False, colormax=None, colormap='YlOrRd'):
    """
    Plot sound field quantities such as velocity or intensity using quiver
    ---------------------------------------------------------------------
    Args:
        P : quantity to plot in meshgrid
        X : X mesh matrix
        Y : Y mesh matrix
    Returns:
        ax : pyplot axes (optionally)

    """
    if tex:
        plt.rc('text', usetex=True)
    # plot quiver
    if ax is None:
        _, ax = plt.subplots()
    xx = np.linspace(np.min(x), np.max(x), 100)
    yy = np.linspace(np.min(y), np.max(y), 100)
    X, Y = np.meshgrid(xx, yy)
    uu_x = griddata((x, y), u[..., 0], (X, Y), method='cubic')
    uu_y = griddata((x, y), u[..., 1], (X, Y), method='cubic')
    uu = np.stack((uu_x, uu_y), axis=-1)
    dx = uu[..., 0]
    dy = uu[..., 1]
    if colormax is None:
        M = np.hypot(dx, dy)
    else:
        M = colormax

    q = ax.streamplot(X, Y, dx, dy, color=M, cmap=colormap, density=.8, arrowstyle='fancy',
                      linewidth=1., broken_streamlines=True)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_xlim([xx.min(), xx.max()])
    ax.set_ylim([yy.min(), yy.max()])

    return ax, q

def plot_stream_3d(lines, ax=None, plane='xy', scale=1.):
    plt.rc('text', usetex=True)

    arrow_size = 7.

    i = 0
    arrow_x = []
    arrow_y = []
    arrow_z = []
    n_arrows = 0
    for line in lines:
        i += 1
        old_x = line.vertices.T[0]
        old_y = line.vertices.T[1]
        # apply for 2d to 3d transformation here
        if plane == 'xy':
            new_z = scale * np.ones_like(old_x)
            new_x = old_x
            new_y = old_y
        elif plane == 'xz':
            new_y = scale * np.ones_like(old_x)
            new_x = old_x
            new_z = old_y
        elif plane == 'yz':
            new_x = scale * np.ones_like(old_x)
            new_y = old_x
            new_z = old_y
        else:
            raise ValueError('plane must be xy, xz or yz')
        ax.plot(new_x, new_y, new_z, color='k', linewidth=1.)
        # plot arrows

        if i % 10 == 1:
            n_arrows += 1
            # make sure new_x, new_y, new_z have been plotted only once
            if (abs(new_x).sum() in arrow_x) and (abs(new_y).sum() in arrow_y) and (abs(new_z).sum() in arrow_z):
                continue
            arrow_vectors = np.column_stack((new_x[0] - new_x[1], new_y[0] - new_y[1], new_z[0] - new_z[1]))
            ax.arrow3D(new_x[0], new_y[0], new_z[0],
                       arrow_vectors[0, 0], arrow_vectors[0, 1], arrow_vectors[0, 2],
                       mutation_scale=arrow_size, color='k', linewidth=.7,
                       arrowstyle="->",
                       )
            arrow_x.append(abs(new_x).sum())
            arrow_y.append(abs(new_y).sum())
            arrow_z.append(abs(new_z).sum())

    return ax

def plot_projections(grid, vectorial_quantity, ax=None, colormax=(None, None),
                     contours = True):
    """
    Plot projections of the vector field Iz onto the cuboid walls.
    -------------------------------------------------------------
    Args:
        grid : Nx3 array representing the cuboid's x, y, and z coordinates
        vectorial_quantity : Nx3 array representing the vector field
        ax : pyplot axes (optionally)
        colormax : tuple of floats representing the min and max of the colormap
        contours : bool, whether to plot contours or not

    Returns:
        ax : pyplot axes (optionally)
    """
    plt.rc('text', usetex=True)
    # make sure font is tex font
    xmin, ymin, zmin = np.min(grid, axis=0)
    xmax, ymax, zmax = np.max(grid, axis=0)

    argminx = np.argwhere(grid[:, 0] == xmin).squeeze(-1)
    argminy = np.argwhere(grid[:, 1] == ymin).squeeze(-1)
    argminz = np.argwhere(grid[:, 2] == zmin).squeeze(-1)

    # Create projections
    xy_projection = {'u': vectorial_quantity[argminz], 'x': grid[argminz, 0], 'y': grid[argminz, 1],
                     'z': zmin * np.ones_like(grid[argminz, 2]), 'plane': 'xy'}
    xz_projection = {'u': vectorial_quantity[argminy], 'x': grid[argminy, 0], 'y': grid[argminy, 2],
                     'z': ymin * np.ones_like(grid[argminy, 1]), 'plane': 'xz'}
    yz_projection = {'u': vectorial_quantity[argminx], 'x': grid[argminx, 1], 'y': grid[argminx, 2],
                     'z': xmin * np.ones_like(grid[argminx, 0]), 'plane': 'yz'}

    # Plot each projection
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    if None in colormax:
        x_proj_mag = np.hypot(vectorial_quantity[:, 1], vectorial_quantity[:, 2])
        y_proj_mag = np.hypot(vectorial_quantity[:, 0], vectorial_quantity[:, 2])
        z_proj_mag = np.hypot(vectorial_quantity[:, 0], vectorial_quantity[:, 1])
        # max
        colormax = np.maximum(x_proj_mag, y_proj_mag, z_proj_mag).max()
        # min
        colormin = np.minimum(x_proj_mag, y_proj_mag, z_proj_mag).min()
        colormax = (colormin, colormax)
    for projection in [xy_projection, xz_projection, yz_projection]:
        # plot stream
        _, q = plot_stream(projection['u'], projection['x'], projection['y'], tex=True)
        ax = plot_stream_3d(q.lines.get_paths(), ax=ax, plane=projection['plane'], scale=projection['z'][0])

        if contours:
            # plot contour
            zdir = [s for s in 'xyz' if s not in projection['plane']][0]
            ax, q = plot_contour(projection['u'], projection['x'], projection['y'], tex=True,
                                 ax=ax, colormap='YlOrRd', zdir=zdir, offset=projection['z'][0] - 0.03,
                                 colormax=colormax)
        else:
            q = None
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    # ax.set_box_aspect((.3, .3, .3))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(30, 45)
    # remove grid
    ax.grid(False)
    # create cuboid around the data and plot it
    cuboid = np.array([[xmin, ymin, zmin],
                       [xmax, ymin, zmin],
                       [xmax, ymax, zmin],
                       [xmin, ymax, zmin],
                       [xmin, ymin, zmin],
                       [xmin, ymin, zmax],
                       [xmax, ymin, zmax],
                       [xmax, ymax, zmax],
                       [xmin, ymax, zmax],
                       [xmin, ymin, zmax],
                       [xmin, ymax, zmax],
                       [xmin, ymax, zmin],
                       [xmax, ymax, zmin],
                       [xmax, ymax, zmax],
                       [xmax, ymin, zmax],
                       [xmax, ymin, zmin]])
    # plot cuboid
    ax.plot(cuboid[:, 0], cuboid[:, 1], cuboid[:, 2], color='darkslategrey', linewidth=1.5,
            linestyle='--')
    # set ax lims to slightly larger than cuboid
    xmaxmin_5pc = 0.05 * (xmax - xmin)
    ymaxmin_5pc = 0.05 * (ymax - ymin)
    zmaxmin_5pc = 0.05 * (zmax - zmin)
    ax.set_xlim([xmin - xmaxmin_5pc, xmax + xmaxmin_5pc])
    ax.set_ylim([ymin - ymaxmin_5pc, ymax + ymaxmin_5pc])
    ax.set_zlim([zmin - zmaxmin_5pc, zmax + zmaxmin_5pc])
    return ax, q

if __name__ == '__main__':
    # example of how to use the function
    # create grid
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    z = np.linspace(0, 1, 10)
    X, Y, Z = np.meshgrid(x, y, z)
    grid = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
    # create vector field
    u = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) * np.ones_like(Z)
    v = np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y) * np.ones_like(Z)
    w = np.zeros_like(Z)

    vectorial_quantity = np.stack((u.ravel(), v.ravel(), w.ravel()), axis=-1)
    # plot projections
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_contours = True
    # plot projections
    ax, q = plot_projections(grid, vectorial_quantity,
                             colormax=(-1, 1), ax=ax,
                             contours = plot_contours)
    if plot_contours:
        # add colorbar
        cbar = fig.colorbar(q, ax=ax, shrink=0.5)
        cbar.ax.set_ylabel('Magnitude')
    fig.show()


