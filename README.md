# Matplotlib 3D streamslice plots

____

This repository contains code to plot streamslice plots in 3D in order
to visualize the flow of a vector field. The vector field must be evaluated
on a grid, and the grid must be uniform in all directions. 

## Usage

The main function is `plot_projections`, which takes the following arguments:

* `grid`: The coordinates of the grid points. This should be a $N \times 3$ array
representing the Cartesian coordinates where the vector field is evaluated.
* `vectorial_quantity`: The vector field evaluated on the grid. This should be a $N \times 3$ array
representing the vector field evaluated on the grid.
* `colormax` (optional) : A tuple of two floats representing the maximum and minimum values of the vector field. 
If not provided, the maximum and minimum values of the vector field are used.
* `ax` (optional) : A matplotlib axis object. If not provided, a new axis is created.
* `contours` (optional) : A boolean indicating whether to plot contours of the magnitude of the vector field.

The function returns the matplotlib axis object.

## Example

```

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
```
 
![Example plot with contour of magnitude](projections_contours.png)



