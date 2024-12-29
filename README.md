# spherical-fractional-brownian-surface

Python implementation for fast generation of spherical fractional Brownian surfaces with given Hurst exponent in
equirectangular projection (each pixel spans an equal number of degrees in longitude
and latitude) but also planar fractional Brownian surfaces. The surfaces admit an exact height-height correlation function projected along a direction, but is not isotropic.

## Theory

The surfaces are generated using the [turning bands method](https://doi.org/10.1029/WR018i005p01379). Instead of simulating the two-dimensional surface directly, multiple one-dimensional fractional Brownian motions (fBm) are simulated along several lines at different orientations. The resulting field is obtained by the sum of the values of the one-dimensional fBms along the respective positions along each line. 

In the planar case, the lines are spaced evenly on the unit circle (for discussion on the benefits over randomly spaced lines, see the original paper) and for the spherical case, the lines are oriented randomly on the unit sphere.

While this doesn't mathematically result in fractional Brownian surfaces (isotropy isn't satisfied), the limit case of high number of lines does. In practice having in the order of 50+ lines is sufficient. The advantage of the approach over other exact methods (e.g. Stein's method) is its speed and memory usage, mainly due to only needing to simulate one-dimensional processes.

## Implementation

The fractional Brownian motions are generated using Hoskin's algorithm via the [stochastic](https://pypi.org/project/stochastic/) package and the code is sped up with [numba](https://pypi.org/project/numba/).

The time complexity for the algorithm is $\mathcal{O}(n^2 * \texttt{num\_components} / \texttt{n\_threads} + n^2 * \texttt{num\_components} * \texttt{fbm\_interpolation\_coef})$ and the memory footprint is $\mathcal{O}(n^2 * \texttt{n\_threads} + n * \texttt{fbm\_interpolation\_coef} * \texttt{num\_components})$, where $n=\mathrm{max}(n_x, n_y)$ is the maximal number of simulated pixels along a direction, $\texttt{num\_components}$ is the number of simulated lines, $\texttt{fbm\_interpolation\_coef}$ is how many times the one-dimensional processes are upsampled compared to the pixel sizes of the two-dimensional surface and $\texttt{n\_threads}$ is the number of threads used for parallelization (only on CPU using `multiprocessing`).


Usage
--------
```py
import fbs

plane = fbs.PlanarFractionalBrownianSurface(a_x=1, a_y=1, H=0.5, num_components=50, seed=7)
z_planar = plane.evaluate_grid(n_x=1600, n_y=900)
z_planar_points = plane.evaluate_points(np.array([0.2, 0.5]), np.array([0.5, 0.5]))

sphere = fbs.SphericalFractionalBrownianSurface(n_fbm=2048*64, H=0.7, num_components=50, seed=13)
z_spherical = sphere.evaluate_equilateral(n_x=1024, n_threads=2)
```

Example results
--------
![](notebooks/example_sfbs_proj_ortho.png)

![](notebooks/example_pfbs.png)