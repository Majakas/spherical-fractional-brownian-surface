import numpy as np
import time
from tqdm import tqdm
import numba

from stochastic.processes.continuous import FractionalBrownianMotion


def downsample_height_map(z, k):
    """Downsamples the height map by averaging blocks of size k x k. k needs to be
    divisible by the height and width of the height map."""
    return z.reshape(z.shape[0] // k, k, z.shape[1] // k, k).mean(axis=(1, 3))


@numba.jit(nopython=True)
def meshgrid(x, y):
    """
    np.meshgrid function that works with numba.
    """
    xx = np.empty(shape=(y.size, x.size), dtype=x.dtype)
    yy = np.empty(shape=(y.size, x.size), dtype=y.dtype)
    for i in range(y.size):
        for j in range(x.size):
            xx[i,j] = x[j]
            yy[i,j] = y[i]
    return xx, yy


@numba.jit
def compute_sfBs_component_2d(fBm_sample, vec_pole, lon, lat):
    """
    Returns the values of the sfBs component with a pole pointing towards vec_pole
    (a 3D unit vector) on a grid that spans the latitudes and longitudes lon and lat
    (both 1D np.arrays). fBm_sample is the sample of the fractional Brownian motion
    and is meant to span from -pi/2 to pi/2 in latitude.
    """
    # 2D grid of the latitudes and longitudes of the scanned area
    grid_lon, grid_lat_pole_frame = meshgrid(np.deg2rad(lon), np.deg2rad(lat))
    # Flattening and converting the latitudes and longitudes to 3D unit vectors
    grid_unit_vector = np.stack((grid_lon.flatten(), grid_lat_pole_frame.flatten()), axis=1)
    grid_unit_vector = np.stack((
        np.cos(grid_unit_vector[:, 1]) * np.cos(grid_unit_vector[:, 0]), # x
        np.cos(grid_unit_vector[:, 1]) * np.sin(grid_unit_vector[:, 0]), # y
        np.sin(grid_unit_vector[:, 1]) # z
    ), axis=1)

    # Calculating the latitude of the grid in the frame of the pole
    z_pole_frame = np.sum(vec_pole * grid_unit_vector, axis=1)
    grid_lat_pole_frame = np.reshape(np.arcsin(z_pole_frame), grid_lat_pole_frame.shape)
    # The pole frame latitude determines the value of the fBm
    fBm_index = np.minimum(((1/2 + grid_lat_pole_frame/np.pi)*len(fBm_sample)).astype(np.int64), len(fBm_sample)-1)
    return np.take(fBm_sample, fBm_index)


@numba.jit
def compute_sfBs_component_1d(fBm_sample, vec_pole, lon, lat):
    """
    Returns the values of the sfBs component with a pole pointing towards vec_pole
    (a 3D unit vector) on points with longitudes and latitudes lon and lat
    (both 1D np.arrays). fBm_sample is the sample of the fractional Brownian motion
    and is meant to span from -pi/2 to pi/2 in latitude.
    """
    unit_vector = np.stack((lon, lat), axis=1)
    unit_vector = np.stack((
        np.cos(unit_vector[:, 1]) * np.cos(unit_vector[:, 0]), # x
        np.cos(unit_vector[:, 1]) * np.sin(unit_vector[:, 0]), # y
        np.sin(unit_vector[:, 1]) # z
    ), axis=1)

    # Calculating the latitude of the grid in the frame of the pole
    z_pole_frame = np.sum(vec_pole * unit_vector, axis=1)
    lat_pole_frame = np.arcsin(z_pole_frame)
    # The pole frame latitude determines the value of the fBm
    fBm_index = np.minimum(((1/2 + lat_pole_frame/np.pi)*len(fBm_sample)).astype(np.int64), len(fBm_sample)-1)
    return np.take(fBm_sample, fBm_index)


def _generate_sfbs_internal_equilateral(fBm_samples, vec_poles, n_x, upsampling_coef, SCAN_SIZE, verbose=True):
    """
    Internal loop for calculating the sfBs in equilateral coordinates.
    The function is split off from the main function to allow for parallelization.
    """
    def ceil_divide(a, b):
        return -(-a // b)
    
    z = np.zeros((n_x//2, n_x))

    # longitudes and latitudes of the grid (1D arrays)
    lon = np.arange(1, 2*n_x*upsampling_coef, 2)*180/n_x/upsampling_coef # within [0, 360)
    lat = np.arange(1, n_x*upsampling_coef, 2)*180/n_x/upsampling_coef - 90 # Within [-90, 90]
    
    # The grid is first generated at upsampling_coef times higher resolution and
    # then downsampled to get the final grid. The following loop goes through the 
    # grid in chunks of size SCAN_SIZE x SCAN_SIZE.
    n_steps_x = ceil_divide(n_x*upsampling_coef, SCAN_SIZE)
    n_steps_y = ceil_divide(n_x*upsampling_coef//2, SCAN_SIZE)
    n_steps_tot = len(fBm_samples) * n_steps_x * n_steps_y

    for i in tqdm(range(n_steps_tot), disable=not verbose):
        i_step_y = i // n_steps_x // len(fBm_samples)
        i_step_x = i // len(fBm_samples) % n_steps_x
        i_fBm = i % len(fBm_samples)
        iy_min = i_step_y*SCAN_SIZE
        iy_max = min((i_step_y+1)*SCAN_SIZE, n_x*upsampling_coef//2)
        ix_min = i_step_x*SCAN_SIZE
        ix_max = min((i_step_x+1)*SCAN_SIZE, n_x*upsampling_coef)

        delta_z = compute_sfBs_component_2d(
            fBm_samples[i_fBm], vec_poles[i_fBm], lon[ix_min:ix_max], lat[iy_min:iy_max]
        )
        if upsampling_coef > 1:
            delta_z = downsample_height_map(delta_z, upsampling_coef)
        z[iy_min//upsampling_coef:iy_max//upsampling_coef, ix_min//upsampling_coef:ix_max//upsampling_coef] +=\
            delta_z
    return z


def _generate_sfbs_internal_points(fBm_samples, vec_poles, lon, lat, SCAN_SIZE, verbose=True):
    """
    Internal loop for calculating the sfBs on the specified points.
    The function is split off from the main function to allow for parallelization.
    """
    def ceil_divide(a, b):
        return -(-a // b)

    z = np.zeros((len(lon)))

    t0 = time.time()
    n_steps = ceil_divide(len(lon), SCAN_SIZE)
    n_steps_tot = len(fBm_samples) * n_steps

    for i in tqdm(range(n_steps_tot), disable=not verbose):
        i_step = i // len(fBm_samples)
        i_fBm = i % len(fBm_samples)
        i_min = i_step*SCAN_SIZE
        i_max = min((i_step+1)*SCAN_SIZE, len(lon))

        delta_z = compute_sfBs_component_1d(
            fBm_samples[i_fBm], vec_poles[i_fBm], lon[i_min:i_max], lat[i_min:i_max]
        )
        z[i_min:i_max] += delta_z
    return z


class SphericalFractionalBrownianSurface:
    """
    Spherical Fractional Brownian Surface. It spans longitudes 0 to 2*pi and latitudes -pi/2 to pi/2.

    The surface is generated by summing num_components 1D fractional Brownian motions projected
    onto a sphere with random orientations. The resulting surface admits a height-height correlation
    function that converges to the true curve with increasing num_components. Regardless of the
    value of num_components, the fractal dimension and the Hurst exponent of the surface are exact
    and related by D = 2 - H.
    """
    def __init__(self, n_fbm, H=0.5, num_components=10, seed=42, verbose=True):
        """
        Parameters
        ----------
        n_fbm : int
            The number of times a 1D fractional Brownian motion is sampled in order to calculate
            the resulting surface. While the fBm is generated using an exact algorithm (Hoskin's method),
            in order to get the value at a fractional latitude, the fBm is sampled at a
            higher resolution and then interpolated to get the value at the fractional latitude.
            Typically on can aim to have n_fbm be 64 or more times the desired resolution.
        H : float
            The Hurst exponent of the fractional Brownian motion. Must be in the range (0, 1).
            The bigger H is, the smoother the surface. H=0.5 is the fastest to compute.
        num_components : int
            The number of 1D fractional Brownian motions to sum over to generate the surface.
            The bigger num_components is, the more isotropic the surface.
        seed : int
            The seed for the random number generator.
        """
        self.H = H
        self.num_components = num_components
        self.rng = np.random.default_rng(seed)

        def generate_random_unit_vector(rng):
            # Generates a random unit vector in Cartesian coordinates
            lon = rng.uniform(-np.pi, np.pi)
            lat = np.arcsin(rng.uniform(-1, 1))
            return np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])


        t0 = time.time()

        fBm = FractionalBrownianMotion(hurst=H, t=n_fbm, rng=self.rng)
        self.fBm_samples = np.array([fBm.sample(n_fbm) for _ in range(num_components)])
        if verbose: print(f"1D fBms generated in {time.time()-t0:.3f} seconds")
        # unit vectors pointing towards the poles of the different components
        self.vec_poles = np.array([generate_random_unit_vector(self.rng) for _ in range(num_components)])

    def evaluate_equilateral(self, n_x, upsampling_coef=1, n_threads=1, verbose=True):
        """
        Returns the surface in equirectangular projection such that each pixel spans an equal
        number of degrees in longitude and latitude (n_y = n_x // 2).

        Time complexity is O(n_x^2 * upsampling_coef^2 * num_components / n_threads).
        Memory complexity is O(n_x^2 * num_components / n_threads).

        Parameters
        ----------
        n_x : int
            The number of pixels in the x direction.
        upsampling_coef : int
            With upsampling_coef=1, the value of a pixel represents the value of the midpoint
            of the pixel (this is what one normaly wants). With upsampling_coef > 1 the pixel
            is generated at a upsampling_coef times higher resolution and then averaged to get
            the final value of the pixel. This can be useful in the niche case when one wants
            the pixels to more closely represents the mean height of a pixel.
        n_threads : int
            The number of threads to use for parallelization. If n_threads > 1, the function
            will use multiprocessing to parallelize the computation of the surface.
        """
        n_y = n_x//2
        z = np.zeros((n_y, n_x))

        # The grid is calculated in chunks of size SCAN_SIZE x SCAN_SIZE in order to reduce memory usage
        SCAN_SIZE = 1024 

        t0 = time.time()
        if n_threads > 1:
            from multiprocessing import Pool
            if verbose: print(f"Using {n_threads} processes")

            fBm_samples = np.array_split(self.fBm_samples, n_threads)
            vec_poles = np.array_split(self.vec_poles, n_threads)
            with Pool(n_threads) as p:
                z = p.starmap(_generate_sfbs_internal_equilateral, [(fBm_samples[i], vec_poles[i], n_x, upsampling_coef, SCAN_SIZE, verbose) for i in range(n_threads)])
                z = np.sum(np.stack(z, axis=2), axis=2)
        else:
            z = _generate_sfbs_internal_equilateral(self.fBm_samples, self.vec_poles, n_x, upsampling_coef, SCAN_SIZE, verbose=verbose)

        if verbose: print(f"spherical fractional Brownian surface calculated in {time.time()-t0:.3f} seconds")
        return z
    
    def evaluate_points(self, lon, lat, n_threads=1, verbose=True):
        """
        Returns the heights of the surface at the given points.

        Time complexity is O(len(lon) * num_components / n_threads).
        Memory complexity is O(len(lon) * num_components / n_threads).

        Parameters
        ----------
        lon : np.array
            The latitudes of the points, within [0, 2*pi).
        lat : np.array
            The longitudes of the points, within [-pi/2, pi/2].
        """
        t0 = time.time()

        # The grid is calculated in chunks of size SCAN_SIZE x SCAN_SIZE in order to reduce memory usage
        SCAN_SIZE = 1024*1024 

        if n_threads > 1:
            from multiprocessing import Pool
            if verbose: print(f"Using {n_threads} processes")

            fBm_samples = np.array_split(self.fBm_samples, n_threads)
            vec_poles = np.array_split(self.vec_poles, n_threads)
            with Pool(n_threads) as p:
                z = p.starmap(_generate_sfbs_internal_points, [(fBm_samples[i], vec_poles[i], lon, lat, SCAN_SIZE, verbose) for i in range(n_threads)])
                z = np.sum(np.stack(z, axis=1), axis=1)
        else:
            z = _generate_sfbs_internal_points(self.fBm_samples, self.vec_poles, lon, lat, SCAN_SIZE, verbose=verbose)

        if verbose: print(f"spherical fractional Brownian surface point heights calculated in {time.time()-t0:.3f} seconds")
        return z


def _generate_pfbs_internal(fBm_samples, n_x, n_y, a_x, a_y, upsampling_coef, n_fbm_density, SCAN_SIZE, fBm_directions, verbose=True):
    """
    Internal loop for calculating the pfBs. The function is split off from the main function
    to allow for parallelization.
    """
    def ceil_divide(a, b):
        return -(-a // b)
    
    x = np.arange(1, 2*n_x*upsampling_coef, 2)/2/n_x/upsampling_coef*a_x # within [0, a_x)
    y = np.arange(1, 2*n_y*upsampling_coef, 2)/2/n_y/upsampling_coef*a_y # within [0, a_y)

    x0, y0 = np.meshgrid(x[:SCAN_SIZE], y[:SCAN_SIZE])
    fBm_kernels = np.stack([x0*fBm_directions[0, i] + y0*fBm_directions[1, i] for i in range(len(fBm_samples))], axis=2)

    z = np.zeros((n_y, n_x))
    n_steps_x = ceil_divide(n_x*upsampling_coef, SCAN_SIZE)
    n_steps_y = ceil_divide(n_y*upsampling_coef, SCAN_SIZE)
    n_steps_tot = n_steps_x * n_steps_y * len(fBm_samples)

    for i in tqdm(range(n_steps_tot), disable=not verbose):
        i_step_y = i // n_steps_x // len(fBm_samples)
        i_step_x = i // len(fBm_samples) % n_steps_x
        i_fBm = i % len(fBm_samples)
        iy_min = i_step_y*SCAN_SIZE
        iy_max = min((i_step_y+1)*SCAN_SIZE, n_y*upsampling_coef)
        ix_min = i_step_x*SCAN_SIZE
        ix_max = min((i_step_x+1)*SCAN_SIZE, n_x*upsampling_coef)
        y_min = iy_min*a_y/n_y/upsampling_coef
        x_min = ix_min*a_x/n_x/upsampling_coef

        fBm_index = ((fBm_kernels[:iy_max-iy_min,:ix_max-ix_min, i_fBm] + x_min*fBm_directions[0, i_fBm] + y_min*fBm_directions[1, i_fBm]) * n_fbm_density).astype(np.int64)
        fBm_index += len(fBm_samples[i_fBm])//2 # To avoid negative indices
        delta_z = np.take(fBm_samples[i_fBm], fBm_index)
        if upsampling_coef > 1:
            delta_z = downsample_height_map(delta_z, upsampling_coef)

        z[iy_min//upsampling_coef:iy_max//upsampling_coef, ix_min//upsampling_coef:ix_max//upsampling_coef] +=\
            delta_z
    return z


def _generate_pfbs_internal_points(fBm_samples, fBm_directions, x, y, n_fbm_density, SCAN_SIZE, verbose=True):
    """
    Internal loop for calculating the pfBs. The function is split off from the main function
    to allow for parallelization.
    """
    def ceil_divide(a, b):
        return -(-a // b)
    
    z = np.zeros((len(x)))

    t0 = time.time()
    n_steps = ceil_divide(len(x), SCAN_SIZE)
    n_steps_tot = len(fBm_samples) * n_steps

    for i in tqdm(range(n_steps_tot), disable=not verbose):
        i_step = i // len(fBm_samples)
        i_fBm = i % len(fBm_samples)
        i_min = i_step*SCAN_SIZE
        i_max = min((i_step+1)*SCAN_SIZE, len(x))

        fBm_index = ((fBm_directions[0, i_fBm] * x[i_min:i_max] + fBm_directions[1, i_fBm] * y[i_min:i_max]) * n_fbm_density).astype(np.int64)
        fBm_index += len(fBm_samples[i_fBm])//2 # To avoid negative indices
        delta_z = np.take(fBm_samples[i_fBm], fBm_index)
        z[i_min:i_max] += delta_z
    return z


class PlanarFractionalBrownianSurface:
    """
    Planar Fractional Brownian Surface. It spans from 0 to a_x in x, and 0 to a_y in y.
    
    The surface is generated by summing num_components 1D fractional Brownian motions angled at
    different directions. The fBms are angled at equal intervals between 0 and pi. If the direction
    vector of the i-th fBm is \vec n_i, then the elevation of the surface has the form
    z = \sum_i fBm_i(\vec n_i \cdot \vec x), where \vec x is the 2D vector of the coordinates of the surface.
    """
    def __init__(self, a_x, a_y, n_fbm_density=64, H=0.5, num_components=10, seed=42, verbose=True, randomize_directions=False):
        """
        Parameters
        ----------
        a_x: float
            The width of the surface. The surface spans from 0 to a_x in the x direction.
        a_y: float
            The height of the surface. The surface spans from 0 to a_y in the y direction.
        n_fbm_density : int
            The number of times a 1D fractional Brownian motion is sampled per unit length
            along the surface. While the fBm is generated using an exact algorithm (Hoskin's method),
            in order to get the value at a fractional position, the fBm is sampled at a
            higher resolution and then interpolated to get the value at the position.
            Typically on can aim to have n_fbm_density be 64 or more times the desired resolution.
        H : float
            The Hurst exponent of the fractional Brownian motion. Must be in the range (0, 1).
            The bigger H is, the smoother the surface. H=0.5 is the fastest to compute.
        num_components : int
            The number of 1D fractional Brownian motions to sum over to generate the surface.
            The bigger num_components is, the more isotropic the surface.
        seed : int
            The seed for the random number generator.
        randomize_directions : bool
            If True, the directions of the 1D fBms are randomized. If False, the directions are
            equally spaced between 0 and pi.
        """
        self.H = H
        self.num_components = num_components
        self.rng = np.random.default_rng(seed)

        self.a_x = a_x
        self.a_y = a_y
        self.n_fbm_density = n_fbm_density

        self.n_fbm_x = int(a_x * n_fbm_density)
        self.n_fbm_y = int(a_y * n_fbm_density)

        t0 = time.time()

        fBm = FractionalBrownianMotion(hurst=H, t=2 * (a_x + a_y), rng=self.rng)
        self.fBm_samples = np.array([fBm.sample(2 * (self.n_fbm_x + self.n_fbm_y)) for _ in range(num_components)])
        if verbose: print(f"1D fBms generated in {time.time()-t0:.3f} seconds")
        # Vectors pointing in the directions of the 1D fBms
        if randomize_directions:
            alphas = self.rng.uniform(0, np.pi, num_components)
        else:
            dalpha = np.pi/num_components # The angle between two Brownian motions
            alphas = np.arange(num_components) * dalpha
        self.fBm_directions = np.zeros((2, num_components))
        self.fBm_directions[0, :] = np.cos(alphas)
        self.fBm_directions[1, :] = np.sin(alphas)

    def evaluate_grid(self, n_x, n_y, upsampling_coef=1, n_threads=1, verbose=True):
        """
        Returns the surface in a cartesian grid. One pixel spans a width of a_x/n_x and a
        height of a_y/n_y.

        Time complexity is O(n_x * n_y * upsampling_coef^2 * num_components / n_threads). 
        Memory complexity is O(n_x * n_y / n_threads).

        Parameters
        ----------
        n_x : int
            The number of pixels in the x direction.
        n_y : int
            The number of pixels in the y direction.
        upsampling_coef : int
            With upsampling_coef=1, the value of a pixel represents the value of the midpoint
            of the pixel (this is what one normaly wants). With upsampling_coef > 1 the pixel
            is generated at a upsampling_coef times higher resolution and then averaged to get
            the final value of the pixel. This can be useful in the niche case when one wants
            the pixels to more closely represents the mean height of a pixel.
        n_threads : int
            The number of threads to use for parallelization. If n_threads > 1, the function
            will use multiprocessing to parallelize the computation of the surface.
        """

        z = np.zeros((n_y, n_x))

        # The grid is calculated in chunks of size SCAN_SIZE x SCAN_SIZE in order to reduce memory usage
        SCAN_SIZE = 1024 

        t0 = time.time()
        # The dot product between the normal vectors of the brownian motions and the coordinates of a block of size SCAN_SIZE x SCAN_SIZE
        # When using this, one has to account for the offset of the block from the origin of the surface
        # One block has a width of a_x/upsampling_coef and a height of a_y/upsampling_coef

        if n_threads > 1:
            from multiprocessing import Pool
            if verbose: print(f"Using {n_threads} processes")

            fBm_samples = np.array_split(self.fBm_samples, n_threads)
            fBm_directions = np.array_split(self.fBm_directions, n_threads, axis=1)
            with Pool(n_threads) as p:
                z = p.starmap(_generate_pfbs_internal, [(fBm_samples[i], n_x, n_y, self.a_x, self.a_y, upsampling_coef, self.n_fbm_density, SCAN_SIZE, fBm_directions[i], verbose) for i in range(n_threads)])
                z = np.sum(np.stack(z, axis=2), axis=2)
        else:
            z = _generate_pfbs_internal(self.fBm_samples, n_x, n_y, self.a_x, self.a_y, upsampling_coef, self.n_fbm_density, SCAN_SIZE, self.fBm_directions, verbose)

        if verbose: print(f"planar fractional Brownian surface generated in {time.time()-t0:.3f} seconds")
        return z
    
    def evaluate_points(self, x, y, n_threads=1, verbose=True):
        """
        Returns the heights of the surface at the given points.

        Time complexity is O(len(x) * num_components / n_threads).
        Memory complexity is O(len(x) * num_components / n_threads).

        Parameters
        ----------
        x : np.array
            The x-coordinate of the points, within [0, a_x).
        y : np.array
            The y-coordinate of the points, within [0, a_y).
        """
        t0 = time.time()

        # The grid is calculated in chunks of size SCAN_SIZE x SCAN_SIZE in order to reduce memory usage
        SCAN_SIZE = 1024 

        if n_threads > 1:
            from multiprocessing import Pool
            if verbose: print(f"Using {n_threads} processes")

            fBm_samples = np.array_split(self.fBm_samples, n_threads)
            fBm_directions = np.array_split(self.fBm_directions, n_threads, axis=1)
            with Pool(n_threads) as p:
                z = p.starmap(_generate_pfbs_internal_points, [(fBm_samples[i], fBm_directions[i], x, y, self.n_fbm_density, SCAN_SIZE, verbose) for i in range(n_threads)])
                z = np.sum(np.stack(z, axis=1), axis=1)
        else:
            z = _generate_pfbs_internal_points(self.fBm_samples, self.fBm_directions, x, y, self.n_fbm_density, SCAN_SIZE, verbose=verbose)

        if verbose: print(f"planar fractional Brownian surface generated in {time.time()-t0:.3f} seconds")
        return z


if __name__ == "__main__":
    #import matplotlib.pyplot as plt
    sphere = SphericalFractionalBrownianSurface(2048*64, H=0.5, num_components=4, seed=42)
    z = sphere.evaluate_equilateral(2048, n_threads=2)
    z_points = sphere.evaluate_points(np.random.uniform(0, 2*np.pi, 1024), np.random.uniform(-np.pi/2, np.pi/2, 1024))
    print(np.mean(z), np.std(z), np.mean(z_points))
    #plt.imshow(z)
    #plt.show()
    
    plane = PlanarFractionalBrownianSurface(1024, 1024, H=0.5, num_components=30, seed=42)
    z = plane.evaluate_grid(1024, 1024, n_threads=2)
    z_points = plane.evaluate_points(np.random.uniform(0, 1024, 1024), np.random.uniform(0, 1024, 1024))
    print(np.mean(z), np.std(z), np.mean(z_points))
    #plt.imshow(z)
    #plt.show()