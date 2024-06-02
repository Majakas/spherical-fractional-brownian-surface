import numpy as np
import time
import os
from tqdm import tqdm
import numba

from stochastic.processes.continuous import FractionalBrownianMotion


def downsample_height_map(z, k):
    """Downsamples the height map by averaging blocks of size k x k. k needs to be
    divisible by the height and width of the height map."""
    return z.reshape(z.shape[0] // k, k, z.shape[1] // k, k).mean(axis=(1, 3))


@numba.jit(nopython=True)
def meshgrid(x, y):
    # np.meshgrid function that works with numba
    xx = np.empty(shape=(y.size, x.size), dtype=x.dtype)
    yy = np.empty(shape=(y.size, x.size), dtype=y.dtype)
    for i in range(y.size):
        for j in range(x.size):
            xx[i,j] = x[j]
            yy[i,j] = y[i]
    return xx, yy


@numba.jit
def compute_sfBs_component(fBm_sample, vec_pole, lon, lat):
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


def _generate_sfbs_internal(fBm_samples, vec_poles, n_x, upsampling_coef, SCAN_SIZE):
    """
    Internal loop for calculating the sfBs. The function is split off from the main function
    to allow for parallelization.
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

    for i in tqdm(range(n_steps_tot)):
        i_step_y = i // n_steps_x // len(fBm_samples)
        i_step_x = i // len(fBm_samples) % n_steps_x
        i_fBm = i % len(fBm_samples)
        iy_min = i_step_y*SCAN_SIZE
        iy_max = min((i_step_y+1)*SCAN_SIZE, n_x*upsampling_coef//2)
        ix_min = i_step_x*SCAN_SIZE
        ix_max = min((i_step_x+1)*SCAN_SIZE, n_x*upsampling_coef)

        delta_z = compute_sfBs_component(
            fBm_samples[i_fBm], vec_poles[i_fBm], lon[ix_min:ix_max], lat[iy_min:iy_max]
        )
        if upsampling_coef > 1:
            delta_z = downsample_height_map(delta_z, upsampling_coef)
        z[iy_min//upsampling_coef:iy_max//upsampling_coef, ix_min//upsampling_coef:ix_max//upsampling_coef] +=\
            delta_z
    return z


def generate_spherical_fractional_brownian_surface(n_x, H=0.5, num_components=10, n_threads=1, upsampling_coef=1, fbm_interpolation_coef=64, seed=42):
    """
    Returns a spherical fractional Brownian surface with the given Hurst exponent in
    equirectangular projection (each pixel spans an equal number of degrees in longitude
    and latitude). The resulting surface has an exact height-height correlation function
    that decays as a power law with exponent 2H and tends to more isotropy as num_components
    increases. The time complexity of the algorithm is O(n_x^2 * num_components + n_x^2 *
    fbm_interpolation_coef) and memory footprint is O(n_x^2 * n_threads + n_x * fbm_interpolation_coef).

    The surface is generated by summing num_components 1D fractional Brownian motions projected
    onto a sphere with random orientations.
    
    Parameters
    ----------
    n_x : int
        The number of pixels in the x direction. The number of pixels in y is n_x // 2
        because of the projection.
    H : float
        The Hurst exponent of the fractional Brownian motion. Must be in the range (0, 1).
        The bigger H is, the smoother the surface. H=0.5 is the fastest to compute.
    num_components : int
        The number of 1D fractional Brownian motions to sum over to generate the surface.
        The bigger num_components is, the more isotropic the surface.
    n_threads : int
        The number of threads to use for parallelization. If n_threads > 1, the function
        will use multiprocessing to parallelize the computation of the surface.
    upsampling_coef : int
        With upsampling_coef=1, the value of a pixel represents the value of the midpoint
        of the pixel (this is what one normaly wants). With upsampling_coef > 1 the pixel
        is generated at a upsampling_coef times higher resolution and then averaged to get
        the final value of the pixel. This can be useful in the niche case when one wants
        the pixels to more closely represents the mean height of a pixel.
    fbm_interpolation_coef : int
        Interpolation factor for the fractional Brownian motion. While the fBm is generated
        using an exact algorithm (Hoskin's method), in order to get the value at a fractional
        latitude, the fBm is sampled  at a higher resolution and then interpolated to get
        the value at the fractional latitude.
    seed : int
        The seed for the random number generator.
    """
    def generate_random_unit_vector(rng):
        # Generates a random unit vector in Cartesian coordinates
        lon = rng.uniform(-np.pi, np.pi)
        lat = np.arcsin(rng.uniform(-1, 1))
        return np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])

    rng = np.random.default_rng(seed)
    n_y = n_x // 2

    z = np.zeros((n_y, n_x))

    # The grid is calculated in chunks of size SCAN_SIZE x SCAN_SIZE in order to reduce memory usage
    SCAN_SIZE = 1024 

    t0 = time.time()

    n_fbm = n_y * upsampling_coef * fbm_interpolation_coef

    fBm = FractionalBrownianMotion(hurst=H, t=n_fbm, rng=rng)
    fBm_samples = np.array([fBm.sample(n_fbm) for _ in range(num_components)])
    print(f"1D fBms generated in {time.time()-t0:.3f} seconds")
    # unit vectors pointing towards the poles of the different components
    vec_poles = np.array([generate_random_unit_vector(rng) for _ in range(num_components)])

    if n_threads > 1:
        from multiprocessing import Pool
        print(f"Using {n_threads} processes")

        fBm_samples = np.array_split(fBm_samples, n_threads)
        vec_poles = np.array_split(vec_poles, n_threads)
        with Pool(n_threads) as p:
            z = p.starmap(_generate_sfbs_internal, [(fBm_samples[i], vec_poles[i], n_x, upsampling_coef, SCAN_SIZE) for i in range(n_threads)])
            z = np.sum(np.stack(z, axis=2), axis=2)
    else:
        z = _generate_sfbs_internal(fBm_samples, vec_poles, n_x, upsampling_coef, SCAN_SIZE)

    print(f"spherical fractional Brownian surface generated in {time.time()-t0:.3f} seconds")
    return z


def _generate_pfbs_internal(fBm_samples, n_x, n_y, upsampling_coef, fbm_interpolation_coef, SCAN_SIZE, fBm_kernels, fBm_directions):
    """
    Internal loop for calculating the pfBs. The function is split off from the main function
    to allow for parallelization.
    """
    def ceil_divide(a, b):
        return -(-a // b)
    
    z = np.zeros((n_y, n_x))
    n_steps_x = ceil_divide(n_x*upsampling_coef, SCAN_SIZE)
    n_steps_y = ceil_divide(n_y*upsampling_coef, SCAN_SIZE)
    n_steps_tot = n_steps_x * n_steps_y * len(fBm_samples)

    for i in tqdm(range(n_steps_tot)):
        i_step_y = i // n_steps_x // len(fBm_samples)
        i_step_x = i // len(fBm_samples) % n_steps_x
        i_fBm = i % len(fBm_samples)
        iy_min = i_step_y*SCAN_SIZE
        iy_max = min((i_step_y+1)*SCAN_SIZE, n_y*upsampling_coef)
        ix_min = i_step_x*SCAN_SIZE
        ix_max = min((i_step_x+1)*SCAN_SIZE, n_x*upsampling_coef)

        fBm_index = ((fBm_kernels[:iy_max-iy_min,:ix_max-ix_min, i_fBm] + ix_min*fBm_directions[0, i_fBm] + iy_min*fBm_directions[1, i_fBm]) * fbm_interpolation_coef).astype(np.int64)
        fBm_index += len(fBm_samples[i_fBm])//2 # To avoid negative indices
        delta_z = np.take(fBm_samples[i_fBm], fBm_index)
        if upsampling_coef > 1:
            delta_z = downsample_height_map(delta_z, upsampling_coef)

        z[iy_min//upsampling_coef:iy_max//upsampling_coef, ix_min//upsampling_coef:ix_max//upsampling_coef] +=\
            delta_z
    return z


def generate_planar_fractional_brownian_surface(n_x, n_y, H=0.5, num_components=2, n_threads=1, upsampling_coef=1, fbm_interpolation_coef=64, seed=42):
    """
    Returns a planar fractional Brownian surface with the given Hurst exponent. The resulting
    surface has an exact height-height correlation function that decays as a power law with
    exponent 2H and tends to more isotropy as num_components increases. The time complexity of
    the algorithm is O(n_x * n_y * num_components + (n_x + n_y)^2 * fbm_interpolation_coef^2)
    and memory footprint is O(n_x * n_y * n_threads + (n_x + n_y) * fbm_interpolation_coef).

    The surface is generated by summing num_components 1D fractional Brownian motions
    angled at different directions. The fBms are angled at equal intervals between 0 and pi.
    If the direction vector of the i-th fBm is \vec n_i, then the elevation of the
    surface has the form z = \sum_i fBm_i(\vec n_i \cdot \vec x), where \vec x is the
    2D vector of the coordinates of the surface.

    Parameters
    ----------
    n_x : int
        The number of pixels in the x direction.
    n_y : int
        The number of pixels in the y direction.
    H : float
        The Hurst exponent of the fractional Brownian motion. Must be in the range (0, 1).
        The bigger H is, the smoother the surface. H=0.5 is the fastest to compute.
    num_components : int
        The number of 1D fractional Brownian motions to sum over to generate the surface.
        The bigger num_components is, the more isotropic the surface.
    n_threads : int
        The number of threads to use for parallelization. If n_threads > 1, the function
        will use multiprocessing to parallelize the computation of the surface.
    upsampling_coef : int
        With upsampling_coef=1, the value of a pixel represents the value of the midpoint
        of the pixel (this is what one normaly wants). With upsampling_coef > 1 the pixel
        is generated at a upsampling_coef times higher resolution and then averaged to get
        the final value of the pixel. This can be useful in the niche case when one wants
        the pixels to more closely represents the mean height of a pixel.
    fbm_interpolation_coef : int
        Interpolation factor for the fractional Brownian motion. While the fBm is generated
        using an exact algorithm (Hoskin's method), in order to get the value at a fractional
        position, the fBm is sampled  at a higher resolution and then interpolated to get
        the value at the fractional position.
    seed : int
        The seed for the random number generator.
    """
    SCAN_SIZE = 1024

    alpha = np.pi/num_components # The angle between two Brownian motions
    n = int(2 * n_x + 2 * n_y) * upsampling_coef * fbm_interpolation_coef

    t0 = time.time()

    rng = np.random.default_rng(seed)
    fbm = FractionalBrownianMotion(hurst=H, t=n/fbm_interpolation_coef/upsampling_coef, rng=rng)
    fBm_samples = np.array([fbm.sample(n) for _ in range(num_components)])
    print(f"1D fBms generated in {time.time()-t0:.3f} seconds")
    fBm_directions = np.zeros((2, num_components)) # Vectors pointing in the directions of the 1D fBms
    fBm_directions[0, :] = np.cos(alpha*np.arange(num_components))
    fBm_directions[1, :] = np.sin(alpha*np.arange(num_components))
    
    # The dot product between the normal vectors of the brownian motions and the coordinates of a block of size SCAN_SIZE x SCAN_SIZE
    # When using this, one has to account for the offset of the block from the origin of the surface
    x0, y0 = np.meshgrid(np.arange(0, SCAN_SIZE), np.arange(0, SCAN_SIZE))
    fBm_kernels = np.stack([x0*fBm_directions[0, i] + y0*fBm_directions[1, i] for i in range(num_components)], axis=2)

    if n_threads > 1:
        from multiprocessing import Pool
        print(f"Using {n_threads} processes")

        fBm_samples = np.array_split(fBm_samples, n_threads)
        fBm_kernels = np.array_split(fBm_kernels, n_threads, axis=2)
        fBm_directions = np.array_split(fBm_directions, n_threads, axis=1)
        with Pool(n_threads) as p:
            z = p.starmap(_generate_pfbs_internal, [(fBm_samples[i], n_x, n_y, upsampling_coef, fbm_interpolation_coef, SCAN_SIZE, fBm_kernels[i], fBm_directions[i]) for i in range(n_threads)])
            z = np.sum(np.stack(z, axis=2), axis=2)
    else:
        z = _generate_pfbs_internal(fBm_samples, n_x, n_y, upsampling_coef, fbm_interpolation_coef, SCAN_SIZE, fBm_kernels, fBm_directions)
    # Make the mean of the surface be 0
    z -= np.mean(z)

    print(f"planar fractional Brownian surface generated in {time.time()-t0:.3f} seconds")
    return z


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    #z = generate_spherical_fractional_brownian_surface(2040, H=0.5, num_components=4, n_threads=4, upsample_coef=1, fbm_interpolation_coef=64, seed=42)
    #z = generate_spherical_fractional_brownian_surface(2048, H=0.5, num_components=10, n_threads=1, upsampling_coef=1, fbm_interpolation_coef=64, seed=42)
    #plt.imshow(z)
    #plt.show()
    z = generate_planar_fractional_brownian_surface(4096, 1024, H=0.5, num_components=30, n_threads=1, upsampling_coef=1, fbm_interpolation_coef=64)
    plt.imshow(z)
    plt.show()