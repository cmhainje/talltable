import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from astropy.time import Time
from astroquery.jplhorizons import Horizons
from scipy.interpolate import interp1d

from talltable.query import DUCK_IMAGE
from talltable.paths import EPHEM_DB_PATH


def get_horizons_data_interpolated(obj_id, mjd_times, step_minutes=2):
    """
    Get Horizons data at regular intervals, then interpolate to specific times.
    This is MUCH faster than querying at every individual time point.
    
    Parameters:
    -----------
    obj_id : str
        Object ID for Horizons query
    mjd_times : array-like
        Array of Modified Julian Date times where we need positions
    step_minutes : float
        Step size in minutes for regular grid (default 2 minutes)
    
    Returns:
    --------
    interpolated_data : dict
        Dictionary with interpolated x, y, z, vx, vy, vz arrays
    """
    
    # Convert MJD to JD
    times_target = Time(mjd_times, format='mjd')
    jd_target = times_target.jd
    
    # Create regular time grid covering the full range with some padding
    mjd_min = mjd_times.min() - 0.1  # Add small padding
    mjd_max = mjd_times.max() + 0.1
    time_start = Time(mjd_min, format='mjd')
    time_end = Time(mjd_max, format='mjd')
    
    print(f"Target times: {len(mjd_times)} points")
    print(f"MJD range: {mjd_times.min():.2f} to {mjd_times.max():.2f}")
    print(f"Querying regular grid with {step_minutes} min step...")
    
    # Query Horizons on regular grid
    obj_bary = Horizons(id=obj_id, location='@0', 
                        epochs={'start': time_start.iso, 
                               'stop': time_end.iso, 
                               'step': f'{step_minutes}m'})
    vectors = obj_bary.vectors(refplane='ecliptic')
    
    print(f"Retrieved {len(vectors)} grid points")
    
    # Extract grid data
    jd_grid = np.array(vectors['datetime_jd'])
    x_grid = np.array(vectors['x'])
    y_grid = np.array(vectors['y'])
    z_grid = np.array(vectors['z'])
    vx_grid = np.array(vectors['vx'])
    vy_grid = np.array(vectors['vy'])
    vz_grid = np.array(vectors['vz'])
    
    # Interpolate to target times
    print(f"Interpolating to {len(jd_target)} target times...")
    
    # Use cubic interpolation for smooth trajectories
    x_interp = interp1d(jd_grid, x_grid, kind='cubic', fill_value='extrapolate')(jd_target)
    y_interp = interp1d(jd_grid, y_grid, kind='cubic', fill_value='extrapolate')(jd_target)
    z_interp = interp1d(jd_grid, z_grid, kind='cubic', fill_value='extrapolate')(jd_target)
    vx_interp = interp1d(jd_grid, vx_grid, kind='cubic', fill_value='extrapolate')(jd_target)
    vy_interp = interp1d(jd_grid, vy_grid, kind='cubic', fill_value='extrapolate')(jd_target)
    vz_interp = interp1d(jd_grid, vz_grid, kind='cubic', fill_value='extrapolate')(jd_target)
    
    print("Interpolation complete!")
    
    return {
        'x': x_interp,
        'y': y_interp,
        'z': z_interp,
        'vx': vx_interp,
        'vy': vy_interp,
        'vz': vz_interp,
        'jd': jd_target
    }


# get a list of MJDs
img_data = duckdb.sql(
    f"SELECT imageid, 0.5 * (t_beg + t_end) as mjd FROM {DUCK_IMAGE}"
).fetchnumpy()
mjd = img_data['mjd']
print(f"{len(mjd)} times found")

data = {}
data['imageid'] = img_data['imageid']

print("Getting SPHEREx data...")
spherex_data = get_horizons_data_interpolated('-163182', mjd, step_minutes=3)
data['spherex_x']  = spherex_data['x']
data['spherex_y']  = spherex_data['y']
data['spherex_z']  = spherex_data['z']
data['spherex_vx'] = spherex_data['vx']
data['spherex_vy'] = spherex_data['vy']
data['spherex_vz'] = spherex_data['vz']

print("\nGetting Earth data...")
earth_data = get_horizons_data_interpolated('399', mjd, step_minutes=3)
data['earth_x']  = earth_data['x']
data['earth_y']  = earth_data['y']
data['earth_z']  = earth_data['z']
data['earth_vx'] = earth_data['vx']
data['earth_vy'] = earth_data['vy']
data['earth_vz'] = earth_data['vz']

print("\nGetting Moon data...")
moon_data = get_horizons_data_interpolated('301', mjd, step_minutes=3)
data['moon_x']  = moon_data['x']
data['moon_y']  = moon_data['y']
data['moon_z']  = moon_data['z']
data['moon_vx'] = moon_data['vx']
data['moon_vy'] = moon_data['vy']
data['moon_vz'] = moon_data['vz']

print("\nGetting Sun data...")
sun_data = get_horizons_data_interpolated('10', mjd, step_minutes=3)
data['sun_x']  = sun_data['x']
data['sun_y']  = sun_data['y']
data['sun_z']  = sun_data['z']
data['sun_vx'] = sun_data['vx']
data['sun_vy'] = sun_data['vy']
data['sun_vz'] = sun_data['vz']

print("\nGetting Jupiter data...")
jupiter_data = get_horizons_data_interpolated('599', mjd, step_minutes=3)
data['jupiter_x']  = jupiter_data['x']
data['jupiter_y']  = jupiter_data['y']
data['jupiter_z']  = jupiter_data['z']
data['jupiter_vx'] = jupiter_data['vx']
data['jupiter_vy'] = jupiter_data['vy']
data['jupiter_vz'] = jupiter_data['vz']

print("\n✓ All ephemeris data retrieved and interpolated!")

pq.write_table(pa.table(data), EPHEM_DB_PATH)

