from typing import Tuple

import numpy as np
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.propagator import Propagator
from adam_core.time import Timestamp


def calculate_astrometric_uncertainty(
    sigma_lon: np.ndarray, sigma_lat: np.ndarray, lat: np.ndarray
) -> np.ndarray:
    """
    Calculate the astrometric uncertainty from the given sigma_lon and sigma_lat

    Parameters
    ----------
    sigma_lon (N)
        The uncertainty in the longitude in degrees.
    sigma_lat (N)
        The uncertainty in the latitude in degrees.
    lat (N)
        The latitude in degrees.

    Returns
    -------
    uncertainty (N)
        The astrometric uncertainty.
    """
    return np.sqrt((sigma_lat) ** 2 + (np.cos(np.radians(lat)) * sigma_lon) ** 2)


def compute_search_window_and_tolerance(
    orbit: Orbits,
    propagator: Propagator,
    max_tolerance: float = 10,
    steps: int = 1000,
    sigma: float = 5.0,
) -> Tuple[float, float, float]:
    """
    Propagate the orbit in even time intervals forward and backward half an orbital period.
    Calculate the time range where the on-sky uncertainty is less than the maximum tolerance.
    This is calculated by propagating the covariance matrix via sigma-point sampling and calculating the
    predicted ephemeris for a geocentric observer. The maximum uncertainty is then
    calculated at desired sigma-level.

    For highly uncertain orbits, the time range where the uncertainty is less than
    the maximum tolerance may be empty.

    Parameters
    ----------
    orbit (1)
        Orbit to propagate. It must have a covariance matrix defined.
    propagator
        The propagator to use to generate the ephemeris.
    max_tolerance
        The maximum tolerance in arcseconds.
    steps
        The number of steps to propagate the orbit to.

    Returns
    -------
    min_mjd_utc
        The minimum MJD in UTC where the uncertainty is less than the maximum tolerance.
    max_mjd_utc
        The maximum MJD in UTC where the uncertainty is less than the maximum tolerance.
    max_uncertainty
        The maximum uncertainty in arcseconds.
    """
    assert len(orbit) == 1, "More than one orbit is not supported."

    # Define the time range
    P = orbit.coordinates.to_keplerian().P[0]
    epoch_mjd = orbit.coordinates.time.rescale("utc").mjd().to_numpy()[0]
    mjd = np.linspace(epoch_mjd - P / 2, epoch_mjd + P / 2, steps, endpoint=True)
    time = Timestamp.from_mjd(mjd, scale="utc")
    observers = Observers.from_code("500", time)

    # Generate ephemeris and propagate the covariance matrix
    # to each point
    ephemeris = propagator.generate_ephemeris(
        orbit,
        observers,
        covariance=True,
        covariance_method="sigma-point",
        chunk_size=1000,
        num_samples=1000,
        max_processes=1,
    )

    # Find the maximum uncertainty in the ephemeris at the desired sigma-level
    tolerance = sigma * calculate_astrometric_uncertainty(
        ephemeris.coordinates.sigma_lon,
        ephemeris.coordinates.sigma_lat,
        ephemeris.coordinates.lat.to_numpy(),
    )

    # Find the time range where the uncertainty is less than the maximum
    # tolerance
    mask = tolerance < (max_tolerance / 3600)
    valid_tolerances = tolerance[mask]
    valid_mjds = mjd[mask]

    if len(valid_mjds) == 0:
        return np.nan, np.nan, np.nan

    return valid_mjds.min(), valid_mjds.max(), valid_tolerances.max() * 3600
