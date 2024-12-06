import logging
from typing import Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.propagator import Propagator
from adam_core.time import Timestamp
from precovery.main import PrecoveryCandidates
from thor.orbit_determination import FittedOrbitMembers, FittedOrbits

logger = logging.getLogger(__name__)


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

    For more eccentric orbits or orbits that are unbound, the period search is limited to 10 years
    (+/- 5 years from orbit's epoch).

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
    if np.isnan(P) or np.isinf(P):
        P = 10 * 365.25  # Default to 5 years if the period is not defined
    P = np.minimum(P, 10 * 365.25)
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


def check_candidates_astrometric_errors(
    candidates: PrecoveryCandidates,
    astrometric_errors: dict[str, Tuple[float, float]],
) -> PrecoveryCandidates:
    """
    Check if the precovery candidates contain any missing uncertainties. If they do then
    assign a default uncertainty to them.

    Parameters
    ----------
    candidates
        The precovery candidates.
    astrometric_error
        The astrometric error to assign to the candidates missing uncertainties.
        Should at least contain a "default" key with a tuple of (ra_sigma_arcsec, dec_sigma_arcsec).

    Returns
    -------
    candidates
        The precovery candidates with the missing uncertainties assigned.
    """
    assert "default" in astrometric_errors, "Default astrometric error is not defined."

    ra_sigma_arcsec = candidates.ra_sigma_arcsec.to_numpy(zero_copy_only=False)
    dec_sigma_arcsec = candidates.dec_sigma_arcsec.to_numpy(zero_copy_only=False)
    observatory_codes = candidates.obscode.to_numpy(zero_copy_only=False)

    invalid_ra_sigma = np.isnan(ra_sigma_arcsec)
    invalid_dec_sigma = np.isnan(dec_sigma_arcsec)
    if np.any(invalid_ra_sigma):
        invalid_observatory_codes = np.unique(observatory_codes[invalid_ra_sigma])
        for code in invalid_observatory_codes:
            logger.warning(
                f"Found missing 1-sigma RA uncertainties for observatory code: {code}"
            )
            if code not in astrometric_errors:
                ra_sigma_arcsec = np.where(
                    observatory_codes == code,
                    astrometric_errors["default"][0],
                    ra_sigma_arcsec,
                )
            else:
                ra_sigma_arcsec = np.where(
                    observatory_codes == code,
                    astrometric_errors[code][0],
                    ra_sigma_arcsec,
                )

        candidates = candidates.set_column("ra_sigma_arcsec", pa.array(ra_sigma_arcsec))

    if np.any(invalid_dec_sigma):
        invalid_observatory_codes = np.unique(observatory_codes[invalid_dec_sigma])
        for code in invalid_observatory_codes:
            logger.warning(
                f"Found missing 1-sigma Dec uncertainties for observatory code: {code}"
            )
            if code not in astrometric_errors:
                dec_sigma_arcsec = np.where(
                    observatory_codes == code,
                    astrometric_errors["default"][1],
                    dec_sigma_arcsec,
                )
            else:
                dec_sigma_arcsec = np.where(
                    observatory_codes == code,
                    astrometric_errors[code][1],
                    dec_sigma_arcsec,
                )

        candidates = candidates.set_column(
            "dec_sigma_arcsec", pa.array(dec_sigma_arcsec)
        )

    return candidates


def drop_coincident_candidates(
    candidates: PrecoveryCandidates,
) -> PrecoveryCandidates:
    """
    Drop coincident candidates from the precovery candidates. These are candidates that have the same
    observation time as another candidate. The candidate with the lowest distance is kept.

    Parameters
    ----------
    candidates
        The precovery candidates.

    Returns
    -------
    candidates
        The precovery candidates with the coincident candidates dropped.
    """
    # Flatten the table so nested columns are dot-delimited at the top level
    flattened_table = candidates.flattened_table()

    # Add index to flattened table
    flattened_table = flattened_table.add_column(
        0, "index", pa.array(np.arange(len(flattened_table)))
    )

    # Sort by time, then distance, then obscode
    flattened_table = flattened_table.sort_by(
        [
            ("time.days", "ascending"),
            ("time.nanos", "ascending"),
            ("distance_arcsec", "ascending"),
        ]
    )

    # Group by orbit ID and observation time
    indices = (
        flattened_table.group_by(
            ["time.days", "time.nanos"],
            use_threads=False,
        )
        .aggregate([("index", "first")])
        .column("index_first")
    )

    filtered = candidates.take(indices)
    if filtered.fragmented():
        filtered = qv.defragment(filtered)

    return filtered


def identify_found_missed_and_new(
    obs_ids_prev: pa.Array, obs_ids_iter: pa.Array
) -> Tuple[list, list, list]:
    """
    Compares the previous observation IDs to current observation IDs and returns:
    - The list of previous observation IDs that were found in the current iteration
    - The list of previous observation IDs that were missed in the current iteration
    - The list of new observation IDs that were not present in the previous iteration

    Parameters
    ----------
    obs_ids_prev : pyarrow.Array
        The previous observation IDs
    obs_ids_iter : pyarrow.Array
        The current observation IDs

    Returns
    -------
    found : list
        The list of previous observation IDs that were found in the current iteration
    missed : list
        The list of previous observation IDs that were missed in the current iteration
    new: list
        The list of new observation IDs that were not present in the previous iteration
    """
    if len(obs_ids_prev) == 0:
        return [], [], obs_ids_iter.to_pylist()
    found_mask = pc.is_in(obs_ids_prev, obs_ids_iter)
    missed_mask = pc.invert(found_mask)
    new_mask = pc.invert(pc.is_in(obs_ids_iter, obs_ids_prev))
    return (
        obs_ids_prev.filter(found_mask).to_pylist(),
        obs_ids_prev.filter(missed_mask).to_pylist(),
        obs_ids_iter.filter(new_mask).to_pylist(),
    )


def assign_duplicate_observations(
    orbits, orbit_members: FittedOrbitMembers
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Assigns observations that have been assigned to multiple orbits to the orbit with t
    he most observations, longest arc length, and lowest reduced chi2.
    Parameters
    ----------
    orbit_members : `~thor.orbit_determination.FittedOrbitMembers`
        Fitted orbit members.
    Returns
    -------
    filtered : `~thor.orbit_determination.FittedOrbits`
        Fitted orbits with duplicate assignments removed.
    filtered_orbit_members : `~thor.orbit_determination.FittedOrbitMembers`
        Fitted orbit members with duplicate assignments removed.
    """
    # Sort by number of observations, arc length, and reduced chi2
    # Here we assume that orbits that generally have more observations, longer arc lengths,
    # and lower reduced chi2 are better as candidates for assigning detections
    # that have been assigned to multiple orbits.
    orbits_sorted = orbits.sort_by(
        [
            ("num_obs", "descending"),
            ("arc_length", "descending"),
            ("reduced_chi2", "ascending"),
        ]
    )

    # Extract the orbit IDs from the sorted table
    orbit_ids = orbits_sorted.orbit_id.unique()

    # Calculate the order in which these orbit IDs appear in the orbit_members table
    order_in_orbits = pc.index_in(orbit_members.orbit_id, orbit_ids)

    # Create an index into the orbit_members table and append the order_in_orbits column
    orbit_members_table = (
        orbit_members.flattened_table()
        .append_column("index", pa.array(np.arange(len(orbit_members))))
        .append_column("order_in_orbits", order_in_orbits)
    )

    # Drop the residual values (a list column) due to: https://github.com/apache/arrow/issues/32504
    orbit_members_table = orbit_members_table.drop_columns(["residuals.values"])

    # Sort orbit members by the orbit IDs (in the same order as the orbits table)
    orbit_members_table = orbit_members_table.sort_by(
        [("order_in_orbits", "ascending")]
    )

    # Now group by the orbit ID and aggregate the index column to get the first index for each orbit ID
    indices = (
        orbit_members_table.group_by("obs_id", use_threads=False)
        .aggregate([("index", "first")])
        .column("index_first")
    )

    # Use the indices to filter the orbit_members table and
    # then use the resulting orbit IDs to filter the orbits table
    filtered_orbit_members = orbit_members.take(indices)
    filtered = orbits.apply_mask(
        pc.is_in(orbits.orbit_id, filtered_orbit_members.orbit_id)
    )

    # Defragment the tables
    if filtered.fragmented():
        filtered = qv.defragment(filtered)
    if filtered_orbit_members.fragmented():
        filtered_orbit_members = qv.defragment(filtered_orbit_members)

    return filtered, filtered_orbit_members
