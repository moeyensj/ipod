import logging
from typing import Optional, Tuple, Type, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from adam_core.orbit_determination import (
    OrbitDeterminationObservations,
    evaluate_orbits,
)
from adam_core.orbits import Orbits
from adam_core.propagator import PYOORB, Propagator
from precovery.precovery_db import PrecoveryDatabase
from thor.orbit_determination import FittedOrbitMembers, FittedOrbits
from thor.orbits.od import od

from .utils import (
    check_candidates_astrometric_errors,
    compute_search_window_and_tolerance,
    drop_coincident_candidates,
    identify_found_missed_and_new,
)

logger = logging.getLogger(__name__)


class SearchSummary(qv.Table):
    orbit_id = qv.LargeStringColumn()
    min_mjd = qv.Float64Column()
    max_mjd = qv.Float64Column()
    num_candidates = qv.Int64Column()
    num_accepted = qv.Int64Column()
    num_rejected = qv.Int64Column()
    arc_length_prev = qv.Float64Column()
    arc_length = qv.Float64Column()
    num_obs_prev = qv.Int64Column()
    num_obs = qv.Int64Column()


def update_tolerance(tolerance, tolerance_step=2.5):
    if tolerance == 1:
        return tolerance_step
    else:
        return tolerance + tolerance_step


def ipod(
    orbit: Union[Orbits, FittedOrbits],
    orbit_observations: Optional[OrbitDeterminationObservations] = None,
    max_tolerance: float = 10.0,
    delta_time: float = 30.0,
    rchi2_threshold: float = 3.0,
    outlier_chi2: float = 9.0,
    reconsider_chi2: float = 8.0,
    max_iter: int = 10,
    min_mjd: Optional[float] = None,
    max_mjd: Optional[float] = None,
    astrometric_errors: dict = {},
    database: Union[str, PrecoveryDatabase] = "",
    datasets: Optional[set[str]] = None,
    propagator: Type[Propagator] = PYOORB,
    propagator_kwargs: dict = {},
) -> Tuple[
    FittedOrbits, FittedOrbitMembers, OrbitDeterminationObservations, SearchSummary
]:

    # Initialize the propagator
    prop = propagator(**propagator_kwargs)

    # If we haven't received an active precovery database then
    # create a connection to one
    if not isinstance(database, PrecoveryDatabase):
        precovery_db = PrecoveryDatabase.from_dir(
            database,
            create=False,
            mode="r",
            allow_version_mismatch=True,
        )
    else:
        precovery_db = database

    # Get the mjd bounds of the frames in the database
    database_min_mjd, database_max_mjd = precovery_db.frames.idx.mjd_bounds(
        datasets=datasets
    )
    logger.debug(f"Database MJD range: {database_min_mjd:.5f} - {database_max_mjd:.5f}")
    if database_min_mjd is None and database_max_mjd is None:
        if datasets is not None:
            logger.debug(f"No frames for datasets {datasets} in the database.")
        else:
            logger.debug("No frames in the database.")
        return (
            FittedOrbits.empty(),
            FittedOrbitMembers.empty(),
            OrbitDeterminationObservations.empty(),
            SearchSummary.empty(),
        )

    # If the min and max MJD are not given then set them to the min and max MJD
    # of the frames in the database
    if min_mjd is None:
        min_mjd = database_min_mjd
    if max_mjd is None:
        max_mjd = database_max_mjd

    # If the min and max MJD are outside the bounds of the frames in the database
    # then set them to the min and max MJD of the frames in the database
    if min_mjd < database_min_mjd:
        min_mjd = database_min_mjd
        logger.debug(
            "Minimum MJD is before the earliest frame in the database. Setting to the earliest frame."
        )
    if max_mjd > database_max_mjd:
        max_mjd = database_max_mjd
        logger.debug(
            "Maximum MJD is after the latest frame in the database. Setting to the latest frame."
        )

    # If observations have been passed lets make sure that
    # the given orbit has been evaluated with the given observations
    if orbit_observations is not None:
        orbit_iter, orbit_members_iter = evaluate_orbits(
            orbit.to_orbits(),
            orbit_observations,
            prop,
        )

        # We recast ouput orbits and orbit members to the FittedOrbits
        # and FittedOrbitMember from THOR
        orbit_iter = FittedOrbits.from_kwargs(
            orbit_id=orbit_iter.orbit_id,
            coordinates=orbit_iter.coordinates,
            arc_length=orbit_iter.arc_length,
            num_obs=orbit_iter.num_obs,
            chi2=orbit_iter.chi2,
            reduced_chi2=orbit_iter.reduced_chi2,
            iterations=orbit_iter.iterations,
            success=orbit_iter.success,
            status_code=orbit_iter.status_code,
        )
        orbit_members_iter = FittedOrbitMembers.from_kwargs(
            orbit_id=orbit_members_iter.orbit_id,
            obs_id=orbit_members_iter.obs_id,
            residuals=orbit_members_iter.residuals,
            outlier=orbit_members_iter.outlier,
            solution=orbit_members_iter.solution,
        )

        orbit_observations_iter = orbit_observations
        obs_ids_iter = orbit_observations.id

    else:
        orbit_iter = orbit
        orbit_members_iter = FittedOrbitMembers.empty()
        orbit_observations_iter = OrbitDeterminationObservations.empty()
        obs_ids_iter = pa.array([])

    # Start with a tolerance of 1 arcsecond
    tolerance_iter = 1

    # Running list of observation IDs that have been rejected by OD
    # or OD failures
    rejected_obs_ids: list[str] = []

    # Running search summary
    search_summary_iter = {
        "orbit_id": orbit_iter.orbit_id[0].as_py(),
        "min_mjd": 0,
        "max_mjd": 0,
        "num_candidates": 0,
        "num_accepted": 0,
        "num_rejected": 0,
        "arc_length_prev": orbit_iter.arc_length[0].as_py(),
        "arc_length": orbit_iter.arc_length[0].as_py(),
        "num_obs_prev": orbit_iter.num_obs[0].as_py(),
        "num_obs": orbit_iter.num_obs[0].as_py(),
    }

    for i in range(max_iter):
        # Set running list of observation IDs that have been
        # processed thus far. This includes observations which have been
        # identifeid as outliers and rejected
        obs_ids_prev = obs_ids_iter

        logger.debug(f"Starting ipod iteration {i+1}...")
        if tolerance_iter > max_tolerance:
            logger.debug(
                f"Maximum tolerance of {max_tolerance} arcseconds reached. Exiting."
            )
            break

        # Compute the min and max observation times if they exist
        if len(orbit_observations_iter) > 0:
            min_search_mjd = (
                orbit_observations_iter.coordinates.time.min().mjd()[0].as_py()
            )
            max_search_mjd = (
                orbit_observations_iter.coordinates.time.max().mjd()[0].as_py()
            )

        # If they do not exist then set the min and max search to epoch of the orbit
        else:
            min_search_mjd = max_search_mjd = (
                orbit_iter.coordinates.time.rescale("utc").mjd()[0].as_py()
            )

        # Compute the "optimal" search window and tolerance
        min_mjd_iter, max_mjd_iter, _ = compute_search_window_and_tolerance(
            orbit_iter.to_orbits(),
            prop,
            max_tolerance=tolerance_iter,
        )

        # If the "optimal" search window doesn't span the entire observation time range of this
        # iteration then set the search window to the entire observation time range
        if not np.isnan(min_mjd_iter) and not np.isnan(max_mjd_iter):
            if min_mjd_iter > min_search_mjd:
                min_mjd_iter = min_search_mjd
                logger.debug(
                    "Proposed search window start is after the earliest observation MJD."
                )
            if max_mjd_iter < max_search_mjd:
                max_mjd_iter = max_search_mjd
                logger.debug(
                    "Proposed search window end is before the latest observation MJD."
                )

        # If the search window time range is nan which can occur for highly uncertain
        # orbits then set the tolerance to the starting tolerance and set the
        # maximum search window to the entire observation time range
        if np.isnan(min_mjd_iter) or np.isnan(max_mjd_iter):
            logger.debug(
                "Orbit's on-sky uncertainity already exceeds the maximum tolerance. "
                f"Setting search window to +-{delta_time} and tolerance to {max_tolerance:.3f}."
            )
            min_mjd_iter = min_search_mjd - delta_time
            max_mjd_iter = max_search_mjd + delta_time

        # If the search window is outside the minimum and maximum MJD, adjust the search window
        # to be within the minimum and maximum MJD
        if min_mjd is not None and min_mjd_iter is not None:
            min_mjd_iter = np.maximum(min_mjd_iter, min_mjd)
            logger.debug("Proposed search window start is before the minimum MJD.")
        if max_mjd is not None and max_mjd_iter is not None:
            max_mjd_iter = np.minimum(max_mjd_iter, max_mjd)
            logger.debug("Proposed search window end is after the maximum MJD.")

        logger.debug(
            f"Running precovery search between {min_mjd_iter:.5f} and {max_mjd_iter:.5f} "
            f"[dt: {max_mjd_iter-min_mjd_iter:.5f}] with a {tolerance_iter:.3f} arcsecond tolerance..."
        )
        candidates, frame_candidates = precovery_db.precover(
            orbit,
            tolerance=tolerance_iter / 3600,
            start_mjd=min_mjd_iter,
            end_mjd=max_mjd_iter,
            window_size=7,
            datasets=datasets,
        )
        candidates = candidates.sort_by(["time.days", "time.nanos", "obscode"])
        num_candidates = len(candidates)
        logger.debug(f"Found {num_candidates} potential precovery observations.")

        # Check if the candidates contain any missing uncertainties. If they do then
        # assign a default uncertainty to them.
        candidates = check_candidates_astrometric_errors(candidates, astrometric_errors)

        # Drop any coincident candidates
        candidates = drop_coincident_candidates(candidates)
        candidates = candidates.sort_by(["time.days", "time.nanos", "obscode"])

        if len(candidates) != num_candidates:
            logger.debug(
                f"Removed {num_candidates - len(candidates)} coincident candidates."
            )
            num_candidates = len(candidates)

        # Current set of observation IDs up for consideration
        obs_ids_iter = candidates.observation_id

        # Update the search summary
        search_summary_iter["min_mjd"] = min_mjd_iter
        search_summary_iter["max_mjd"] = max_mjd_iter
        search_summary_iter["num_candidates"] = num_candidates

        if len(candidates) < 6:
            logger.debug(
                "Insufficient candidates for orbit determination. Increasing tolerance."
            )
            tolerance_iter = update_tolerance(tolerance_iter)
            continue

        # Convert candidates to OrbitDeterminationObservations
        orbit_observations_iter = OrbitDeterminationObservations.from_kwargs(
            id=obs_ids_iter,
            coordinates=candidates.to_spherical_coordinates(),
            observers=candidates.get_observers(),
        )

        # The orbit solution may have changed so if we have any rejected observations
        # lets evaluate the ensemble and see if any should be re-accepted
        if len(rejected_obs_ids) > 0:
            _, orbit_members_iter_ = evaluate_orbits(
                orbit_iter.to_orbits(),
                orbit_observations_iter,
                prop,
            )

            # We recast ouput orbit members to the FittedOrbits
            orbit_members_iter = FittedOrbitMembers.from_kwargs(
                orbit_id=orbit_members_iter.orbit_id,
                obs_id=orbit_members_iter.obs_id,
                residuals=orbit_members_iter.residuals,
                outlier=orbit_members_iter.outlier,
                solution=orbit_members_iter.solution,
            )

            obs_ids_reaccept = orbit_members_iter_.apply_mask(
                pc.and_(
                    pc.less_equal(orbit_members_iter_.residuals.chi2, reconsider_chi2),
                    pc.is_in(orbit_members_iter_.obs_id, pa.array(rejected_obs_ids)),
                )
            ).obs_id
            if len(obs_ids_reaccept) > 0:
                logger.debug(
                    f"Re-accepting {len(obs_ids_reaccept)} previously rejected observations."
                )
                rejected_obs_ids = list(
                    set(rejected_obs_ids) - set(obs_ids_reaccept.to_pylist())
                )
                search_summary_iter["num_rejected"] = len(rejected_obs_ids)

        found, missed, new = identify_found_missed_and_new(obs_ids_prev, obs_ids_iter)
        logger.debug(f"Found {len(found)} observations from the previous iteration.")
        logger.debug(f"Missed {len(missed)} observations from the previous iteration.")
        logger.debug(f"Found {len(new)} new observations.")

        # If no new observations were found, then lets increase the tolerance and jump to the next
        # iteration. There is no point in orbit fitting if no new observations were found.
        if len(new) == 0:
            tolerance_iter = update_tolerance(tolerance_iter)
            if tolerance_iter > max_tolerance:
                logger.debug(
                    f"Maximum tolerance of {max_tolerance} arcseconds reached. Exiting."
                )
                break
            logger.debug(
                f"No new observations found. Increasing the tolerance to {tolerance_iter} arcseconds."
            )
            continue

        # Remove any candidates that we have previously rejected
        if len(rejected_obs_ids) > 0:
            orbit_observations_iter = orbit_observations_iter.apply_mask(
                pc.invert(
                    pc.is_in(orbit_observations_iter.id, pa.array(rejected_obs_ids))
                )
            )
            logger.debug(f"Removed {len(rejected_obs_ids)} rejected observations.")
            num_candidates = len(orbit_observations_iter)
            search_summary_iter["num_candidates"] = num_candidates

        # Attempt to differentially correct the orbit given the new observations
        # If we have previous observations then we will use them to calculate
        # the contamination percentage
        if len(orbit_observations_iter) > 0:
            contamination_percentage = np.minimum(
                (len(new) + 1) / num_candidates * 100, 100
            )

        # If we do not have previous observations then we will use a default
        # maximum contamination percentage of 50%
        else:
            logger.debug(
                "No previous observations. Running orbit fit with all new observations."
            )
            contamination_percentage = 50.0

        logger.debug(
            f"Running orbit fit with {len(new)} new observations "
            f"and a contamination percentage of {contamination_percentage:.2f}%..."
        )
        orbit_iter_fit, orbit_members_iter_fit = od(
            orbit_iter,
            orbit_observations_iter,
            rchi2_threshold=rchi2_threshold,
            min_obs=6,
            min_arc_length=1,
            contamination_percentage=contamination_percentage,
        )

        if len(orbit_iter_fit) == 0:
            # If the orbit fit failed completely, then remove the newly added observations
            # and add them to the rejected observations list. We are trusting that the OD
            # code did its best job at removing the outliers but that it still failed
            # despite its best efforts. The rejected observations can still be re-accepted
            # in a later iteration if their chi2 is less than the reconsider_chi2.
            if len(obs_ids_prev) > 0:
                obs_ids = orbit_observations_iter.id
                obs_ids_added = obs_ids.filter(
                    pc.invert(pc.is_in(obs_ids, obs_ids_prev))
                )
                rejected_obs_ids.extend(obs_ids_added.to_pylist())

                search_summary_iter["num_rejected"] = len(rejected_obs_ids)
                logger.debug(
                    "Orbit fit failed. Removing newly added observations and adding "
                    "them to the rejected observations list."
                )

            # Skip to the next iteration since the best-fit orbit has not changed
            # since the previous iteration
            continue
        else:
            orbit_iter = orbit_iter_fit
            orbit_members_iter = orbit_members_iter_fit

        # Remove any potential outliers: observations already identified
        # by the OD code as outliers or any observations with a chi2
        # greater than the outlier_chi2
        outlier_mask = pc.or_(
            pc.equal(orbit_members_iter.outlier, True),
            pc.greater(orbit_members_iter.residuals.chi2, outlier_chi2),
        )
        outlier_ids = orbit_members_iter.apply_mask(outlier_mask).obs_id

        # Add identified outliers to the rejected observations list
        rejected_obs_ids.extend(outlier_ids.to_pylist())

        orbit_observations_iter = orbit_observations_iter.apply_mask(
            pc.invert(pc.is_in(orbit_observations_iter.id, outlier_ids))
        )
        orbit_members_iter = orbit_members_iter.apply_mask(
            pc.invert(pc.is_in(orbit_members_iter.obs_id, outlier_ids))
        )
        obs_ids_iter = orbit_observations_iter.id
        logger.debug(f"Found {len(outlier_ids)} outliers.")

        # Update the rest of the search summary
        search_summary_iter["num_accepted"] = len(obs_ids_iter)
        search_summary_iter["num_rejected"] = len(rejected_obs_ids)
        search_summary_iter["arc_length"] = orbit_iter.arc_length[0].as_py()
        search_summary_iter["num_obs"] = orbit_iter.num_obs[0].as_py()

        if (
            len(obs_ids_prev) > 0
            and pc.all(pc.is_in(obs_ids_iter, obs_ids_prev)).as_py()
        ):
            # If the observations have not changed since the previous iteration, lets
            # try increasing the search window and tolerance
            if min_mjd == min_mjd_iter and max_mjd == max_mjd_iter and i > 0:
                logger.debug(
                    "Observations have not changed since the previous iteration "
                    "and the search window and tolerance have not changed. Exiting."
                )

                search_summary = SearchSummary.from_kwargs(
                    **{k: [v] for k, v in search_summary_iter.items()}
                )
                return (
                    orbit_iter,
                    orbit_members_iter,
                    orbit_observations_iter,
                    search_summary,
                )

            else:
                tolerance_iter = update_tolerance(tolerance_iter)
                logger.debug(
                    "Observations have not changed since the previous iteration. "
                    f"Increasing the tolerance to {tolerance_iter}."
                )

    if i == max_iter - 1:
        logger.debug("Maximum number of iterations reached. Exiting.")

    if orbit_iter.reduced_chi2[0].as_py() > rchi2_threshold:
        logger.debug(
            f"Final reduced chi2 of {orbit_iter.reduced_chi2[0].as_py()} "
            f"is greater than the threshold of {rchi2_threshold}."
        )
        return (
            FittedOrbits.empty(),
            FittedOrbitMembers.empty(),
            OrbitDeterminationObservations.empty(),
            SearchSummary.empty(),
        )

    if len(orbit_observations_iter) == 0:
        logger.debug("No observations found. Exiting.")
        return (
            FittedOrbits.empty(),
            FittedOrbitMembers.empty(),
            OrbitDeterminationObservations.empty(),
            SearchSummary.empty(),
        )

    search_summary = SearchSummary.from_kwargs(
        **{k: [v] for k, v in search_summary_iter.items()}
    )
    return orbit_iter, orbit_members_iter, orbit_observations_iter, search_summary
