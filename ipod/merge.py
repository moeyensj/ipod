import logging
import multiprocessing as mp
import time
from typing import Optional, Tuple, Type

import numpy as np
import pyarrow.compute as pc
import quivr as qv
from adam_core.propagator import PYOORB, Propagator
from adam_core.ray_cluster import initialize_use_ray
from thor.observations import Observations, Photometry
from thor.observations.observations import calculate_state_id_hashes
from thor.orbit_determination.fitted_orbits import (
    FittedOrbitMembers,
    FittedOrbits,
    drop_duplicate_orbits,
)
from thor.orbits.od import differential_correction
from thor.utils.quivr import drop_duplicates

from .ipod import PrecoveryCandidates, SearchSummary
from .main import iterative_precovery_and_differential_correction
from .utils import assign_duplicate_observations

logger = logging.getLogger("ipod")


def merge_and_extend_orbits(
    orbits: FittedOrbits,
    orbit_members: Optional[FittedOrbitMembers] = None,
    observations: Optional[Observations] = None,
    max_tolerance: float = 10.0,
    delta_time: float = 30.0,
    rchi2_threshold: float = 3.0,
    outlier_chi2: float = 9.0,
    reconsider_chi2: float = 8.0,
    max_iter: int = 10,
    min_mjd: Optional[float] = None,
    max_mjd: Optional[float] = None,
    database_directory: str = "",
    datasets: Optional[set[str]] = None,
    propagator: Type[Propagator] = PYOORB,
    propagator_kwargs: dict = {},
    chunk_size: int = 100,
    max_processes: Optional[int] = 1,
) -> Tuple[FittedOrbits, FittedOrbitMembers, PrecoveryCandidates, SearchSummary]:
    time_start = time.perf_counter()
    logger.info("Running merge and extend...")

    orbits_out = FittedOrbits.empty()
    orbit_members_out = FittedOrbitMembers.empty()
    orbit_candidates_out = PrecoveryCandidates.empty()
    search_summary_out = SearchSummary.empty()

    if max_processes is None:
        max_processes = mp.cpu_count()

    # Initialize ray cluster or connect to an
    # existing ray cluster
    initialize_use_ray(num_cpus=max_processes)

    orbits_iter = orbits
    orbit_members_iter = orbit_members
    observations_iter = observations

    # Drop duplicate orbits (orbits with identical state vectors)
    num_orbits = len(orbits_iter)
    orbits_iter, orbit_members_iter = drop_duplicate_orbits(
        orbits_iter, orbit_members_iter
    )
    logger.info(
        f"Dropped {num_orbits - len(orbits_iter)} state vector duplicate orbits."
    )
    num_orbits = len(orbits_iter)

    iterations = 0
    while len(orbits_iter) > 0:
        logger.info(f"Starting iteration {iterations + 1} of {max_iter}.")

        chunk_size_iter = np.minimum(
            np.ceil(len(orbits_iter) / max_processes).astype(int), chunk_size
        )

        # Run iterative precovery and differential correction
        (
            orbits_iter,
            orbit_members_iter,
            precovery_candidates_iter,
            search_summary_iter,
        ) = iterative_precovery_and_differential_correction(
            orbits_iter,
            orbit_members=orbit_members_iter,
            observations=observations_iter,
            max_tolerance=max_tolerance,
            delta_time=delta_time,
            rchi2_threshold=rchi2_threshold,
            outlier_chi2=outlier_chi2,
            reconsider_chi2=reconsider_chi2,
            max_iter=10,
            min_mjd=min_mjd,
            max_mjd=max_mjd,
            database_directory=database_directory,
            datasets=datasets,
            propagator=propagator,
            propagator_kwargs=propagator_kwargs,
            chunk_size=chunk_size_iter,
            max_processes=max_processes,
        )

        orbits_iter, orbit_members_iter = assign_duplicate_observations(
            orbits_iter, orbit_members_iter
        )
        if orbits_iter.fragmented():
            orbits_iter = qv.defragment(orbits_iter)
        if orbit_members_iter.fragmented():
            orbit_members_iter = qv.defragment(orbit_members_iter)

        precovery_candidates_unique = drop_duplicates(
            precovery_candidates_iter, subset=["observation_id"]
        )
        if precovery_candidates_unique.fragmented():
            precovery_candidates_unique = qv.defragment(precovery_candidates_unique)

        if observations is None:
            coordinates = precovery_candidates_unique.to_spherical_coordinates()
            observations_iter = Observations.from_kwargs(
                id=precovery_candidates_unique.observation_id,
                exposure_id=precovery_candidates_unique.exposure_id,
                coordinates=coordinates,
                photometry=Photometry.from_kwargs(
                    mag=precovery_candidates_unique.mag,
                    mag_sigma=precovery_candidates_unique.mag_sigma,
                    filter=precovery_candidates_unique.filter,
                ),
                state_id=calculate_state_id_hashes(coordinates),
            )

        orbits_iter, orbit_members_iter = differential_correction(
            orbits_iter,
            orbit_members_iter,
            observations_iter,
            min_obs=6,
            min_arc_length=1.0,
            contamination_percentage=0.0,
            rchi2_threshold=reconsider_chi2,
            delta=1e-8,
            max_iter=5,
            method="central",
            propagator=propagator,
            propagator_kwargs=propagator_kwargs,
            chunk_size=chunk_size_iter,
            max_processes=max_processes,
        )

        # Filter the precovery candidates and search summary
        precovery_candidates_iter = precovery_candidates_iter.apply_mask(
            pc.is_in(precovery_candidates_iter.orbit_id, orbits_iter.orbit_id)
        )
        search_summary_iter = search_summary_iter.apply_mask(
            pc.is_in(search_summary_iter.orbit_id, orbits_iter.orbit_id)
        )

        # Identify orbits that have not had their differential correction converge to a new solution
        # and add them to the outgoing tables, also identify any orbits that have not
        # had any new observations added since the previous iteration
        mask = pc.equal(orbits_iter.success, False)
        if iterations > 0:
            orbit_ids = search_summary_iter.orbit_id.filter(
                pc.equal(search_summary_iter.num_obs_prev, search_summary_iter.num_obs)
            )
            logger.info(
                f"Identified {len(orbit_ids)} orbits that have not had any new observations "
                "added since the previous iteration."
            )
            if len(orbit_ids) > 0:
                mask = pc.or_(mask, pc.is_in(orbits_iter.orbit_id, orbit_ids))

        orbits_out_iter = orbits_iter.apply_mask(mask)
        orbit_members_out_iter = orbit_members_iter.apply_mask(
            pc.is_in(orbit_members_iter.orbit_id, orbits_out_iter.orbit_id)
        )
        orbit_candidates_out_iter = precovery_candidates_iter.apply_mask(
            pc.is_in(precovery_candidates_iter.orbit_id, orbits_out_iter.orbit_id)
        )
        search_summary_out_iter = search_summary_iter.apply_mask(
            pc.is_in(search_summary_iter.orbit_id, orbits_out_iter.orbit_id)
        )

        logger.info(
            f"{len(orbits_out_iter)} orbits have been removed from the active pool."
        )

        orbits_out = qv.concatenate([orbits_out, orbits_out_iter])
        if orbits_out.fragmented():
            orbits_out = qv.defragment(orbits_out)
        orbit_members_out = qv.concatenate([orbit_members_out, orbit_members_out_iter])
        if orbit_members_out.fragmented():
            orbit_members_out = qv.defragment(orbit_members_out)
        orbit_candidates_out = qv.concatenate(
            [orbit_candidates_out, orbit_candidates_out_iter]
        )
        if orbit_candidates_out.fragmented():
            orbit_candidates_out = qv.defragment(orbit_candidates_out)
        search_summary_out = qv.concatenate(
            [search_summary_out, search_summary_out_iter]
        )
        if search_summary_out.fragmented():
            search_summary_out = qv.defragment(search_summary_out)

        # Filter the orbits and orbit members to those we will continue to iterate on
        orbits_iter = orbits_iter.apply_mask(
            pc.invert(pc.is_in(orbits_iter.orbit_id, orbits_out_iter.orbit_id))
        )
        orbit_members_iter = orbit_members_iter.apply_mask(
            pc.is_in(orbit_members_iter.orbit_id, orbits_iter.orbit_id)
        )
        precovery_candidates_iter = precovery_candidates_iter.apply_mask(
            pc.is_in(precovery_candidates_iter.orbit_id, orbits_iter.orbit_id)
        )
        search_summary_iter = search_summary_iter.apply_mask(
            pc.is_in(search_summary_iter.orbit_id, orbits_iter.orbit_id)
        )

        logger.info(f"{len(orbits_iter)} orbits remain in the active pool.")

        iterations += 1
        if iterations == max_iter:
            logger.info(
                "{} orbits were still not successfully corrected after {} iterations.".format(
                    len(orbits_iter), max_iter
                )
            )
            orbits_out = qv.concatenate([orbits_out, orbits_iter])
            if orbits_out.fragmented():
                orbits_out = qv.defragment(orbits_out)
            orbit_members_out = qv.concatenate([orbit_members_out, orbit_members_iter])
            if orbit_members_out.fragmented():
                orbit_members_out = qv.defragment(orbit_members_out)
            orbit_candidates_out = qv.concatenate(
                [orbit_candidates_out, precovery_candidates_iter]
            )
            if orbit_candidates_out.fragmented():
                orbit_candidates_out = qv.defragment(orbit_candidates_out)
            search_summary_out = qv.concatenate(
                [search_summary_out, search_summary_iter]
            )
            if search_summary_out.fragmented():
                search_summary_out = qv.defragment(search_summary_out)

            break

    time_end = time.perf_counter()
    logger.info(
        f"Merged and extended {len(orbits)} orbits into {len(orbits_out)} orbits."
    )
    logger.info(f"Merge and extended completed in {time_end - time_start:.3f} seconds.")
    return orbits_out, orbit_members_out, orbit_candidates_out, search_summary_out
