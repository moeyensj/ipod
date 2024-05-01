import os
import shutil

import pyarrow as pa
import pytest
from adam_core.utils.helpers import make_observations, make_real_orbits
from precovery.ingest import index
from precovery.precovery_db import PrecoveryDatabase

OBJECT_IDS = [
    "594913 'Aylo'chaxnim (2020 AV2)",
    "163693 Atira (2003 CP20)",
    "(2010 TK7)",
    "3753 Cruithne (1986 TO)",
    "54509 YORP (2000 PH5)",
    "2063 Bacchus (1977 HB)",
    "1221 Amor (1932 EA1)",
    "433 Eros (A898 PA)",
    "3908 Nyx (1980 PA)",
    "434 Hungaria (A898 RB)",
    "1876 Napolitania (1970 BA)",
    "2001 Einstein (1973 EB)",
    "2 Pallas (A802 FA)",
    "6 Hebe (A847 NA)",
    "6522 Aci (1991 NQ)",
    "10297 Lynnejones (1988 RJ13)",
    "17032 Edlu (1999 FM9)",
    "202930 Ivezic (1998 SG172)",
    "911 Agamemnon (A919 FB)",
    "1143 Odysseus (1930 BH)",
    "1172 Aneas (1930 UA)",
    "3317 Paris (1984 KF)",
    "5145 Pholus (1992 AD)",
    "5335 Damocles (1991 DA)",
    "15760 Albion (1992 QB1)",
    "15788 (1993 SB)",
    "15789 (1993 SC)",
    "1I/'Oumuamua (A/2017 U1)",
]
TOLERANCES = {
    "default": 0.1 / 3600,
    "594913 'Aylo'chaxnim (2020 AV2)": 2 / 3600,
    "1I/'Oumuamua (A/2017 U1)": 5 / 3600,
}


@pytest.fixture
def observations():
    yield make_observations()


@pytest.fixture
def orbits():
    return make_real_orbits()


@pytest.fixture
def precovery_db(observations):

    exposures, detections, associations = observations
    exposures_df = exposures.to_dataframe()
    exposures_df["exposure_mjd_start"] = exposures.start_time.mjd().to_numpy()
    exposures_df["exposure_mjd_mid"] = exposures_df["exposure_mjd_start"] + (
        (exposures_df["duration"] / 86400) / 2
    )
    # getting issues with Rubin obscode plus pyoorb
    exposures_df = exposures_df[exposures_df["observatory_code"] != "X05"]

    detections_df = detections.to_dataframe()
    detections_df["mjd"] = detections.time.mjd().to_numpy()
    # join on exposure_id
    observations_df = detections_df.merge(
        exposures_df, left_on="exposure_id", right_on="id"
    )
    print(observations_df.columns)

    observations_df["obs_id"] = observations_df["id_x"]
    observations_df["dataset_id"] = "test"
    observations_df.rename(columns={"duration": "exposure_duration"}, inplace=True)

    # Add some offset data/noise to exposures, detections, associations
    # ex concat a copy of the observation off by half arcsec
    # precovery will return multiple obs at a single frame, could try both ways (offset obs, old+offset)

    # Save to csv in a dataset directory for indexing
    data_dir = os.path.join(os.path.dirname(__file__), "data/index/test/")

    os.makedirs(data_dir, exist_ok=True)
    observations_df.to_csv(
        os.path.join(data_dir, "observations.csv"),
        index=False,
        columns=[
            "obs_id",
            "mjd",
            "ra",
            "dec",
            "ra_sigma",
            "dec_sigma",
            "mag_sigma",
            "filter",
            "mag",
            "observatory_code",
            "exposure_mjd_start",
            "exposure_mjd_mid",
            "exposure_duration",
            "dataset_id",
            "exposure_id",
        ],
    )

    test_db_dir = os.path.join(os.path.dirname(__file__), "data/db/")
    index(
        out_dir=test_db_dir,
        dataset_id="test",
        dataset_name="test",
        data_dir=data_dir,
        nside=16,
    )
    yield PrecoveryDatabase.from_dir(str(test_db_dir))
    shutil.rmtree(os.path.join(os.path.dirname(__file__), "data"))


def test_import():
    # flake8: noqa
    from ipod import iterative_precovery_and_differential_correction


def test_index(precovery_db, orbits):
    db = precovery_db
    # precover one of our orbits
    orbit = orbits[0]
    matches = [m for m in db.precover(orbit)]
    assert len(matches) > 0
    print(matches)
