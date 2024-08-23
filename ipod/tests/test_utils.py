from thor.orbit_determination.fitted_orbits import FittedOrbitMembers

from ipod.utils import ExpectedMembers, analyze_me_output


def test_me_analysis():

    # Initialize the expected members
    data_expected = {
        "orbit_id": ["orbit1", "orbit1", "orbit2", "orbit2"],
        "obs_id": ["obs1", "obs2", "obs3", "obs4"],
        "primary_designation": [
            "designation1",
            "designation1",
            "designation2",
            "designation2",
        ],
    }

    expected_members = ExpectedMembers.from_kwargs(
        orbit_id=data_expected["orbit_id"],
        obs_id=data_expected["obs_id"],
        primary_designation=data_expected["primary_designation"],
    )

    # Initialize the initial members
    data_initial = {
        "orbit_id": ["orbit1", "orbit2", "orbit2"],
        "obs_id": ["obs1", "obs3", "obs4"],
    }
    initial_members = FittedOrbitMembers.from_kwargs(
        orbit_id=data_initial["orbit_id"], obs_id=data_initial["obs_id"]
    )

    # Initialize the ME output members
    data_me = {
        "orbit_id": ["orbit1", "orbit1", "orbit3"],
        "obs_id": ["obs1", "obs2", "obs5"],
    }
    me_members = FittedOrbitMembers.from_kwargs(
        orbit_id=data_me["orbit_id"], obs_id=data_me["obs_id"]
    )

    # Call the function with the test data
    result = analyze_me_output(expected_members, initial_members, me_members)

    # Check the results using asserts
    assert result.loc["designation1", "num_missing_obs"] == 0
    assert result.loc["designation1", "num_extra_obs"] == 0
    assert result.loc["designation1", "num_bogus_obs"] == 0
    assert result.loc["designation1", "num_initial_orbits_with_attributed_members"] == 1
    assert result.loc["designation1", "num_result_orbits_with_attributed_members"] == 1
    assert result.loc["designation1", "best_result_orbit_id"] == "orbit1"
    assert result.loc["designation1", "initial_orbits_with_attributed_members"] == [
        "orbit1"
    ]
    assert result.loc["designation1", "result_orbits_with_attributed_members"] == [
        "orbit1",
    ]

    assert result.loc["designation2", "num_missing_obs"] == 2
    assert result.loc["designation2", "num_extra_obs"] == 0
    assert result.loc["designation2", "num_bogus_obs"] == 0
    assert result.loc["designation2", "num_initial_orbits_with_attributed_members"] == 1
    assert result.loc["designation2", "num_result_orbits_with_attributed_members"] == 0
    assert result.loc["designation2", "best_result_orbit_id"] == "None"
    assert result.loc["designation2", "initial_orbits_with_attributed_members"] == [
        "orbit2"
    ]
    assert result.loc["designation2", "result_orbits_with_attributed_members"] == []
