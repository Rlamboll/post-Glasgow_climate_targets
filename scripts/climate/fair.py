import json
import logging

from openscm_runner.adapters import FAIR

LOGGER = logging.getLogger(__name__)
DEFAULT_FAIR_VERSION = "1.6.2"


def get_fair_configurations(fair_version, fair_probabilistic_file, num_cfgs):
    """
    Get configuration for FaIR
    """

    if FAIR.get_version() != fair_version:
        raise AssertionError(FAIR.get_version())

    with open(fair_probabilistic_file, "r") as fh:
        cfgs_raw = json.load(fh)

    fair_cfgs = [
        {
            "run_id": i,
            **c,
        }
        for i, c in enumerate(cfgs_raw[:num_cfgs])
    ]

    return fair_cfgs


def fair_post_process(climate_output):
    # convert units to W/m^2
    climate_output = climate_output.convert_unit(
        "W/m^2", variable="Effective Radiative Forcing*"
    )

    return climate_output
