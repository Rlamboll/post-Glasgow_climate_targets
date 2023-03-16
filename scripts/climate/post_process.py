import json
import logging
import os.path
from functools import lru_cache

import numpy as np
import numpy.testing as npt
import pandas as pd
import scmdata
import scmdata.database
import scmdata.processing
from pint.errors import DimensionalityError

from .ciceroscm import ciceroscm_post_process
from .fair import fair_post_process
from .magicc7 import calculate_co2_and_nonco2_warming_magicc, magicc7_post_process

LOGGER = logging.getLogger(__name__)
_CLIMATE_VARIABLE_DEFINITION_CSV = os.path.join(
    os.path.dirname(__file__), "variable_definitions.csv"
)


@lru_cache()
def _get_climate_variable_definitions(fname):
    return pd.read_csv(fname)


def convert_openscm_runner_variables_to_ar6_wg3_variables(in_var):
    mapping = {
        "Surface Air Temperature Change": "Raw Surface Temperature (GSAT)",
        "Surface Air Ocean Blended Temperature Change": "Raw Surface Temperature (GMST)",
        "Heat Uptake|Ocean": "Ocean Heat Uptake",
        "Effective Radiative Forcing|Aerosols|Direct Effect|SOx": "Effective Radiative Forcing|Aerosols|Direct Effect|Sulfur",
    }

    try:
        return mapping[in_var]
    except KeyError:
        return in_var


def calculate_exceedance_probability_timeseries(
    res, exceedance_probability_calculation_var, test_run=False
):
    """
    Calculate the timeseries with which we should determine exceedance probabilities
    """
    HIST_TEMP_REF_PERIOD = range(1850, 1900 + 1)
    HIST_TEMP_EVALUATION_PERIOD = range(1995, 2014 + 1)
    HIST_TEMP_ASSESSMENT_CENTRAL = 0.85
    HIST_TEMP_ASSESSMENT_CENTRAL_UNIT = "K"
    LOGGER.info(
        "Adjusting median of %s-%s warming (rel. to %s-%s) to %s%s",
        HIST_TEMP_EVALUATION_PERIOD[0],
        HIST_TEMP_EVALUATION_PERIOD[-1],
        HIST_TEMP_REF_PERIOD[0],
        HIST_TEMP_REF_PERIOD[-1],
        HIST_TEMP_ASSESSMENT_CENTRAL,
        HIST_TEMP_ASSESSMENT_CENTRAL_UNIT,
    )

    exceedance_probability_timeseries_raw = res.filter(
        variable="Raw {}".format(exceedance_probability_calculation_var),
        unit=HIST_TEMP_ASSESSMENT_CENTRAL_UNIT,
    )
    exceedance_probability_timeseries = (
        exceedance_probability_timeseries_raw.relative_to_ref_period_mean(
            year=HIST_TEMP_REF_PERIOD
        )
        .drop_meta(["reference_period_start_year", "reference_period_end_year"])
        .adjust_median_to_target(
            HIST_TEMP_ASSESSMENT_CENTRAL,
            HIST_TEMP_EVALUATION_PERIOD,
            process_over=("run_id",),
        )
    )

    # output checks
    hist_temp_grouper_cols = exceedance_probability_timeseries.get_meta_columns_except(
        "run_id"
    )

    def _get_median_hist_warming(inp):
        return (
            inp.filter(year=HIST_TEMP_EVALUATION_PERIOD)
            .timeseries()
            .mean(axis="columns")
            .groupby(hist_temp_grouper_cols)
            .median()
        )

    inp_vals = _get_median_hist_warming(exceedance_probability_timeseries_raw)
    check_vals = _get_median_hist_warming(exceedance_probability_timeseries)
    shifts = check_vals - inp_vals

    if not np.isclose(shifts, shifts[0], atol=5 * 1e-3).all():
        LOGGER.exception(
            "Careful of scenarios which break match with history! " "`shifts`: %s",
            shifts,
        )
    else:
        LOGGER.info("`shifts`: %s", shifts)

    if not test_run:
        try:
            npt.assert_allclose(
                check_vals,
                HIST_TEMP_ASSESSMENT_CENTRAL,
                rtol=1e-2,  # to within 1%
                err_msg="{}".format(check_vals),
            )
        except AssertionError:
            LOGGER.exception("Careful of scenarios which break match with history!")

    exceedance_probability_timeseries[
        "variable"
    ] = exceedance_probability_calculation_var
    return exceedance_probability_timeseries


def calculate_co2_and_nonco2_warming_and_remove_extras(res):
    """
    Calculate non-CO2 warming
    """
    out = []
    for cmrun in res.groupby("climate_model"):
        climate_model = cmrun.get_unique_meta("climate_model", no_duplicates=True)
        if climate_model.startswith("MAGICC"):
            all_forcers_run = cmrun.filter(rf_total_runmodus="ALL").drop_meta(
                "rf_total_runmodus"
            )
            out.append(all_forcers_run)
            out.append(calculate_co2_and_nonco2_warming_magicc(cmrun))
        else:
            raise NotImplementedError(climate_model)

    return scmdata.run_append(out)


def post_process(
    res,
    outdir,
    test_run=False,
    save_raw_output=False,
    co2_and_non_co2_warming=False,
    # for exceedance probability calculations
    temp_thresholds=(1.5, 2.0, 2.5, 3.0),
    peak_percentiles=(33, 50, 67),
):
    LOGGER.info("Beginning climate post-processing")
    LOGGER.info("Removing unknown units and keeping only World data")
    res = res.filter(unit="unknown", keep=False).filter(region="World")

    LOGGER.info(
        "Renaming variables from OpenSCM-Runner conventions to AR6 WG3 conventions"
    )
    res["variable"] = res["variable"].apply(
        convert_openscm_runner_variables_to_ar6_wg3_variables
    )

    LOGGER.info("Performing climate model specific fixes")
    all_res = []
    for res_cm in res.groupby("climate_model"):
        climate_model = res_cm.get_unique_meta("climate_model", no_duplicates=True)
        LOGGER.info("Processing %s data", climate_model)

        if climate_model.startswith("MAGICCv7"):
            res_cm = magicc7_post_process(res_cm)

        if climate_model.startswith("FaIRv1.6"):
            res_cm = fair_post_process(res_cm)

        if climate_model.startswith("CICERO-SCM"):
            res_cm = ciceroscm_post_process(res_cm)

        all_res.append(res_cm)

    LOGGER.info("Recombining post-processed data")
    res = scmdata.run_append(all_res)

    def _rename_vars(v):
        mapping = {
            "Effective Radiative Forcing|Greenhouse Gases": "Effective Radiative Forcing|Basket|Greenhouse Gases",
            "Effective Radiative Forcing|Anthropogenic": "Effective Radiative Forcing|Basket|Anthropogenic",
        }

        try:
            out = mapping[v]
            LOGGER.info("Renaming %s to %s", v, out)
            return out
        except KeyError:
            LOGGER.debug("Not renaming %s", v)
            return v

    LOGGER.info("Performing further variable renaming")
    res["variable"] = res["variable"].apply(_rename_vars)

    if save_raw_output:
        LOGGER.info("Saving raw output (with renamed variables) to disk")
        if "parameters" in res.metadata:
            res.metadata["parameters"] = json.dumps(res.metadata["parameters"])

        database = scmdata.database.ScmDatabase(
            os.path.join(outdir, "raw_climate_output"),
            levels=["climate_model", "variable", "model", "scenario"],
        )
        for c in [
            "climate_model",
            "model",
            "scenario",
            "variable",
            "region",
            "rf_total_runmodus",
        ]:
            if c in res.meta:
                res[c] = res[c].astype(str)

        database.save(res)

    if co2_and_non_co2_warming:
        LOGGER.info("Calculating non-CO2 warming")
        res = calculate_co2_and_nonco2_warming_and_remove_extras(res)

    LOGGER.info("Calculating exceedance probability timeseries")
    exceedance_probability_calculation_var = "Surface Temperature (GSAT)"
    exceedance_probability_timeseries = calculate_exceedance_probability_timeseries(
        res, exceedance_probability_calculation_var, test_run=test_run
    )
    res = res.append(exceedance_probability_timeseries)

    year_filter = range(1995, 2101)
    LOGGER.info("Keeping only data from %s", year_filter)
    res = res.filter(year=year_filter)

    LOGGER.info("Calculating Non-CO2 GHG ERF")
    helper = res.filter(variable="Effective Radiative Forcing*")
    res = [res]

    erf_nonco2_ghg = helper.filter(
        variable="Effective Radiative Forcing|Basket|Greenhouse Gases"
    ).subtract(
        helper.filter(variable="Effective Radiative Forcing|CO2"),
        op_cols={
            "variable": "Effective Radiative Forcing|Basket|Non-CO2 Greenhouse Gases"
        },
    )

    if (
        erf_nonco2_ghg.get_unique_meta("unit", no_duplicates=True)
        != "watt / meter ** 2"
    ):
        raise AssertionError("Unexpected forcing unit")

    erf_nonco2_ghg["unit"] = "W/m^2"
    res.append(erf_nonco2_ghg)

    LOGGER.info("Calculating Non-CO2 Anthropogenic ERF")
    erf_nonco2_anthropogenic = helper.filter(
        variable="Effective Radiative Forcing|Basket|Anthropogenic"
    ).subtract(
        helper.filter(variable="Effective Radiative Forcing|CO2"),
        op_cols={
            "variable": "Effective Radiative Forcing|Basket|Non-CO2 Anthropogenic"
        },
    )
    if (
        erf_nonco2_anthropogenic.get_unique_meta("unit", no_duplicates=True)
        != "watt / meter ** 2"
    ):
        raise AssertionError("Unexpected forcing unit")

    erf_nonco2_anthropogenic["unit"] = "W/m^2"
    res.append(erf_nonco2_anthropogenic)

    LOGGER.info("Joining derived variables and data back together")
    res = scmdata.run_append(res)

    # check all variable names
    LOGGER.info("Converting all variable names and units to standard definitions")

    def _convert_to_standard_name_and_unit(vdf):
        climate_variable_definitions = _get_climate_variable_definitions(
            _CLIMATE_VARIABLE_DEFINITION_CSV
        )
        variable = vdf.get_unique_meta("variable", True)

        try:
            standard_unit = climate_variable_definitions.set_index("Variable").loc[
                variable
            ]["Unit"]
        except KeyError as exc:
            raise ValueError(
                "{} not in {}".format(variable, _CLIMATE_VARIABLE_DEFINITION_CSV)
            ) from exc
        try:
            return vdf.convert_unit(standard_unit)
        except DimensionalityError as exc:
            raise ValueError(
                "Cannot convert {} units of {} to {}".format(
                    variable, vdf.get_unique_meta("unit", True), standard_unit
                )
            ) from exc

    res = res.groupby("variable").map(_convert_to_standard_name_and_unit)

    LOGGER.info("Calculating percentiles")
    percentiles = [5, 10, 1 / 6 * 100, 33, 50, 67, 5 / 6 * 100, 90, 95]
    res_percentiles = res.quantiles_over(
        "run_id", np.array(percentiles) / 100
    ).reset_index()
    res_percentiles["percentile"] = res_percentiles["quantile"] * 100
    res_percentiles = res_percentiles.drop("quantile", axis="columns")

    LOGGER.info("Mangling variable name with climate model and percentile")
    res_percentiles["variable"] = (
        res_percentiles["variable"].astype(str)
        + "|"
        + res_percentiles["climate_model"].astype(str)
        + "|"
        + res_percentiles["percentile"].astype(float).round(1).astype(str)
        + "th Percentile"
    )
    res_percentiles = scmdata.ScmRun(
        res_percentiles.drop(["climate_model", "percentile"], axis="columns")
    )

    LOGGER.info(
        "Calculating exceedance probabilities and exceedance probability timeseries"
    )
    res_exceedance_probs_var = res.filter(
        variable=exceedance_probability_calculation_var
    )
    exceedance_probs_by_temp_threshold = []
    exceedance_probs_tss = []
    for threshold in temp_thresholds:
        LOGGER.info("Calculating %s exceedance probabilities", threshold)
        for store, func, output_name in (
            (
                exceedance_probs_by_temp_threshold,
                scmdata.processing.calculate_exceedance_probabilities,
                "Exceedance Probability {}C".format(threshold),
            ),
            (
                exceedance_probs_tss,
                scmdata.processing.calculate_exceedance_probabilities_over_time,
                "Exceedance Probability {}C".format(threshold),
            ),
        ):
            store.append(
                func(
                    res_exceedance_probs_var,
                    threshold,
                    process_over_cols=("run_id",),
                    output_name=output_name,
                )
            )

    exceedance_probs_tss = pd.concat(exceedance_probs_tss).reset_index()
    exceedance_probs_tss["variable"] = (
        exceedance_probs_tss["variable"].astype(str)
        + "|"
        + exceedance_probs_tss["climate_model"].astype(str)
    )
    exceedance_probs_tss = exceedance_probs_tss.drop("climate_model", axis="columns")
    res_percentiles = res_percentiles.append(exceedance_probs_tss)

    reporting_groups = ["climate_model", "model", "scenario"]
    exceedance_probs_by_temp_threshold = pd.concat(
        exceedance_probs_by_temp_threshold, axis=1
    )
    exceedance_probs_by_temp_threshold = exceedance_probs_by_temp_threshold.reset_index(
        list(
            set(exceedance_probs_by_temp_threshold.index.names) - set(reporting_groups)
        ),
        drop=True,
    )

    LOGGER.info("Calculating peak warming and peak warming year")
    peaks = scmdata.processing.calculate_peak(res_exceedance_probs_var)
    peak_years = scmdata.processing.calculate_peak_time(res_exceedance_probs_var)

    def _get_quantiles(idf):
        return (
            idf.groupby(reporting_groups)
            .quantile(np.array(peak_percentiles) / 100)
            .unstack()
        )

    peaks_quantiles = _get_quantiles(peaks)
    peak_years_quantiles = _get_quantiles(peak_years).astype(int)

    def rename_quantiles(quantile):
        percentile = int(quantile * 100)
        if np.isclose(percentile, 50):
            plabel = "median"
        else:
            plabel = "p{}".format(percentile)

        return plabel

    peaks_quantiles.columns = (
        peaks_quantiles.columns.map(rename_quantiles).astype(str) + " peak warming"
    )
    peak_years_quantiles.columns = (
        peak_years_quantiles.columns.map(rename_quantiles).astype(str)
        + " year of peak warming"
    )

    LOGGER.info("Creating meta table")
    meta_table = pd.concat(
        [
            exceedance_probs_by_temp_threshold,
            peaks_quantiles,
            peak_years_quantiles,
        ],
        axis=1,
    )

    def mangle_meta_table_climate_model(idf):
        out = idf.copy()
        climate_model = idf.name
        out.columns = out.columns + " ({})".format(climate_model)

        return out

    meta_table = (
        meta_table.groupby("climate_model")
        .apply(mangle_meta_table_climate_model)
        .reset_index("climate_model", drop=True)
    )

    LOGGER.info("Exiting post-processing")
    return res, res_percentiles, meta_table
