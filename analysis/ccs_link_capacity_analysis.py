"""Compares observed CCS hourly volumes against the TDM's own spdcap capacity
assumptions, by facility type (FT).

For each CCS station and each year in YEAR_WORKBOOKS, finds the max volume
recorded in a single lane, single direction, within that year's Sept 1 - Nov
15 (Tue-Thu) calibration window, from the raw hourly-by-lane workbooks in
data/. Also synthesizes an "all" pseudo-year per station: whichever year had
that station's single highest max (station-matching and link geometry don't
change by year, so this is a cheap re-use of the per-year results rather than
a separate pass). Matches each station to its nearest link on the filtered
TDM network (centroid connectors, ramps, and HOV links excluded) to get that
link's FT, lane count, and CAP1HR1LN (the model's per-lane-per-hour capacity
assumption, itself sourced from data/Lookup_Speed_Capacity.csv by FT + lane
count). Flags stations inside a mountainous TAZ (MARGINAL=1).

Three station sets are excluded from the final tables (not from the raw
per-station output, which keeps everything for inspection):
  - OUTLIER_STATIONS: >150% of assumed capacity even within the calibration
    window -- implausible single-hour readings, likely sensor/data issues.
  - BANGERTER_EXPRESSWAY_STATIONS: Bangerter Hwy stations currently coded
    FT 13/15 (Expressway). Bangerter is being converted to freeway, so its
    current FT may not represent it well. -636 is already FT 32 (Freeway)
    and isn't in this list.
  - Stations inside a MARGINAL=1 TAZ (mountainous canyon zones) -- excluded
    dynamically via the spatial join below, not a hardcoded list.

All data under data/ is a static snapshot copied from the WF-TDM-Calibration
repo (see README.md for provenance/regeneration notes) -- this script has no
dependency on that repo, its tdm/ submodule, or Cube Voyager.

Usage (from repo root):
    .venv/Scripts/python.exe analysis/ccs_link_capacity_analysis.py
"""

import json
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "output"
DASHBOARD_DATA_DIR = REPO_ROOT / "docs" / "data"

# Static snapshot of the TDM's scenario network links (calibration run C74,
# tdm_ref calib/C74 -- see README.md for provenance). Trimmed to the columns
# this script actually needs; not auto-refreshed if C74 is ever re-run.
LINK_GPKG = DATA_DIR / "tdm_scenario_C74_links.gpkg"
TAZ_SHP = DATA_DIR / "taz" / "WFv1000_TAZ.shp"
REGION_GPKG = DATA_DIR / "RegionalBoundary.gpkg"
CCS_LOCATIONS = DATA_DIR / "SiteData.csv"
CCS_LOCATIONS_SUPPLEMENT = DATA_DIR / "SiteData_supplement.csv"
SPDCAP_LOOKUP = DATA_DIR / "Lookup_Speed_Capacity.csv"

# One or more hourly-by-lane workbooks per year -- same sheet-per-station
# layout (STATION, DATE, ROUTE, MP, LANE, H0000..H2300) in every year so far;
# add a year here once its workbook(s) land in data/.
YEAR_WORKBOOKS = {
    2023: [DATA_DIR / "2023 Station 301-431.xlsx", DATA_DIR / "2023 Station 501-733.xlsx"],
    2024: [DATA_DIR / "2024 Station 301-513.xlsx", DATA_DIR / "2024 Station 601-851.xlsx"],
    2025: [DATA_DIR / "2025 Station 301-609.xlsx", DATA_DIR / "2025 Station 611-892.xlsx"],
}

CRS_UTM = 26912  # NAD83 / UTM zone 12N
HOUR_COLS = [f"H{str(h).zfill(2)}00" for h in range(24)]

# Same Sept 1 - Nov 15, Tue-Thu window (per year) WF-TDM-Calibration's report/
# 4-assignhwy.qmd uses for its own CCS "Target" volumes -- keeps the 2023
# calibration-window max directly comparable to what that report calls
# Target. Applying the same relative window to every other year keeps
# year-over-year comparisons on the same seasonal footing.
CALIB_WEEKDAYS = [1, 2, 3]  # Tue, Wed, Thu


def calib_window(year: int) -> tuple[str, str]:
    return f"{year}-09-01", f"{year}-11-15"

# Stations manually dropped -- see module docstring for why.
OUTLIER_STATIONS = ["-635", "-665", "-319", "-306", "-626"]
BANGERTER_EXPRESSWAY_STATIONS = ["-627", "-711", "-819", "-820"]

# Hand-removed from the CCS bridge elsewhere in WF-TDM-Calibration's report
# pipeline (ramp/interchange stations, not representative of a through-lane
# count).
DROP_SITES = ["-680", "-816", "-302"]


def load_ccs_site_locations() -> gpd.GeoDataFrame:
    df_sites = pd.read_csv(CCS_LOCATIONS)
    df_sites_supp = pd.read_csv(CCS_LOCATIONS_SUPPLEMENT)
    df_sites = pd.concat([df_sites, df_sites_supp], ignore_index=True).drop_duplicates(
        subset="SITE", keep="first"
    )

    gdf_region = (
        gpd.read_file(REGION_GPKG, layer="RegionalBoundaryComponents", where="PlanOrg IN ('MAG MPO', 'WFRC MPO')")
        .dissolve()
        .to_crs(CRS_UTM)
    )
    gdf_region["geometry"] = gdf_region.buffer(1).buffer(-1).buffer(16093.4)  # 10mi buffer

    gdf_ccs = (
        gpd.GeoDataFrame(df_sites, geometry=gpd.points_from_xy(df_sites.LONGITUDE, df_sites.LATITUDE), crs="EPSG:4326")
        .to_crs(CRS_UTM)
        .clip(gdf_region)
    )
    gdf_ccs["SITE"] = gdf_ccs["SITE"].astype(str)
    gdf_ccs = gdf_ccs[~gdf_ccs["SITE"].isin(DROP_SITES)].reset_index(drop=True)
    return gdf_ccs


def load_filtered_links() -> gpd.GeoDataFrame:
    """Network links with centroid connectors (FT 1), ramps (FT 28/29/41/42),
    and HOV links (STREET contains "HOV") removed."""
    gdf_links = gpd.read_file(LINK_GPKG).to_crs(CRS_UTM)
    is_centroid_connector = gdf_links["FT"] == 1
    is_ramp = gdf_links["FT"].isin([28, 29, 41, 42])
    is_hov = gdf_links["STREET"].str.contains("HOV", case=False, na=False)
    return gdf_links[~is_centroid_connector & ~is_ramp & ~is_hov].reset_index(drop=True)


def match_stations_to_links(gdf_ccs: gpd.GeoDataFrame, gdf_links: gpd.GeoDataFrame) -> pd.DataFrame:
    """Nearest-link match by geometry -- the link shapefile has no clean
    route/milepost field to match on (unlike the HPMS segment shapefile), so
    this uses straight-line nearest distance instead."""
    matches = []
    for _, row in gdf_ccs.iterrows():
        nearest_idx = gdf_links.distance(row.geometry).idxmin()
        link = gdf_links.loc[nearest_idx]
        matches.append(
            {
                "SITE": row["SITE"],
                "SITE_DESCRIPTION": row.get("SITE_DESCRIPTION"),
                "LATITUDE": row["LATITUDE"],
                "LONGITUDE": row["LONGITUDE"],
                "LINK_STREET": link["STREET"],
                "LINKID": link["LINKID"],
                "LINK_FT": link["FT"],
                "LINK_FTCLASS": link["FTCLASS"],
                "LANES": link["LANES"],
                "CAP1HR1LN": link["CAP1HR1LN"],
                "DIST_M": row.geometry.distance(link.geometry),
            }
        )
    return pd.DataFrame(matches)


def flag_mountainous_stations(gdf_ccs: gpd.GeoDataFrame) -> pd.DataFrame:
    """Flags each station as inside a mountainous (MARGINAL=1) TAZ zone."""
    taz = gpd.read_file(TAZ_SHP)[["TAZID", "MARGINAL", "geometry"]]
    marginal_taz = taz[taz["MARGINAL"] == 1]
    joined = gpd.sjoin(gdf_ccs[["SITE", "geometry"]], marginal_taz, how="left", predicate="within")
    joined["IS_MARGINAL"] = joined["TAZID"].notna()
    return joined[["SITE", "IS_MARGINAL"]].drop_duplicates(subset="SITE")


def max_hourly_volume_by_station(sites: list[str], year: int) -> pd.DataFrame:
    """Max volume in one lane, one direction, within that year's calibration
    window, per station -- from that year's raw hourly-by-lane workbooks."""
    window_start, window_end = calib_window(year)
    results = []
    for workbook in YEAR_WORKBOOKS[year]:
        all_sheets = gpd_safe_read_excel_sheet_names(workbook)
        sheet_to_site = {s.lstrip("-").zfill(4): s for s in sites if s.lstrip("-").zfill(4) in all_sheets}
        if not sheet_to_site:
            continue
        loaded = pd.read_excel(workbook, sheet_name=list(sheet_to_site.keys()))
        for sheet, site in sheet_to_site.items():
            df = loaded[sheet].copy()
            df["DATE"] = pd.to_datetime(df["DATE"])
            mask_date = (df["DATE"] >= window_start) & (df["DATE"] <= window_end)
            mask_weekday = df["DATE"].dt.weekday.isin(CALIB_WEEKDAYS)
            df_window = df[mask_date & mask_weekday]
            if df_window.empty:
                results.append({"SITE": site, "MAX_HOURLY_VOL": None, "LANE": None, "DIRECTION": None, "DATE": None, "HOUR": None})
                continue
            long = df_window.melt(id_vars=["STATION", "DATE", "ROUTE", "MP", "LANE"], value_vars=HOUR_COLS, var_name="HOUR", value_name="VOL")
            best = long.loc[long["VOL"].idxmax()]
            results.append(
                {
                    "SITE": site,
                    "MAX_HOURLY_VOL": best["VOL"],
                    "LANE": best["LANE"],
                    "DIRECTION": "N" if best["LANE"][0] == "N" else "P",
                    "DATE": best["DATE"].strftime("%Y-%m-%d"),
                    "HOUR": best["HOUR"],
                }
            )
    return pd.DataFrame(results)


def gpd_safe_read_excel_sheet_names(workbook: Path) -> list[str]:
    import openpyxl

    wb = openpyxl.load_workbook(workbook, read_only=True)
    try:
        return wb.sheetnames
    finally:
        wb.close()


def spdcap_lookup_table() -> pd.DataFrame:
    lut = pd.read_csv(SPDCAP_LOOKUP)
    lut.columns = [c.lstrip(";") for c in lut.columns]
    return lut


def build_final_table() -> pd.DataFrame:
    """One row per (station, year) for every year in YEAR_WORKBOOKS, plus a
    synthesized YEAR="all" row per station carrying whichever year had that
    station's single highest max (station-matching and link geometry are
    year-independent, so this re-uses the per-year merges above rather than
    a separate pass)."""
    gdf_ccs = load_ccs_site_locations()
    gdf_links = load_filtered_links()

    link_match = match_stations_to_links(gdf_ccs, gdf_links)
    marginal_flag = flag_mountainous_stations(gdf_ccs)
    sites = gdf_ccs["SITE"].tolist()

    year_frames = []
    for year in YEAR_WORKBOOKS:
        volumes = max_hourly_volume_by_station(sites, year)
        df = link_match.merge(marginal_flag, on="SITE", how="left").merge(volumes, on="SITE", how="left")
        df["YEAR"] = year
        df["PCT_OF_CAP"] = (df["MAX_HOURLY_VOL"] / df["CAP1HR1LN"] * 100).round(0)
        year_frames.append(df)

    all_years = pd.concat(year_frames, ignore_index=True)
    return pd.concat([all_years, all_time_max_slice(all_years)], ignore_index=True)


def all_time_max_slice(all_years: pd.DataFrame) -> pd.DataFrame:
    """One row per station: the (station, year) row with that station's
    highest MAX_HOURLY_VOL across every year, tagged YEAR="all". Stations
    with no volume in any year keep one row (any year) so they still appear
    on the "all years" map/list, just with a null MAX_HOURLY_VOL."""
    with_vol = all_years.dropna(subset=["MAX_HOURLY_VOL"])
    best = all_years.loc[with_vol.groupby("SITE")["MAX_HOURLY_VOL"].idxmax()].copy()

    sites_without_vol = set(all_years["SITE"]) - set(best["SITE"])
    if sites_without_vol:
        filler = all_years[all_years["SITE"].isin(sites_without_vol)].drop_duplicates(subset="SITE", keep="first")
        best = pd.concat([best, filler], ignore_index=True)

    best["YEAR"] = "all"
    return best


def apply_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    excluded = OUTLIER_STATIONS + BANGERTER_EXPRESSWAY_STATIONS
    out = df[~df["SITE"].isin(excluded) & ~df["IS_MARGINAL"].fillna(False)].copy()
    out["_year_sort"] = out["YEAR"].apply(lambda y: 9999 if y == "all" else y)
    out = out.sort_values(["LINK_FT", "SITE", "_year_sort"]).drop(columns=["_year_sort"])
    return out.reset_index(drop=True)


def ft_summary(df: pd.DataFrame) -> pd.DataFrame:
    lut = spdcap_lookup_table()
    cap_cols = [c for c in lut.columns if c.startswith("CAPLN")]
    lut_slim = lut[["FT"] + cap_cols]

    grp = df.groupby(["LINK_FT", "LINK_FTCLASS"]).agg(
        n_stations=("SITE", "count"),
        n_with_volume=("MAX_HOURLY_VOL", lambda s: s.notna().sum()),
        max_of_max=("MAX_HOURLY_VOL", "max"),
        avg_of_max=("MAX_HOURLY_VOL", "mean"),
    ).reset_index()
    grp["avg_of_max"] = grp["avg_of_max"].round(0)
    return grp.merge(lut_slim, left_on="LINK_FT", right_on="FT", how="left").drop(columns=["FT"])


def lane_breakdown_for_ft(df: pd.DataFrame, ft: int) -> pd.DataFrame:
    """Same max/avg-of-max breakdown as ft_summary(), but split out by lane
    count within one FT -- useful for seeing whether the model's
    lane-count-dependent capacity actually tracks observed volume by lane."""
    lut = spdcap_lookup_table()
    cap_row = lut[lut["FT"] == ft].iloc[0]

    subset = df[df["LINK_FT"] == ft]
    grp = subset.groupby("LANES").agg(
        n_stations=("SITE", "count"),
        n_with_volume=("MAX_HOURLY_VOL", lambda s: s.notna().sum()),
        max_of_max=("MAX_HOURLY_VOL", "max"),
        avg_of_max=("MAX_HOURLY_VOL", "mean"),
    ).reset_index()
    grp["avg_of_max"] = grp["avg_of_max"].round(0)
    grp["model_cap_per_lane"] = grp["LANES"].apply(lambda lanes: cap_row[f"CAPLN{min(int(lanes), 7)}"])
    grp["pct_of_cap_at_max"] = (grp["max_of_max"] / grp["model_cap_per_lane"] * 100).round(0)
    grp["pct_of_cap_at_avg"] = (grp["avg_of_max"] / grp["model_cap_per_lane"] * 100).round(0)
    return grp


def capacity_lookup_for_dashboard(final: pd.DataFrame) -> pd.DataFrame:
    """Long-format {LINK_FT, LINK_FTCLASS, LANES, CAP_PER_LANE} -- the model's
    own spdcap assumption for every lane count 1-7, for every facility type
    this dashboard actually covers (the FTs present in `final`, year-
    independent so this isn't duplicated per year). Not derived from any CCS
    observation -- this is the model's input assumption, not an output."""
    lut = spdcap_lookup_table()
    cap_cols = [c for c in lut.columns if c.startswith("CAPLN")]

    fts = final[["LINK_FT", "LINK_FTCLASS"]].drop_duplicates().sort_values("LINK_FT")
    long = fts.merge(lut[["FT"] + cap_cols], left_on="LINK_FT", right_on="FT", how="left").drop(columns=["FT"])
    long = long.melt(
        id_vars=["LINK_FT", "LINK_FTCLASS"], value_vars=cap_cols,
        var_name="LANES", value_name="CAP_PER_LANE",
    )
    long["LANES"] = long["LANES"].str.replace("CAPLN", "", regex=False).astype(int)
    return long.sort_values(["LINK_FT", "LANES"]).reset_index(drop=True)


def write_dashboard_data(final: pd.DataFrame, summary: pd.DataFrame, cap_lookup: pd.DataFrame) -> None:
    """Writes docs/data/capacity_data.json, the GitHub Pages dashboard's one
    data file -- {"stations": [...], "ft_summary": [...], "cap_lookup": [...]},
    derived from the same tables as the CSV outputs so there's a single
    source of truth for both. Everything the dashboard needs lives inside
    docs/ -- GitHub Pages only serves that folder, so the dashboard must
    never fetch a path outside it (e.g. ../output/)."""
    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)

    stations = final[
        [
            "SITE", "SITE_DESCRIPTION", "LATITUDE", "LONGITUDE", "LINK_STREET",
            "LINK_FT", "LINK_FTCLASS", "LANES", "CAP1HR1LN", "YEAR",
            "MAX_HOURLY_VOL", "PCT_OF_CAP", "DATE", "HOUR",
        ]
    ].copy()

    payload = {
        "stations": stations.to_dict(orient="records"),
        "ft_summary": summary.to_dict(orient="records"),
        "cap_lookup": cap_lookup.to_dict(orient="records"),
    }
    with open(DASHBOARD_DATA_DIR / "capacity_data.json", "w") as f:
        json.dump(_nan_to_none(payload), f, indent=2)


def _nan_to_none(obj):
    """Recursively replaces float NaN with None so json.dump emits valid
    `null` instead of the bare `NaN` token Python's json module writes by
    default -- NaN is not valid JSON and JSON.parse() rejects it in every
    browser. DataFrame.where(notna, None) doesn't reliably fix this itself:
    pandas casts None back to NaN for numeric-dtype columns, so sanitizing
    the plain dict/list payload after to_dict() is the only safe point."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def main():
    OUT_DIR.mkdir(exist_ok=True)

    raw = build_final_table()
    raw.to_csv(OUT_DIR / "ccs_station_link_capacity_raw.csv", index=False)

    final = apply_exclusions(raw)
    final.to_csv(OUT_DIR / "ccs_station_link_capacity_final.csv", index=False)

    summary_frames = []
    for year in [*YEAR_WORKBOOKS, "all"]:
        s = ft_summary(final[final["YEAR"] == year])
        s["YEAR"] = year
        summary_frames.append(s)
    summary = pd.concat(summary_frames, ignore_index=True)
    summary.to_csv(OUT_DIR / "ft_summary.csv", index=False)

    baseline_year = min(YEAR_WORKBOOKS)
    arterial_by_lane = lane_breakdown_for_ft(final[final["YEAR"] == baseline_year], ft=2)
    arterial_by_lane.to_csv(OUT_DIR / "ft2_principal_arterial_by_lane.csv", index=False)

    cap_lookup = capacity_lookup_for_dashboard(final)
    cap_lookup.to_csv(OUT_DIR / "cap_lookup.csv", index=False)

    write_dashboard_data(final, summary, cap_lookup)

    print(f"{len(raw)} CCS station-years matched, {len(final)} remaining after exclusions")
    print()
    print(summary.to_string(index=False))
    print()
    print(f"FT 2 (Principal Arterial) by lane count, {baseline_year}:")
    print(arterial_by_lane.to_string(index=False))


if __name__ == "__main__":
    main()
