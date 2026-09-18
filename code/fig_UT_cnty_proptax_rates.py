"""
Module for creating image of Utah counties and their average effective property
tax rates
"""
# Import packages
from pathlib import Path
import os
import json
import pickle
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.optimize import root_scalar

from bokeh.io import output_file
from bokeh.plotting import figure, show
from bokeh.palettes import Reds9
from bokeh.models import (
    GeoJSONDataSource, HoverTool, ColumnDataSource, Legend, LegendItem, Title
)

# Vertex simplification tolerance, in degrees, for the lake and river layers.
# At this figure's 600 px width one pixel spans roughly 0.009 degrees, so
# 0.0005 (about 50 m) stays well below what the map can resolve while cutting
# the embedded GeoJSON by a factor of six.
HYDRO_SIMPLIFY_TOL = 0.0005

# The only NHD attribute worth carrying on the lake and river layers. The
# other ~15 columns are unused and just inflate the embedded GeoJSON.
HYDRO_KEEP_COLS = ["GNIS_Name", "geometry"]


def make_ut_cnty_proptax_rate_map(
    create_data=False, save_data=True, title=True
):
    """
    Create Bokeh map of Utah counties and their average effective property tax
    rates
    """
    main_dir = Path(__file__).resolve().parent.parent
    data_dir = os.path.join(main_dir, "data")
    images_dir = os.path.join(main_dir, "images")

    if create_data:
        print("Creating all the data from shapefiles.")
        start_time_all = time.time()
        # ---------------------------------------------------------------------
        # Add Utah state boundary shape file
        # ---------------------------------------------------------------------
        # Download U.S. states shape files from US Census Bureau
        # "https://www2.census.gov/geo/tiger/GENZ2023/shp/" +
        # "cb_2023_us_state_500k.zip"
        print("")
        print("Creating Utah state boundary shapefile,")
        start_time_ut = time.time()
        us_shapefile_path = (
            os.path.join(
                data_dir, "shp", "cb_2023_us_state_500k",
                "cb_2023_us_state_500k.shp"
            )
        )
        states_gdf = gpd.GeoDataFrame.from_file(us_shapefile_path)
        states_gdf_json = states_gdf.to_json()
        states_gjson = json.loads(states_gdf_json)

        # Build a Utah polygon GeoDataFrame (not GeoJSON) for spatial ops
        ut_gdf = states_gdf.loc[states_gdf["STUSPS"] == "UT"].copy()
        # Dissolve in case UT is multipart; makes a single boundary geometry
        ut_gdf = ut_gdf.dissolve()

        ut_gdf_str = ut_gdf.to_json()
        ut_src = GeoJSONDataSource(geojson=ut_gdf_str)

        elapsed_time_ut = time.time() - start_time_ut
        min = int(elapsed_time_ut // 60)
        sec = np.round(elapsed_time_ut % 60, 1)
        print(f"took {min} minutes and {sec} seconds.")

        # ---------------------------------------------------------------------
        # Add Utah county boundaries shape file
        # ---------------------------------------------------------------------
        print("")
        print("Creating county boundaries shapefile")
        start_time_cnt = time.time()
        county_shapefile_path = os.path.join(
            data_dir, "shp", "cb_2023_us_county_500k",
            "cb_2023_us_county_500k.shp"
        )
        counties_gdf = gpd.GeoDataFrame.from_file(county_shapefile_path)
        # Filter to Utah counties (STATEFP for Utah = 49)
        ut_counties_gdf = counties_gdf.loc[
            counties_gdf["STATEFP"] == "49"
        ].copy()
        # Match CRS
        ut_counties_gdf = ut_counties_gdf.to_crs(ut_gdf.crs)
        ut_counties_gdf_str = ut_counties_gdf.to_json()
        # Convert to Bokeh GeoJSON
        ut_counties_src = GeoJSONDataSource(geojson=ut_counties_gdf_str)

        elapsed_time_cnt = time.time() - start_time_cnt
        min = int(elapsed_time_cnt // 60)
        sec = np.round(elapsed_time_cnt % 60, 1)
        print(f"took {min} minutes and {sec} seconds.")

        # ---------------------------------------------------------------------
        # Create lake, reservoirs, and rivers shape file
        # ---------------------------------------------------------------------
        # Source: Utah Lakes NHD comes from the Utah UGRC - Authoritative Data,
        # Utah Automated Geographic Reference Center (AGRC)
        # https://opendata.gis.utah.gov/datasets/utah-lakes-nhd/about.
        # Utah Streams NHD comes from the Utah UGRC - Authoritative Data,
        # Utah Automated Geographic Reference Center (AGRC)
        # https://opendata.gis.utah.gov/datasets/utah-streams-nhd/about.
        print("")
        print("Creating lakes and reservoirs shapefile")
        start_time_lk = time.time()
        lakes_res_shapefile_path = os.path.join(
            data_dir, "shp",
            "UtahLakesNHD",
            "LakesNHDHighRes.shp"
        )
        lakes_res_gdf = gpd.GeoDataFrame.from_file(lakes_res_shapefile_path)

        # Keep only the major waterbodies. The full high-resolution NHD layer
        # has 46,770 Utah waterbodies, whose GeoJSON is too big to embed in
        # the figure HTML. Filtering here, before the reprojection and clip,
        # also keeps those two steps fast.
        lakes_res_gdf = lakes_res_gdf.loc[
            lakes_res_gdf["IsMajor"] == 1
        ].copy()

        # Ensure lakes and reservoirs are in same CRS as Utah
        lakes_res_gdf = lakes_res_gdf.to_crs(ut_gdf.crs)

        # Clip lakes/reservoirs to Utah boundary
        lakes_res_ut_gdf = gpd.clip(lakes_res_gdf, ut_gdf)

        # Drop the unused attributes and thin the shoreline vertices. This
        # also removes the datetime columns, so no ISO-string conversion is
        # needed before to_json().
        lakes_res_ut_gdf = lakes_res_ut_gdf[HYDRO_KEEP_COLS].copy()
        lakes_res_ut_gdf["geometry"] = lakes_res_ut_gdf.geometry.simplify(
            HYDRO_SIMPLIFY_TOL
        )

        lakes_res_ut_gdf_str = lakes_res_ut_gdf.to_json()
        lakes_res_ut_src = GeoJSONDataSource(geojson=lakes_res_ut_gdf_str)

        elapsed_time_lk = time.time() - start_time_lk
        min = int(elapsed_time_lk // 60)
        sec = np.round(elapsed_time_lk % 60, 1)
        print(f"took {min} minutes and {sec} seconds.")

        # print("")
        # print("Creating rivers and streams shapefile")
        # start_time_riv = time.time()
        # riv_shapefile_path = os.path.join(
        #     data_dir, "shp",
        #     "UtahStreamsNHD",
        #     "StreamsNHDHighRes.shp"
        # )
        # riv_gdf = gpd.GeoDataFrame.from_file(riv_shapefile_path)

        # # Keep only the major streams. The full high-resolution NHD layer has
        # # 305,685 Utah reaches, whose GeoJSON is too big to embed in the
        # # figure HTML. Filtering here, before the reprojection and clip, also
        # # keeps those two steps fast.
        # riv_gdf = riv_gdf.loc[riv_gdf["IsMajor"] == 1].copy()

        # # Ensure rivers and streams are in same CRS as Utah
        # riv_gdf = riv_gdf.to_crs(ut_gdf.crs)

        # # Clip rivers/streams to Utah boundary
        # riv_ut_gdf = gpd.clip(riv_gdf, ut_gdf)

        # # Drop the unused attributes and thin the channel vertices. This also
        # # removes the datetime columns, so no ISO-string conversion is needed
        # # before to_json().
        # riv_ut_gdf = riv_ut_gdf[HYDRO_KEEP_COLS].copy()
        # riv_ut_gdf["geometry"] = riv_ut_gdf.geometry.simplify(
        #     HYDRO_SIMPLIFY_TOL
        # )

        # riv_ut_gdf_str = riv_ut_gdf.to_json()
        # riv_ut_src = GeoJSONDataSource(geojson=riv_ut_gdf_str)

        # elapsed_time_riv = time.time() - start_time_riv
        # min = int(elapsed_time_riv // 60)
        # sec = np.round(elapsed_time_riv % 60, 1)
        # print(f"took {min} minutes and {sec} seconds.")

        # ---------------------------------------------------------------------
        # Save gdf and geojson data files
        # ---------------------------------------------------------------------
        # Create dictionaries of GeoDataFrames and GeoJSONDataSources for all
        # layers
        gdf_dict = {
            "ut_gdf": ut_gdf,
            "ut_counties_gdf": ut_counties_gdf,
            "lakes_res_ut_gdf": lakes_res_ut_gdf
            # "riv_ut_gdf": riv_ut_gdf
        }
        geojson_dict = {
            "ut_gdf_str": ut_gdf_str,
            "ut_counties_gdf_str": ut_counties_gdf_str,
            "lakes_res_ut_gdf_str": lakes_res_ut_gdf_str
            # "riv_ut_gdf_str": riv_ut_gdf_str,
        }
        src_dict = {
            "ut_src": ut_src,
            "ut_counties_src": ut_counties_src,
            "lakes_res_ut_src": lakes_res_ut_src
            # "riv_ut_src": riv_ut_src
        }
        if save_data:
            for name, gdf in gdf_dict.items():
                pickle.dump(
                    gdf, open(
                        os.path.join(data_dir, "gdf", f"{name}.pkl"), "wb"
                    )
                )
            for name, geojson in geojson_dict.items():
                with open(
                    os.path.join(data_dir, "geojson", f"{name}.geojson"),
                    "w", encoding="utf-8"
                ) as f:
                    f.write(geojson)

        elapsed_time_all = time.time() - start_time_all
        min = int(elapsed_time_all // 60)
        sec = np.round(elapsed_time_all % 60, 1)
        print("")
        print(f"Total data creation took {min} minutes and {sec} seconds.")
    else:
        print("")
        print("Reading in all the data from hard drive,")
        start_time = time.time()

        gdf_name_list = [
            "ut_gdf", "ut_counties_gdf", "lakes_res_ut_gdf"
            # , "riv_ut_gdf"
        ]
        gdf_dict = {
            name: pickle.load(
                open(os.path.join(data_dir, "gdf", f"{name}.pkl"), "rb")
            ) for name in gdf_name_list
        }

        geojson_name_list = [
            "ut_gdf_str", "ut_counties_gdf_str", "lakes_res_ut_gdf_str"
            # "riv_ut_gdf_str"
        ]
        src_dict = {}
        for name in geojson_name_list:
            path = os.path.join(data_dir, "geojson", f"{name}.geojson")
            with open(path, "r", encoding="utf-8") as f:
                obj_str = f.read()
            obj_src = GeoJSONDataSource(geojson=obj_str)
            src_name = name.split("_gdf_str")[0] + "_src"
            src_dict[src_name] = obj_src

        elapsed_time = time.time() - start_time
        min = int(elapsed_time // 60)
        sec = np.round(elapsed_time % 60, 1)
        print(f"took {min} minutes and {sec} seconds.")

    # -------------------------------------------------------------------------
    # Merge property tax rates into counties GDF and assign Reds9 colors
    # Data are available in "./data/avg_proptax_rate_by_cnty_ut_2025.csv" file.
    # -------------------------------------------------------------------------
    tax_df = pd.read_csv(
        os.path.join(data_dir, "avg_proptax_rate_by_cnty_ut_2025.csv"),
        header=0
    )

    ut_counties_gdf = gdf_dict["ut_counties_gdf"].merge(
        tax_df[["cnty_name", "rate_cnty_avg"]],
        left_on="NAME",
        right_on="cnty_name",
        how="left"
    )

    # Reds9[8] = prop tax rate 0.0043-0.0055, Reds9[7] = 0.0055-0.0067, ..., Reds9[0] = 0.0139-0.0151
    def proptax_rate_color(rate):
        if pd.isna(rate):
            return "gray"
        return Reds9[8 - int(np.clip((rate - 0.0043) / 0.0012, 0, 8))]

    ut_counties_gdf["color"] = ut_counties_gdf["rate_cnty_avg"].apply(
        proptax_rate_color
    )

    # Rebuild counties source with tax data and color
    src_dict["ut_counties_src"] = GeoJSONDataSource(
        geojson=ut_counties_gdf.to_json()
    )

    # -------------------------------------------------------------------------
    # Make figure
    # -------------------------------------------------------------------------
    fig_title = (
        "Utah map of 2025 average effective property tax rates by county"
    )

    # fig_title = ""
    fig_filename = "ut_proptaxratemap.html"
    output_file(
        os.path.join(images_dir, fig_filename),
        title=fig_title, mode='inline'
    )

    TOOLS = "pan, box_zoom, wheel_zoom, save, reset, help"

    fig = figure(
        title=fig_title,
        height=700,
        width=700,
        tools=TOOLS,
        min_border = 0,
        x_axis_location = None, y_axis_location = None,
        toolbar_location="right"
    )
    fig.toolbar.logo = None
    fig.grid.grid_line_color = None

    # Utah state outline
    print("Fig Status: Plotting ut_src")
    fig.patches(
        "xs", "ys",
        source=src_dict["ut_src"],
        fill_alpha=0.00,
        line_color="black",
        line_width=2,
        fill_color="white"
    )

    # Utah counties outline (colored by property tax rate)
    print("Fig Status: Plotting ut_counties_src")
    r_counties = fig.patches(
        "xs", "ys",
        source=src_dict["ut_counties_src"],
        fill_color="color",
        fill_alpha=0.7,
        line_color="black",
        line_width=1,
        hover_fill_color="color",
        hover_fill_alpha=0.9,
        hover_line_color="black",
        hover_line_width=3
    )

    # Utah lakes / reservoirs
    print("Fig Status: Plotting lakes_res_ut_src")
    fig.patches(
        "xs", "ys",
        source=src_dict["lakes_res_ut_src"],
        fill_color="blue",
        fill_alpha=0.3,
        line_alpha=0.3,
        line_width=0.2
    )

    # # Utah rivers / streams
    # print("Fig Status: Plotting riv_ut_src")
    # fig.multi_line(
    #     "xs", "ys",
    #     source=src_dict["riv_ut_src"],
    #     line_color="blue",
    #     line_alpha=0.3,
    #     line_width=0.2
    # )

    # Build 9-entry color legend for mill rate ranges
    legend_items = []
    for i in range(9):
        low = 0.43 + (0.12 * i)
        dummy = fig.rect(
            x=[float("nan")], y=[float("nan")], width=0, height=0,
            fill_color=Reds9[8 - i], fill_alpha=0.7,
            line_color="black", line_width=0.5
        )
        legend_items.append(
            LegendItem(
                label=f"{low:.2f}% to {low + 0.12:.2f}%",
                renderers=[dummy]
            )
        )
    fig_legend = Legend(
        items=legend_items,
        title="County property tax rates"
    )
    fig.add_layout(fig_legend, 'right')

    # Legend properties
    fig.legend.orientation = "vertical"
    fig.legend.background_fill_color = "white"
    fig.legend.background_fill_alpha = 0.9
    fig.legend.border_line_color = "black"
    fig.legend.border_line_width = 1
    fig.legend.label_text_font_size = "10pt"
    fig.legend.spacing = 2
    fig.legend.padding = 6
    fig.legend.margin = 6

    # Set up hover tool (counties only)
    hover = HoverTool(renderers=[r_counties])
    hover.point_policy = "follow_mouse"
    hover.tooltips = [
        ("County", "@NAME"),
        ("Avg. eff. prop. tax rate (2025)", "@rate_cnty_avg{0.00%}")
    ]
    fig.add_tools(hover)

    note_text_list = [
        (
            "Source: Richard W. Evans (@RickEcon), updated Sep. 17, 2026. " +
            "Average effective property"
        ),
        (
            "    taxe rates come from the Utah State Tax Commission's " +
            "(USTC) 2025 Tax Rates by Tax Area."
        )
    ]
    for note_text in note_text_list:
        caption = Title(
            text=note_text, align='left', text_font_size='9pt',
            text_font_style='italic',
            text_color='black',
            standoff=0
        )
        fig.add_layout(caption, 'below')

    show(fig)


if __name__ == "__main__":
    make_ut_cnty_proptax_rate_map(create_data=True, save_data=True, title=True)
