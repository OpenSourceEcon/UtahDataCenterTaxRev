"""
Module for creating plot of data center size by peak electrical capacity and
fair market value.
"""
# Import packages
from pathlib import Path
import os
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


def a_exp_root_func1(a_exp, ElecCapacCutoff, a_lin):
    """
    Two-parameter exponential root function: of g(x) = e^{a_exp1 * x} + c_exp1
    Let f(x) = a_lin * x + b_lin. Solve for the exponential parameters
    a_exp1 from the condition g'(95)=f'(95)
    ln(a_exp) + a_exp * ElecCapacCutoff - ln(a_lin) = 0
    """
    error = np.log(a_exp) + a_exp * ElecCapacCutoff - np.log(a_lin)
    return error


def a_exp_root_func2(a_exp, ElecCapacCutoff, ElecCapacLow, a_lin, b_lin):
    """
    Three-parameter exponential root function:
    g(x) = e^{a_exp * x + b_exp} + c_exp
    Let f(x) = a_lin * x + b_lin. Solve for the exponential parameters
    a_exp, b_exp, and c_exp such that (i) g(95)=f(95), (ii) g(0.9)=f(0.9), and
    (iii) g'(95)=f'(95). Because we used (i) and (iii) to solve for b_exp and
    c_exp as functions of a_exp, we use (ii) for the root function to solve for
    a_exp.
    """
    b_exp, c_exp = b_exp_c_exp_func_a_exp(
        a_exp, ElecCapacCutoff, ElecCapacLow, a_lin, b_lin
    )
    error = (
        np.exp(a_exp * ElecCapacLow + b_exp) + c_exp - 0.0007786
    )
    return error


def b_exp_c_exp_func_a_exp(
    a_exp, ElecCapacCutoff, ElecCapacLow, a_lin, b_lin
):
    """
    Solve for b_exp and c_exp given a_exp from the three conditions on the
    three-parameter exponential function.
    g(x) = e^{a_exp * x + b_exp} + c_exp such that (i) g(95)=f(95),
    (ii) g(0.9)=f(0.9), and (iii) g'(95)=f'(95). Condition (iii) gives us b_exp
    as a function of a_exp:

    b_exp = ln(a_lin) - ln(a_exp) - a_exp * ElecCapacCutoff

    Then get c_exp as a function of a_exp and b_exp from condition (i):

    c_exp = a_lin*ElecCapacCutoff + b_lin - e^{a_exp * ElecCapacCutoff + b_exp}
    """
    b_exp = np.log(a_lin) - np.log(a_exp) - a_exp * ElecCapacCutoff
    c_exp = (
        a_lin * ElecCapacCutoff + b_lin -
        np.exp(a_exp * ElecCapacCutoff + b_exp)
    )
    return b_exp, c_exp


def make_mw_fmv_plot(title=True):
    """
    Create Bokeh scatter plot of electrical capacity (MW) and fair market value
    (FMV) along with linear regression line that estimates the realtionship.
    """
    # -------------------------------------------------------------------------
    # Load the data
    # -------------------------------------------------------------------------
    main_dir = Path(__file__).resolve().parent.parent
    data_dir = os.path.join(main_dir, "data")
    images_dir = os.path.join(main_dir, "images")
    df = pd.read_excel(
        os.path.join(data_dir, "us_data_centers_FMV.xlsx"),
        sheet_name="dataset"
    )

    # Create fmv_billions variable
    df["fmv_billions"] = df["supplemental_public_value_usd"] / 1e9
    # For row 3 (Vantage VA31 (43330 Data Dr parcel), use the
    # assessed_full_market_value_usd variable
    df.loc[2, "fmv_billions"] = df.loc[
        2, "assessed_full_market_value_usd"
    ] / 1e9

    # Create a sample version of the DataFrame df_samp with only the columns
    # facility_name, operator, city, county, state,
    # electrical_capacity_mw_public, and fmv_billions
    df_samp = df[[
        "facility_name", "operator", "city", "county", "state",
        "electrical_capacity_mw_public", "fmv_billions", "status"
    ]].copy()
    print(df_samp.keys())
    print(df_samp.describe())

    # Remove three outliers with electrical capacity above 600 MW and FMV < $1B
    df_samp = df_samp.loc[
        ~(
            (df["electrical_capacity_mw_public"] > 600) &
            (df["fmv_billions"] < 1)
        )
    ].copy()
    # Sort df_samp by electrical_capacity_mw_public in descending order and
    # reset the index
    df_samp = df_samp.sort_values(
        by="electrical_capacity_mw_public", ascending=False
    ).reset_index(drop=True)

    # Remove any rows that have missing or NAN values in the
    # electrical_capacity_mw_public or fmv_billions.
    df_samp = df_samp.dropna(
        subset=["electrical_capacity_mw_public", "fmv_billions"]
    ).reset_index(drop=True)
    # Save df_samp to a csv file in the data/all/ directory
    df_samp.to_csv(
        os.path.join(data_dir, "tab_us_data_centers_FMV_samp.csv"),
        index=False
    )

    # -------------------------------------------------------------------------
    # Estimate linear regression coefficients:
    # FMV = f(ElecCapac) = a_lin * ElecCapac + b_lin for ElecCapac > 95 MW
    #     = g(ElecCapac) = e^{a_exp * ElecCapac + b_exp} + c_exp
    #                      for ElecCapac <= 95 MW
    # where a_exp, b_exp, and c_exp are chosen such that
    # g(95) = f(95), g(0.9) = f(0.9), and g'(95) = f'(95)
    # -------------------------------------------------------------------------
    ElecCapacCutoff = 95
    ElecCapacLow = 0.9
    Xlin_mat = np.column_stack((
        np.ones(len(df_samp)),
        df_samp["electrical_capacity_mw_public"]
    ))
    yvec = df_samp["fmv_billions"].values
    ab_vec_lin = np.linalg.inv(Xlin_mat.T @ Xlin_mat) @ Xlin_mat.T @ yvec
    a_lin, b_lin = ab_vec_lin[1], ab_vec_lin[0]
    print("a_lin (slope coefficient):", a_lin)
    print("b_lin (intercept):", b_lin)

    # Start with fitting the 2-parameter exponential stitched function
    # g(x) = e^{a_exp1 * x} + c_exp1, such that g(95)=f(95) and g'(95)=f'(95).
    # This gives us a good initial guess for a_exp2 in our main specification.
    sol1 = root_scalar(
        a_exp_root_func1, args=(ElecCapacCutoff, a_lin),
        bracket=[1e-9, 1e4], x0=0.5
    )
    a_exp1 = sol1.root
    c_exp1 = a_lin * ElecCapacCutoff + b_lin - np.exp(a_exp1 * ElecCapacCutoff)
    print("a_exp1 (exponential coefficient):", a_exp1)
    print("c_exp1 (additive constant):", c_exp1)

    # Write a root finder that solves for a_exp such that the following
    # equation holds: np.ln(a_exp) + a_exp * ElecCapacCutoff - np.ln(a_lin) = 0
    sol2 = root_scalar(
        a_exp_root_func2, args=(ElecCapacCutoff, ElecCapacLow, a_lin, b_lin),
        x0=a_exp1
    )
    a_exp2 = sol2.root
    print("a_exp2 (exponential coefficient):", a_exp2)
    b_exp2, c_exp2 = b_exp_c_exp_func_a_exp(
        a_exp2, ElecCapacCutoff, ElecCapacLow, a_lin, b_lin
    )
    print("b_exp2 (exponential intercept):", b_exp2)
    print("c_exp2 (additive constant):", c_exp2)
    print(dir(sol2))

    elec_capac_vec = np.linspace(
        df_samp["electrical_capacity_mw_public"].min(),
        df_samp["electrical_capacity_mw_public"].max(),
        500
    )
    elec_capac_vec_gtcut = elec_capac_vec[elec_capac_vec >= ElecCapacCutoff]
    elec_capac_vec_lecut = elec_capac_vec[elec_capac_vec <= ElecCapacCutoff]
    FMV_pred_lin = a_lin * elec_capac_vec_gtcut + b_lin
    FMV_pred_lin_lecut = a_lin * elec_capac_vec_lecut + b_lin
    # FMV_pred_exp1 = np.exp(a_exp1 * elec_capac_vec_lecut) + c_exp1
    FMV_pred_exp2 = np.exp(a_exp2 * elec_capac_vec_lecut + b_exp2) + c_exp2

    # -------------------------------------------------------------------------
    # Make figure
    # -------------------------------------------------------------------------
    if title:
        fig_title = (
            "US data centers by electrical capacity (MW) and fair " +
            "market value (FMV)"
        )
    else:
        fig_title = ""
    fig_filename = "fig_MWtoFMV.html"
    output_file(
        os.path.join(images_dir, fig_filename),
        title=fig_title, mode='inline'
    )

    TOOLS = "pan, box_zoom, wheel_zoom, save, reset, help"

    fig = figure(
        title=fig_title,
        tools=TOOLS,
        x_axis_label="Peak Electrical Capacity (MW)",
        y_axis_label="Fair Market Value ($Bil)",
        toolbar_location="right"
    )
    fig.toolbar.logo = None
    source = ColumnDataSource(df_samp)
    scatter_renderer = fig.scatter(
        x="electrical_capacity_mw_public",
        y="fmv_billions",
        source=source,
        size=10,
        color="green",
        alpha=0.5,
        hover_color="red",
        hover_alpha=1.0,
        hover_line_color="black",
        hover_line_width=1.5,
        legend_label="Data Centers"
    )
    fig.line(
        x=elec_capac_vec_gtcut,
        y=FMV_pred_lin,
        line_width=2,
        color="blue",
        legend_label="Linear Fit"
    )
    fig.line(
        x=elec_capac_vec_lecut,
        y=FMV_pred_lin_lecut,
        line_width=2,
        color="blue",
        line_dash="dashed",
        legend_label="Linear Fit (for <= 95 MW)"
    )
    fig.line(
        x=elec_capac_vec_lecut,
        y=FMV_pred_exp2,
        line_width=2,
        color="red",
        legend_label="Exponential Fit (for <= 95 MW)"
    )
    hover = HoverTool(renderers=[scatter_renderer])
    hover.tooltips = [
        ("Facility Name", "@facility_name"),
        ("City", "@city"),
        ("County", "@county"),
        ("State", "@state"),
        ("Electrical Capacity (MW)", "@electrical_capacity_mw_public"),
        ("Fair Market Value (Billions USD)", "@fmv_billions{$0.000}"),
        ("Status", "@status")
    ]
    fig.add_tools(hover)
    fig.legend.location = "top_left"
    fig.add_layout(fig.legend[0], 'center')

    note_text_list = [
        (
            'Source: Richard W. Evans (@RickEcon), updated May 29, 2026. ' +
            'Electrical capacity and fair market'
        ),
        (
            '    value data come from an individual search. The data in ' +
            'this figure with sources are available in the'
        ),
        (
            '    data/all/us_data_centers_FMV.xlsx file in the ' +
            'https://github.com/OpenSourceEcon/DataCenterAtlas'
        ),
        (
            '    GitHub repository.'
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
    make_mw_fmv_plot(title=True)
