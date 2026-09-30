"""
Streamlit Based Wind Resource Analysis Tool
======================================
Generic, upload-your-own-data version of the wind measurement + modelled
wind data (ERA5 / CFSR / MERRA-2 / etc.) correlation and long-term correction
pipeline. Both the measurement file AND the modelled dataset have their
columns mapped interactively - nothing about column names or file layout is
assumed fixed.

Run with:  streamlit run wind_streamlit_app.py
"""

import re
import io
import os
import csv
import json
import zipfile
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats
from matplotlib.path import Path as MplPath

st.set_page_config(page_title="Wind Resource Analysis", layout="wide")

# --- In-app Help / Read Me content -----------------------------------------
# Deliberately just the "how to use" material from README.md - the deployment/
# Azure-Pipelines/dependency-pinning sections are for whoever maintains the
# repo, not for someone using the deployed tool, so they're left out here.
HELP_TEXT = {
    "Long-Term Correction": """
1. **Upload measurement data** - any number of files, any mix of `.csv`,
   Campbell Scientific TOA5 `.dat`/`.sta`, a second `.dat`/`.sta` layout seen
   from floating LiDAR buoy systems (a header row starting with
   'timestamp'), or NetCDF `.nc`. Files are auto-detected by format,
   concatenated, and sorted by time; overlapping timestamps are
   de-duplicated. TOA5 files need no mapping; CSV, the generic `.dat`/`.sta`
   layout, and NetCDF each get one shared mapping step, applied to every
   file of that type.
2. **Map columns**: pick, for each height you care about, the wind speed
   column (required) and wind direction column (optional - needed only for
   wind roses), from whatever columns the combined dataset ended up with.
   Heights can be entered in any order - they're sorted low-to-high
   automatically everywhere in the results.
3. **Set your measurement's timezone offset** - set once here, right after
   mapping columns, since it doesn't change no matter how many modelled
   sources you compare against.
   Beyond whatever invalid-value codes you enter, any wind speed reading
   over 100 m/s (or negative) or wind direction outside 0-360° is
   automatically treated as invalid too, in your measurement data - a
   sensor fault can produce a garbage value that isn't one of the usual
   sentinels (9999, -999, etc.), and even a couple of such rows can
   quietly wreck a whole correlation. A message shows exactly what got
   caught and where, whenever this catches something. Modelled datasets
   aren't checked this way, since they're assumed clean.
4. **Upload one or more modelled wind datasets** - any hourly modelled/
   reanalysis wind time series works (ERA5, CFSR, MERRA-2, Vortex, or
   similar), and any number of them can be added - e.g. a Vortex ERA5
   extraction alongside a Vortex CFSR one and a MERRA2 series, to compare
   how each correlates before picking one. Each source is configured once,
   then collapses to a summary row; "+ Add a modelled source" adds another.
   Vortex `.txt` exports are auto-detected and parsed automatically
   (timestamp, height, and timezone all read from the file header); other
   formats are mapped the same way as the measurement file. Each source
   keeps its own timezone offset (Vortex's pre-filled from the file header).
   Modelled data is assumed clean - there's no invalid-value-code field for
   it, unlike the measurement file. A map of every configured source's
   location appears once you've added at least one, so you can check where
   each one actually is.
5. Browse the result tabs: Data Availability, Monthly Means, **Wind Rose**
   (pick a height once, see every configured source's rose for it - each
   using its own native-resolution data over the period it overlaps with
   your measurement, not hourly-averaged or matched to concurrent
   timestamps, since a rose describes a distribution rather than a point-by-
   point comparison), Shear Profile, and **Correlation** - organised by
   panel type (Hourly/Daily/Monthly, which DOES need concurrent, hourly-
   matched data), with every configured source shown side by side so the
   same statistic is directly comparable across sources.
6. **Long-Term Result**: every configured source gets its own full
   breakdown - concurrent stats, chart, everything - not just a single
   pick. A selector defaults to whichever source had the highest hourly R²
   for headline reporting (marked with a star), but nothing is hidden
   behind that choice.
7. **Download**: package every chart (plus a summary and the availability
   table) into a single ZIP from the Download section at the bottom -
   covers every configured source, not just the one selected for headline
   reporting.
""",
    "Measurement Campaign Planning": """
1. **Site boundary**: `.geojson`, `.kml`, or a CSV/Excel list of
   Easting/Northing boundary vertices (UTM zone or a custom EPSG code).
2. **Wind maps**: each source gets its own named column - rename, remove, or
   add a source as needed, then drop that source's `.asc` files straight in
   (height is read from the filename, no extra step to add them).
3. **Turbine layout** (optional but needed for step 6): `.geojson`, `.gpkg`,
   `.xlsx` (Latitude/Longitude columns), or a CSV/Excel of Easting/Northing
   turbine positions.
4. **Interactive map**: the active wind map renders as a heatmap with the
   boundary and layout overlaid. Click anywhere inside the boundary to drop
   a labeled measurement point (A, B, C...); click "Fix measurement points"
   once you're happy with them.
5. **Compare wind speed at chosen points**: enter combinations like `A,B;A,C`
   to get an inline chart comparing wind speed at those points across every
   loaded wind map.
6. **Locate best measurement points**: choose how many measurement locations
   you want; the tool clusters your turbines with K-Means and searches inside
   the boundary for the point in each cluster that best represents that
   cluster's modelled wind speed, downloadable as CSV.
""",
    "Preliminary Wind Resource Assessment": """
1. **Site Boundary** and **Layout**: same as Measurement Campaign Planning.
2. **Wind Maps**: upload every `.asc` file for ERA5 and for CFSR separately -
   height is read from each filename.
3. **Interactive Map**: browse any loaded map with the layout overlaid.
4. **Shear Calculation**: fits a power-law shear exponent to ERA5's own
   heights and CFSR's own heights *separately, at every turbine position*,
   then averages the two per position.
5. **Hub Height & Weighting**: set the hub height and an ERA5/CFSR weight
   slider; extrapolates each source to hub height per position using that
   position's own shear, then blends them.
6. **Calibration**: add long-term-corrected wind speeds at known locations,
   then choose:
   - **Site Average CF**: one calibration factor applied everywhere.
   - **Distance Weighted CF**: each turbine gets a blend of every
     calibration point's factor, weighted by distance - nearer points
     count more.
   - **Kriging (Ordinary)**: a geostatistical blend using the same distance
     decay, but it additionally accounts for redundancy between calibration
     points that sit close together. Generally the more statistically sound
     choice once you have 3+ points, particularly if any are clustered.
     With few points - especially if one has a much shorter record than the
     others - kriging can mathematically produce a negative weight; "Prevent
     negative kriging weights" (on by default) clips that to zero and
     renormalizes the rest, which testing found meaningfully more reliable
     in that situation. Check the cross-validation table with it toggled
     either way for your own points, since it isn't a universal fix.

   Each calibration point also has an optional **Measurement Duration (months)**
   field - leave it at 0 if you don't want to flag anything (treated as
   full-confidence, same as before). Duration is converted to a confidence
   weight via inverse-variance weighting against a real measured uncertainty
   curve (Abascal Mendez et al., 2026, *Inventions* - 30 real masts, ~2.1%
   uncertainty at 3 months down to ~0.4% at 12 months), not an assumed
   shape - this penalizes a short record considerably harder than a simple
   proportional scale would. An optional **Correlation R²** field applies a
   further adjustment if you have it from your own MCP regression. A
   **manual weight (0-1)** overrides both entirely for a point where you'd
   rather set confidence yourself. See the "?" next to Point quality in the
   Calibration section for the full explanation.

   With 2 or more calibration points, a leave-one-out cross-validation table
   shows the actual RMSE/MAE each method achieves on your own points - use
   this to see which method genuinely fits your site rather than assuming.
7. **Download Plots and Table**: packages an Excel workbook plus static map
   images into a ZIP.
""",
    "Wind Measurement Tracker": """
**Continuing from a previous month?** At the top of the page, choose
"Continue tracking" and upload the transferable package you downloaded last
time, before uploading anything else - it carries forward your combined
measurement history and settings, so you only need to add the new month's
file(s) rather than re-uploading and re-mapping everything from scratch.

1. **Upload measurement files**: as many as you have, from the first month
   through the most recent - `.csv`, Campbell Scientific TOA5 `.dat`/`.sta`,
   a second `.dat`/`.sta` layout seen from floating LiDAR buoy systems (a
   header row starting with 'timestamp'), or NetCDF `.nc`. Files are
   auto-detected by format, concatenated, and sorted by time; overlapping
   timestamps are de-duplicated.
2. **Map heights**: for each height, choose the wind speed column
   (required), and optionally wind direction and turbulence intensity -
   either a direct TI column or computed from a wind-speed standard-
   deviation column alongside the mean. Pre-filled automatically if you
   loaded a package. Beyond whatever invalid-value codes were used during
   upload, any wind speed reading over 100 m/s (or negative) or wind
   direction outside 0-360° is automatically treated as invalid too, in
   your measurement data - a sensor fault can produce a garbage value that
   isn't one of the usual sentinels, and even a couple of such rows can
   quietly wreck a whole correlation. A message shows exactly what got
   caught, whenever this catches something. The modelled dataset used in
   Long-Term Analysis isn't checked this way, since it's assumed clean.
3. Browse the result tabs: Data Availability, Monthly Mean WS, **TI vs Wind
   Speed** (one small chart per calendar month showing how turbulence
   intensity varies with wind speed - the standard wind-industry
   characterisation, rather than a single average that would hide the
   relationship), and Wind Rose - one small rose per calendar month on a
   shared scale, to spot directional shifts across the tracked period at
   a glance.
4. **Long-Term Analysis**: upload a modelled dataset (Vortex or any hourly
   reanalysis - ERA5, CFSR, MERRA-2) to see whether your running measured
   average is converging toward the long-term estimate as more months are
   added. The long-term estimate is fit once, using every concurrent hour
   currently available, and shown as a fixed reference line - what moves is
   the cumulative measured average as data accumulates, not the regression.
5. **Download**: package every chart into a single ZIP, or download the
   **transferable package** - carries the combined measurement data and
   your settings forward to next month, so the whole history doesn't need
   re-uploading and re-mapping each time.
""",
}


@st.dialog("How to Use This Tool", width="large")
def show_help_dialog(current_mode):
    modes = list(HELP_TEXT.keys())
    chosen = st.radio("Section", modes, index=modes.index(current_mode),
                       horizontal=True, label_visibility="collapsed")
    st.markdown(HELP_TEXT[chosen])


st.sidebar.title("Wind Analysis Toolkit")
mode = st.sidebar.radio(
    "Choose tool",
    ["Long-Term Correction", "Measurement Campaign Planning",
     "Preliminary Wind Resource Assessment", "Wind Measurement Tracker"],
    help="Long-Term Correction: measurement + modelled data correlation and long-term "
         "wind speed. Measurement Campaign Planning: preliminary wind look-up and "
         "LiDAR/FLiDAR siting from modelled maps, a site boundary, and a turbine layout. "
         "Preliminary Wind Resource Assessment: multi-source (ERA5/CFSR) shear, hub-height "
         "extrapolation, weighting, and long-term calibration across a turbine layout. "
         "Wind Measurement Tracker: upload monthly measurement files as they arrive and "
         "track wind speed, direction, turbulence intensity, and data availability over time.")
if st.sidebar.button("📖 How to Use This Tool", width="stretch"):
    show_help_dialog(mode)
st.sidebar.divider()

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})

ACCENT = "#2b6cb0"
FLAG = "#d64545"
NEW_HIGHLIGHT = "#e8a33d"
PLOT_DPI = 400  # bumped for headroom on high-DPI/retina displays, where "stretch" filling a
                # large monitor can demand more physical pixels than a lower dpi can supply

# Fixed display widths (px). None of these use "stretch" any more: an unbounded width means
# a large/high-DPI monitor can demand more physical pixels than a fixed-dpi render can supply,
# which is what was causing the haziness - a bounded width keeps the resolution requirement
# achievable regardless of screen size.
WIDTH_ROSE = 800
WIDTH_ROSE_GRID = 1400
WIDTH_SHEAR = 500
WIDTH_AVAILABILITY = 1200
WIDTH_MONTHLY = 1200


def fig_to_png_bytes(fig, dpi=PLOT_DPI):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def show_fig(fig, width="stretch"):
    """Streamlit deprecated passing savefig kwargs (like dpi=) through
    st.pyplot - it now only supports its own fixed default (dpi=200,
    bbox_inches='tight'), with a bare passthrough to st.pyplot triggering a
    visible deprecation warning in the app. Their own recommended fix is to
    render the PNG ourselves and display it with st.image instead, which
    also means this can just reuse fig_to_png_bytes rather than duplicating
    the savefig-to-buffer logic - and st.image's width parameter accepts
    the exact same values (an int pixel width, or 'stretch') already used
    at every call site here, so nothing else needs to change."""
    st.image(fig_to_png_bytes(fig, dpi=PLOT_DPI), width=width)


def sorted_heights(height_map):
    """Canonical low-to-high ordering, used for every dropdown/subplot/legend
    so results always read in a sensible physical order regardless of the
    order heights were typed into the mapping form."""
    return sorted(height_map, key=lambda hm: hm["height"])


def detect_height_from_colname(col_name, default=100.0):
    """Best-effort guess of a height embedded in a column name (e.g. 'WS100',
    'wind_speed_100m', 'u100') to pre-fill the height field - always editable."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*m\b", col_name, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)", col_name)
    if m:
        return float(m.group(1))
    return default


# ==============================================================================
# HELPERS - DATA LOADING & CLEANING (shared by measurement AND modelled data)
# ==============================================================================

def parse_invalid_codes(text):
    if not text.strip():
        return []
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok:
            try:
                out.append(float(tok))
            except ValueError:
                pass
    return out


@st.cache_data(show_spinner=False)
def read_raw_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))


def sniff_vortex_format(file_bytes):
    """Vortex text exports start with a metadata block (Lat=.. Lon=.. Hub-Height=..)
    before the real header row - and some exports (especially ones with a longer
    licensing/disclaimer preamble, or a metadata field worded slightly differently
    across Vortex product versions) can push that block well past any fixed byte
    count. The one truly load-bearing, format-defining signal - the same one
    parse_vortex_txt itself relies on to locate its header row - is the tabular
    header line starting with 'YYYYMMDD'. Scanning line-by-line for that (rather
    than a byte-window keyword combination) means an unusually long preamble or
    an unfamiliar metadata phrasing can no longer cause a false negative here."""
    text = file_bytes.decode("utf-8", errors="ignore")
    for line in text.splitlines()[:500]:  # generous cap; real preambles never need more
        if line.strip().upper().startswith("YYYYMMDD"):
            return True
    return False


@st.cache_data(show_spinner="Parsing Vortex file...")
def parse_vortex_txt(file_bytes):
    """
    Parses a Vortex-format whitespace-delimited .txt export:
      - a few metadata lines (Lat=.. Lon=.. Hub-Height=.. Timezone=.. ...)
      - a header row starting with YYYYMMDD HHMM
      - whitespace-delimited data, with date and time in separate columns
    Combines YYYYMMDD + HHMM into a single 'Timestamp' column automatically,
    and pulls Lat / Lon / Hub-Height / Timezone out of the metadata for auto-fill.
    Returns (df, detected_height, detected_tz_offset, detected_lat, detected_lon).
    """
    text = file_bytes.decode("utf-8", errors="ignore")
    lines = text.splitlines()

    detected_height = 100.0
    detected_tz = 0.0
    detected_lat, detected_lon = None, None
    header_idx = None
    for i, line in enumerate(lines):
        m = re.search(r"Hub-Height\s*=\s*([\d.]+)", line, flags=re.IGNORECASE)
        if m:
            detected_height = float(m.group(1))
        m = re.search(r"Timezone\s*=\s*(-?[\d.]+)", line, flags=re.IGNORECASE)
        if m:
            detected_tz = float(m.group(1))
        m = re.search(r"\bLat\s*=\s*(-?[\d.]+)", line, flags=re.IGNORECASE)
        if m:
            detected_lat = float(m.group(1))
        m = re.search(r"\bLon\s*=\s*(-?[\d.]+)", line, flags=re.IGNORECASE)
        if m:
            detected_lon = float(m.group(1))
        if line.strip().upper().startswith("YYYYMMDD"):
            header_idx = i
            break

    if header_idx is None:
        header_idx = 3  # fallback matching Vortex's usual 3 metadata lines

    from io import StringIO
    data_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(StringIO(data_text), sep=r"\s+", engine="python")

    date_col, time_col = df.columns[0], df.columns[1]
    ts = pd.to_datetime(
        df[date_col].astype(str) + df[time_col].astype(str).str.zfill(4),
        format="%Y%m%d%H%M", errors="coerce")
    df.insert(0, "Timestamp", ts)
    df = df.drop(columns=[date_col, time_col])
    return df, detected_height, detected_tz, detected_lat, detected_lon


def guess_column(cols, keywords, default_idx=0):
    """Best-effort default selection for a selectbox - e.g. picks 'M(m/s)' for
    wind speed, 'D(deg)' for direction - always overridable by the user."""
    for kw in keywords:
        for i, c in enumerate(cols):
            if kw.lower() in c.lower():
                return i
    return default_idx


@st.cache_data(show_spinner="Parsing timestamps and cleaning data...")
def build_clean_df(raw_df, ts_col, dayfirst, invalid_codes):
    """Generic cleaner - used for both the measurement file and the modelled
    wind dataset, since neither has a fixed column layout any more."""
    df = raw_df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], dayfirst=dayfirst, errors="coerce")
    df = df.dropna(subset=[ts_col]).set_index(ts_col).sort_index()
    # Numeric conversion MUST happen before the invalid-code replace: if a column has
    # any stray non-numeric text in it (even just a whitespace-only cell), pandas keeps
    # that whole column as text, and a text "9999" won't match a float invalid code of
    # 9999.0 - so codes would silently slip through uncaught. Converting first guarantees
    # every column is numeric before we try to match codes against it.
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if invalid_codes:
        df = df.replace(invalid_codes, np.nan)
    return df


MAX_PLAUSIBLE_WS = 100.0  # m/s - generous beyond any real-world reading (the fastest surface
# wind speeds ever recorded are around this level, in the most extreme tornadoes/cyclones on
# Earth); anything above it in a wind resource dataset is a sensor fault or placeholder value,
# not a real reading


def apply_physical_sanity_filter(df, cols_to_check, max_ws=MAX_PLAUSIBLE_WS):
    """A sensor fault or logging glitch can produce a garbage value that
    isn't one of the invalid-value codes actually entered (9999, -999,
    NaN, etc.) - a wind speed of hundreds or thousands of m/s, for
    instance, which silently wrecks a regression the same way one real
    outlier row wouldn't (an R2 of 0.02 dragged down entirely by 2 rows out
    of tens of thousands is exactly this failure mode, not a genuinely poor
    correlation). This catches what invalid-code matching can't: any wind
    speed reading above max_ws m/s or below 0, and any wind direction
    reading outside [0, 360) degrees, gets replaced with NaN regardless of
    what invalid codes were or weren't specified.

    cols_to_check: list of dicts, each optionally with 'ws_col' and/or
    'wd_col' (the actual column names to check) and a 'label' used only for
    the report (e.g. a height like '100 m', or a source name). Works
    unchanged for both a height_map-style list and a single modelled
    source, since both already share this same dict shape.

    Returns (cleaned_df, report) where report is a list of human-readable
    strings describing what was caught and where - empty if nothing was -
    so this never silently changes data without the caller being able to
    show the person what happened."""
    df = df.copy()
    report = []
    for entry in cols_to_check:
        label = entry.get("label", "")
        ws_col = entry.get("ws_col")
        if ws_col and ws_col in df.columns:
            bad = (df[ws_col] > max_ws) | (df[ws_col] < 0)
            n_bad = int(bad.sum())
            if n_bad:
                df.loc[bad, ws_col] = np.nan
                report.append(f"{n_bad} wind speed value(s)"
                               + (f" at {label}" if label else "")
                               + f" (column '{ws_col}') outside 0-{max_ws:.0f} m/s")
        wd_col = entry.get("wd_col")
        if wd_col and wd_col in df.columns:
            bad = (df[wd_col] >= 360) | (df[wd_col] < 0)
            n_bad = int(bad.sum())
            if n_bad:
                df.loc[bad, wd_col] = np.nan
                report.append(f"{n_bad} wind direction value(s)"
                               + (f" at {label}" if label else "")
                               + f" (column '{wd_col}') outside 0-360 degrees")
    return df, report


def diagnose_timestamp_parsing(raw_df, ts_col, dayfirst):
    """
    Parses the timestamp column both ways (dayfirst True/False) and compares
    row-drop / duplicate rates, so a wrong date-format choice (rows silently
    becoming NaT or colliding into duplicates) gets caught before analysis.
    """
    n = len(raw_df)

    def score(df_choice):
        ts = pd.to_datetime(raw_df[ts_col], dayfirst=df_choice, errors="coerce")
        return ts, ts.isna().sum(), ts.duplicated().sum()

    ts_chosen, na_chosen, dup_chosen = score(dayfirst)
    ts_other, na_other, dup_other = score(not dayfirst)

    bad_rate_chosen = (na_chosen + dup_chosen) / n
    bad_rate_other = (na_other + dup_other) / n

    return {
        "chosen_bad_rate": bad_rate_chosen, "chosen_na": na_chosen, "chosen_dup": dup_chosen,
        "other_bad_rate": bad_rate_other, "other_na": na_other, "other_dup": dup_other,
        "chosen_range": (ts_chosen.min(), ts_chosen.max()),
        "suggest_switch": bad_rate_other < bad_rate_chosen - 0.005 and bad_rate_chosen > 0.005,
    }


def timestamp_diagnostics_ui(raw_df, ts_col, dayfirst, key_prefix):
    diag = diagnose_timestamp_parsing(raw_df, ts_col, dayfirst)
    st.write(f"Parsed timestamp range: **{diag['chosen_range'][0]}** to "
             f"**{diag['chosen_range'][1]}** "
             f"({diag['chosen_na']} unparseable rows dropped, "
             f"{diag['chosen_dup']} duplicate timestamps).")
    if diag["suggest_switch"]:
        st.error(
            f"This looks like the wrong date format. With '{'day-first' if dayfirst else 'month-first'}' "
            f"selected, {100*diag['chosen_bad_rate']:.0f}% of rows are unparseable or duplicated. "
            f"Switching to '{'month-first' if dayfirst else 'day-first'}' only causes "
            f"{100*diag['other_bad_rate']:.0f}% - try toggling the checkbox above."
        )
    elif diag["chosen_bad_rate"] > 0.01:
        st.warning(f"{100*diag['chosen_bad_rate']:.1f}% of timestamp rows were dropped or "
                   f"duplicated - double check the date format and invalid-value codes.")


def detect_resolution_minutes(index):
    if len(index) < 2:
        return 10.0
    diffs = index.to_series().diff().dt.total_seconds().dropna() / 60
    med = diffs.median()
    return med if med and med > 0 else 10.0


# ==============================================================================
# HELPERS - DATA AVAILABILITY
# ==============================================================================

@st.cache_data(show_spinner="Calculating data availability...")
def availability_table(df, height_map):
    """Rows = calendar month, columns = height (sorted low to high). Values = % present."""
    total_per_month = df.resample("ME").size()
    rows = {}
    for hm in sorted_heights(height_map):
        col = hm["ws_col"]
        counts = df[col].resample("ME").count()
        pct = 100 * counts / total_per_month.replace(0, np.nan)
        rows[f"{hm['height']:.0f} m"] = pct
    table = pd.DataFrame(rows)
    table.index = table.index.strftime("%b-%Y")
    return table


def overall_availability(df, ws_col):
    return 100 * df[ws_col].notna().sum() / len(df)


def plot_availability_bars(table, threshold=80.0):
    heights = table.columns.tolist()
    n = len(heights)
    fig_w = min(max(8, 0.3 * len(table)), 14)
    # Bound height/width ratio directly so more heights can't make this run tall when
    # stretched to fill a wide container - this was the actual cause of "massive" before.
    fig_h = min(1.1 * n, 0.5 * fig_w, 7)
    fig, axes = plt.subplots(n, 1, figsize=(fig_w, fig_h), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, h in zip(axes, heights):
        vals = table[h].values
        colors = [ACCENT if v >= threshold else FLAG for v in vals]
        ax.bar(table.index, vals, color=colors, width=0.75)
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(0, 105)
        ax.set_ylabel(h, fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    axes[-1].set_xticks(range(len(table.index)))
    axes[-1].set_xticklabels(table.index, rotation=45, ha="right", fontsize=8)
    fig.suptitle(f"Data Availability by Height and Month (dashed = {threshold:.0f}% threshold)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ==============================================================================
# HELPERS - MONTHLY MEAN WIND SPEED
# ==============================================================================

@st.cache_data(show_spinner="Calculating monthly means...")
def monthly_mean_data(df, ws_col, min_day_fraction=0.5, samples_per_day=144):
    ws = df[ws_col]
    daily_count = ws.resample("D").count()
    valid_days = (daily_count >= samples_per_day * min_day_fraction).resample("ME").sum()
    monthly_mean = ws.resample("ME").mean()
    days_in_month = monthly_mean.index.days_in_month
    incomplete = (valid_days.reindex(monthly_mean.index) < days_in_month * 0.65).fillna(True)
    return monthly_mean, incomplete, ws.mean()


def render_monthly_fig(monthly_mean, incomplete, overall_mean, height_label, highlight_after=None):
    """highlight_after (optional): if given a timestamp, months ending after
    it are colored differently from the rest - used by the Wind Measurement
    Tracker's Continue Tracking flow to show at a glance which month(s) were
    just added versus what was already in a loaded package. None (the
    default) draws every bar the same way, unchanged from before this
    parameter existed - Long-Term Correction and Preliminary WRA, which
    also call this function, never pass it."""
    labels = monthly_mean.index.strftime("%b-%Y")
    fig_w = min(max(8, 0.35 * len(monthly_mean)), 14)
    fig_h = min(0.35 * fig_w, 5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    if highlight_after is not None:
        bar_colors = [NEW_HIGHLIGHT if m > highlight_after else ACCENT
                      for m in monthly_mean.index]
    else:
        bar_colors = ACCENT
    bars = ax.bar(labels, monthly_mean.values, color=bar_colors, width=0.7,
                   edgecolor="white", linewidth=0.5)

    for bar, flag in zip(bars, incomplete.values):
        if flag:
            ax.plot(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    marker="*", color=FLAG, markersize=8)

    ax.axhline(overall_mean, color="#444444", linestyle="--", linewidth=1.2,
               label=f"Overall mean = {overall_mean:.2f} m/s")
    if highlight_after is not None:
        ax.bar(0, 0, color=NEW_HIGHLIGHT, label="Newly added this update")
    ax.set_ylabel("Mean Wind Speed (m/s)")
    ax.set_title(f"Monthly Mean Wind Speed - {height_label}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ==============================================================================
# HELPERS - WIND ROSE
# ==============================================================================

def rose_source_data(meas_ws, meas_wd, model_ws, model_wd):
    """Builds independent (ws, wd) pairs for the measured and modelled panels
    of a wind rose comparison - deliberately NOT matched to concurrent
    timestamps or averaged to hourly. A wind rose describes the distribution
    of direction/speed over a period, which doesn't need paired, simultaneous
    samples the way a correlation does - and averaging a circular quantity
    like direction is worth avoiding wherever it isn't actually required.
    Each side just uses its own native-resolution data, dropping only its
    own missing values. Callers should already have sliced each series to
    whatever period they want represented - typically the overlap between
    the measurement campaign's own date range and the modelled dataset's
    coverage, so the modelled panel reflects the same period being measured
    rather than the model's entire multi-year history. Not cached - just a
    concat and dropna on data that's already been sliced, cheap enough that
    caching wouldn't add anything but a stale spinner message to maintain."""
    meas_combined = pd.concat([meas_ws.rename("ws"), meas_wd.rename("wd")], axis=1).dropna()
    model_combined = pd.concat([model_ws.rename("ws"), model_wd.rename("wd")], axis=1).dropna()
    return meas_combined, model_combined


def _rose_bins(ws, wd, n_dir_bins, speed_edges):
    dir_bin_width = 360 / n_dir_bins
    dir_bins = (np.floor((wd + dir_bin_width / 2) / dir_bin_width) % n_dir_bins).astype(int)
    freqs = []
    for i in range(len(speed_edges) - 1):
        lo, hi = speed_edges[i], speed_edges[i + 1]
        mask = (ws >= lo) & (ws < hi)
        counts = np.array([np.sum(mask & (dir_bins == d)) for d in range(n_dir_bins)])
        freqs.append(100 * counts / len(ws))
    return freqs, dir_bin_width


def _draw_rose(ax, ws, wd, n_dir_bins, n_speed_bins, title, speed_edges, title_pad=18):
    theta = np.deg2rad(np.arange(n_dir_bins) * (360 / n_dir_bins))
    freqs, dir_bin_width = _rose_bins(ws, wd, n_dir_bins, speed_edges)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    bottoms = np.zeros(n_dir_bins)
    cmap = plt.get_cmap("Blues")
    for i, freq in enumerate(freqs):
        lo, hi = speed_edges[i], speed_edges[i + 1]
        color = cmap(0.3 + 0.6 * i / max(1, len(freqs) - 1))
        ax.bar(theta, freq, width=np.deg2rad(dir_bin_width * 0.9), bottom=bottoms,
               label=f"{lo:.1f}-{hi:.1f} m/s", color=color, edgecolor="white", linewidth=0.3)
        bottoms += freq
    ax.set_title(title, pad=title_pad)
    ax.grid(True, alpha=0.4)
    return bottoms.max()


def render_rose_fig(meas_combined, model_combined, meas_label, model_label,
                     n_dir_bins=16, n_speed_bins=6):
    if len(meas_combined) < 2 or len(model_combined) < 2:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), subplot_kw={"projection": "polar"})
    all_ws = pd.concat([meas_combined["ws"], model_combined["ws"]])
    shared_edges = np.linspace(0, np.nanpercentile(all_ws, 99), n_speed_bins + 1)
    shared_edges[-1] = max(shared_edges[-1], all_ws.max()) + 0.1

    _draw_rose(axes[0], meas_combined["ws"], meas_combined["wd"], n_dir_bins, n_speed_bins,
               f"Measured - {meas_label} (n={len(meas_combined)})", shared_edges)
    _draw_rose(axes[1], model_combined["ws"], model_combined["wd"], n_dir_bins, n_speed_bins,
               f"Modelled - {model_label} (n={len(model_combined)})", shared_edges)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_speed_bins, fontsize=8,
               bbox_to_anchor=(0.5, -0.05), frameon=False)
    fig.suptitle("Wind Rose Comparison (independent periods, not concurrent-matched)", y=1.03)
    fig.tight_layout()
    return fig


def render_monthly_rose_grid_fig(ws, wd, title, n_dir_bins=12, n_speed_bins=5, max_cols=6):
    """A grid of small wind roses, one per calendar month - built specifically
    to spot directional shifts across the tracked period, which neither an
    overall single rose (collapses every month together) nor a monthly mean
    direction (collapses an entire month's distribution to one number, and
    can be actively misleading for a spread-out or bimodal distribution)
    can show. Both the speed-bin colors AND the radial (percentage) scale
    are shared across every month's subplot, not each auto-scaled to its
    own peak - without that, a visually 'long' bar in one month could
    represent a smaller actual percentage than a visually 'short' bar in
    another, which would make the whole point of comparing months across
    the grid misleading rather than useful."""
    combined = pd.concat([ws.rename("ws"), wd.rename("wd")], axis=1).dropna()
    if len(combined) < 2:
        return None
    periods = combined.index.to_period("M")
    months = sorted(periods.unique())
    n_months = len(months)
    if n_months == 0:
        return None
    ncols = min(max_cols, n_months)
    nrows = int(np.ceil(n_months / ncols))

    shared_edges = np.linspace(0, np.nanpercentile(combined["ws"], 99), n_speed_bins + 1)
    shared_edges[-1] = max(shared_edges[-1], combined["ws"].max()) + 0.1

    fig, axes = plt.subplots(nrows, ncols, figsize=(1.9 * ncols, 2.3 * nrows),
                              subplot_kw={"projection": "polar"})
    axes_flat = np.atleast_1d(axes).flatten()

    peak_values = []
    for i, period in enumerate(months):
        ax = axes_flat[i]
        month_data = combined[periods == period]
        peak = _draw_rose(ax, month_data["ws"], month_data["wd"], n_dir_bins, n_speed_bins,
                           period.strftime("%b-%Y"), shared_edges, title_pad=4)
        peak_values.append(peak)
        ax.title.set_fontsize(8)
        ax.tick_params(labelsize=5)
        ax.set_xticklabels([])

    shared_rmax = max(peak_values) * 1.05 if peak_values else 1.0
    for i in range(n_months):
        axes_flat[i].set_rlim(0, shared_rmax)
    for j in range(n_months, len(axes_flat)):
        axes_flat[j].axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(n_speed_bins, 6), fontsize=7,
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.suptitle(f"Monthly Wind Rose - {title}", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    return fig


@st.cache_data(show_spinner="Building monthly wind rose grid...")
def render_monthly_rose_grid_png(ws, wd, title, n_dir_bins=12, n_speed_bins=5, max_cols=6):
    """Cached wrapper around render_monthly_rose_grid_fig, returning PNG bytes
    rather than a Figure. Profiling showed the actual matplotlib rendering
    (creating and drawing many small polar subplots) - not the numeric
    binning, which is comparatively trivial - is what makes this expensive:
    several seconds for a realistic multi-year dataset. Since this function
    wasn't cached, Streamlit was rebuilding the whole grid from scratch on
    EVERY rerun of the entire app script - including ones triggered by
    typing in a completely unrelated field elsewhere, like Long-Term
    Analysis - which is what made the app feel like it was stuck on the
    wind rose before anything downstream would respond. Returns bytes
    rather than a Figure specifically because a cached Figure object could
    later be closed by a caller (show_fig does exactly this) and then be
    broken on the next cache hit - bytes carry no such risk."""
    fig = render_monthly_rose_grid_fig(ws, wd, title, n_dir_bins, n_speed_bins, max_cols)
    if fig is None:
        return None
    return fig_to_png_bytes(fig)


def render_monthly_ti_vs_ws_grid_fig(df, ws_col, ti_col, height_label, ws_bin_width=1.0,
                                      max_cols=4):
    """Grid of TI-vs-wind-speed plots, one per calendar month - the standard
    way turbulence intensity is characterized in the wind industry (TI
    varies systematically with wind speed - typically higher and noisier
    at low speeds, decreasing and flattening at higher ones), rather than
    a single monthly average value that hides this relationship entirely.
    Both axes are shared across every month's subplot for fair visual
    comparison, matching the same principle already used for the monthly
    wind rose grid."""
    combined = pd.concat([df[ws_col].rename("ws"), df[ti_col].rename("ti")], axis=1).dropna()
    combined = combined[combined["ws"] >= 1.0]  # TI is noisy/not meaningful near-zero wind speed
    if len(combined) < 2:
        return None
    periods = combined.index.to_period("M")
    months = sorted(periods.unique())
    n_months = len(months)
    if n_months == 0:
        return None
    ncols = min(max_cols, n_months)
    nrows = int(np.ceil(n_months / ncols))
    ws_shared_max = combined["ws"].quantile(0.995)

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.3 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    bin_means_all = []
    for i, period in enumerate(months):
        ax = axes_flat[i]
        month_data = combined[periods == period]
        bin_centers = (month_data["ws"] // ws_bin_width) * ws_bin_width + ws_bin_width / 2
        grouped = month_data.groupby(bin_centers)["ti"].agg(["mean", "count"])
        grouped = grouped[grouped["count"] >= 3]  # drop bins too sparse to trust
        if len(grouped) > 0:
            ax.plot(grouped.index, grouped["mean"], marker="o", markersize=3,
                    color=ACCENT, linewidth=1.3)
            bin_means_all.append(grouped["mean"])
        ax.set_title(period.strftime("%b-%Y"), fontsize=9, pad=4)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    for j in range(n_months, len(axes_flat)):
        axes_flat[j].axis("off")

    if bin_means_all:
        shared_ti_max = max(s.max() for s in bin_means_all) * 1.15
        for i in range(n_months):
            axes_flat[i].set_ylim(0, shared_ti_max)
            axes_flat[i].set_xlim(0, ws_shared_max)

    fig.supxlabel("Wind Speed (m/s)", fontsize=10)
    fig.supylabel("Turbulence Intensity (%)", fontsize=10)
    fig.suptitle(f"Monthly TI vs Wind Speed - {height_label}", fontsize=12, y=1.0)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.96])
    return fig


@st.cache_data(show_spinner="Building monthly TI vs wind speed grid...")
def render_monthly_ti_vs_ws_grid_png(df, ws_col, ti_col, height_label, ws_bin_width=1.0,
                                      max_cols=4):
    fig = render_monthly_ti_vs_ws_grid_fig(df, ws_col, ti_col, height_label, ws_bin_width,
                                            max_cols)
    if fig is None:
        return None
    return fig_to_png_bytes(fig)


@st.cache_data(show_spinner="Calculating cumulative convergence...")
def running_cumulative_mean_by_month(hourly_series):
    """For each calendar month boundary present in the data, the cumulative
    mean of every hourly value from the very start of the record through
    the end of that month - i.e. 'what would my average look like if I'd
    stopped collecting data at this point' - not each month's own isolated
    mean, which wouldn't show convergence toward anything."""
    s = hourly_series.dropna()
    if len(s) == 0:
        return pd.Series(dtype=float)
    month_ends = s.resample("ME").mean().index
    return pd.Series([s.loc[:me].mean() for me in month_ends], index=month_ends)


def render_convergence_fig(cum_means, lt_reference, height_label):
    """The long-term reference value is computed ONCE, from a single
    regression fit using every concurrent hour currently available - it
    does not change as the plot is read left to right. What's being tracked
    is whether the running, ever-more-complete measured average is
    settling in near that fixed reference as more months accumulate, not
    whether the regression itself is stabilizing."""
    labels = cum_means.index.strftime("%b-%Y")
    fig_w = min(max(8, 0.35 * len(cum_means)), 14)
    fig_h = min(0.35 * fig_w, 5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.plot(labels, cum_means.values, marker="o", color=ACCENT, linewidth=1.8,
            markeredgecolor="white", markeredgewidth=0.5, label="Cumulative measured mean")
    ax.axhline(lt_reference, color=FLAG, linestyle="--", linewidth=1.5,
               label=f"Long-term estimate = {lt_reference:.2f} m/s")
    ax.set_ylabel("Wind Speed (m/s)")
    ax.set_title(f"Long-Term Convergence - {height_label}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def build_tracker_package(combined_df, height_map, extra_config):
    """Serializes the combined measurement dataset plus configuration into a
    single downloadable ZIP - the 'transferable package' a user downloads
    at the end of a session and re-uploads next month to continue where
    they left off, rather than needing to re-upload every raw file from
    every prior month and redo every column mapping each time. CSV rather
    than a binary format like Parquet specifically to avoid adding a new
    dependency (pyarrow) for something a plain CSV handles perfectly well
    at this data scale, and because it stays directly inspectable if
    something ever needs checking by hand."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        csv_buf = io.StringIO()
        out_df = combined_df.copy()
        out_df.index.name = "Timestamp"
        out_df.to_csv(csv_buf)
        zf.writestr("measurements.csv", csv_buf.getvalue())
        config = {"package_version": 1, "height_map": height_map, **extra_config}
        zf.writestr("config.json", json.dumps(config, indent=2, default=str))
    buf.seek(0)
    return buf.read()


def load_tracker_package(file_bytes):
    """Reads back a package built by build_tracker_package. Raises a plain
    ValueError (caught and shown as st.error by the caller) rather than
    letting a cryptic zipfile/KeyError surface if someone uploads an
    unrelated ZIP file here."""
    zf = zipfile.ZipFile(io.BytesIO(file_bytes))
    if "measurements.csv" not in zf.namelist() or "config.json" not in zf.namelist():
        raise ValueError("This doesn't look like a Wind Measurement Tracker package - "
                          "missing measurements.csv or config.json.")
    meas_df = pd.read_csv(zf.open("measurements.csv"))
    meas_df["Timestamp"] = pd.to_datetime(meas_df["Timestamp"], errors="coerce")
    meas_df = meas_df.dropna(subset=["Timestamp"]).set_index("Timestamp")
    config = json.loads(zf.read("config.json"))
    return meas_df, config


# ==============================================================================
# HELPERS - WIND MEASUREMENT TRACKER FILE FORMATS (TOA5 .dat/.sta, NetCDF)
# ==============================================================================

def sniff_toa5(file_bytes):
    """Campbell Scientific TOA5 files start with a literal 'TOA5' token
    (quoted) as the first field of line 1."""
    head = file_bytes[:20].decode("utf-8", errors="ignore")
    return "TOA5" in head.upper()


def parse_toa5(file_bytes):
    """Parses a Campbell Scientific TOA5-format .dat/.sta file: a 4-line
    ASCII header (station/logger info, field names, units, aggregation
    type), then comma-delimited data from line 5, first column always
    TIMESTAMP (ISO-like, unambiguous - no dayfirst setting needed), second
    column RECORD (a sequential integer, dropped here - not physically
    meaningful for wind analysis). Missing values are recorded as the
    literal string 'NAN' in this format, which pd.to_numeric already
    converts to a real NaN without any special-casing needed."""
    text = file_bytes.decode("utf-8", errors="ignore")
    rows = list(csv.reader(text.splitlines()))
    if len(rows) < 5 or not rows[0] or "TOA5" not in rows[0][0].upper():
        raise ValueError("Not a recognized TOA5 file (missing the 'TOA5' signature on line 1).")
    field_names = rows[1]
    units = dict(zip(field_names, rows[2]))
    data_rows = [r for r in rows[4:] if len(r) == len(field_names)]
    skipped = len(rows) - 4 - len(data_rows)
    df = pd.DataFrame(data_rows, columns=field_names)
    if "RECORD" in df.columns:
        df = df.drop(columns=["RECORD"])
    return df, units, skipped


def sniff_generic_dat(file_bytes):
    """A second, distinct .dat/.sta layout seen in practice (not a TOA5
    variant - a completely different structure, e.g. from floating LiDAR
    buoy systems): a short title/identifier line, then an UNQUOTED comma-
    separated header row whose first field is literally 'timestamp', often
    followed by a units row, then unquoted comma-separated data. Detected
    by scanning the first several lines for a comma-separated row
    containing that literal field, the same way Vortex .txt parsing scans
    for its own header row rather than assuming a fixed line count -
    different vendors' export layouts aren't guaranteed to agree on how
    many lines precede the real data."""
    text = file_bytes[:4000].decode("utf-8", errors="ignore")
    for line in text.splitlines()[:10]:
        fields = [f.strip().strip('"').lower() for f in line.split(",")]
        if "timestamp" in fields:
            return True
    return False


def parse_generic_dat(file_bytes):
    """Parses the generic (non-TOA5) .dat/.sta layout described in
    sniff_generic_dat. The line immediately after the header is treated as
    a units row (and skipped) only if its first field looks like a date-
    format descriptor (contains YYYY/MM/DD/HH) rather than actual data -
    that descriptor also gives a reliable dayfirst signal, used as the
    default (still user-editable in the UI, since this is inferred, not
    guaranteed). Note: a real-world sample showed the units row can have
    FEWER fields than the header/data rows (an inconsistency in that
    vendor's own export) - this is tolerated here since units aren't used
    for anything load-bearing, only the header-to-data alignment is."""
    text = file_bytes.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines[:10]):
        fields = [f.strip().strip('"').lower() for f in line.split(",")]
        if "timestamp" in fields:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find a header row containing 'timestamp' in this file.")

    field_names = [f.strip().strip('"') for f in lines[header_idx].split(",")]
    data_start = header_idx + 1
    dayfirst = True
    if data_start < len(lines):
        next_fields = lines[data_start].split(",")
        first_field = next_fields[0].strip().strip('"').upper()
        if any(tok in first_field for tok in ("YYYY", "MM", "DD", "HH")):
            data_start += 1
            dayfirst = "DD" in first_field and (
                first_field.index("DD") < first_field.index("MM") if "MM" in first_field else True)

    data_lines = [ln for ln in lines[data_start:] if ln.strip()]
    good_lines = [ln for ln in data_lines if len(ln.split(",")) == len(field_names)]
    skipped = len(data_lines) - len(good_lines)
    data_text = "\n".join(good_lines)
    df = pd.read_csv(io.StringIO(data_text), names=field_names, header=None) if good_lines \
        else pd.DataFrame(columns=field_names)
    return df, dayfirst, skipped


def list_netcdf_variables(file_bytes):
    """Lists every variable in a NetCDF file with its dimensions and units,
    for the user to map onto Timestamp/WS/WD/TI roles - NetCDF is self-
    describing, so unlike .dat/.sta this doesn't need a hardcoded format
    assumption, just an introspection step."""
    import netCDF4
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        ds = netCDF4.Dataset(tmp_path, "r")
        info = {name: {"dims": var.dimensions, "shape": var.shape,
                        "units": getattr(var, "units", ""),
                        "long_name": getattr(var, "long_name", "")}
                for name, var in ds.variables.items()}
        ds.close()
    finally:
        os.unlink(tmp_path)
    return info


@st.cache_data(show_spinner="Reading NetCDF file...")
def read_netcdf_to_df(file_bytes, time_var_name, value_var_names):
    """Reads a NetCDF file into a flat DataFrame with a real Timestamp
    column, given a chosen time variable and a list of value variables to
    keep. Only handles 1-D (time-indexed) variables - a single-height
    dataset - since that's the common case for a met-buoy or single-level
    station export; a multi-dimensional (height-resolved) NetCDF would need
    a different reader, deliberately not guessed at without a real example."""
    import netCDF4
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        ds = netCDF4.Dataset(tmp_path, "r")
        tvar = ds.variables[time_var_name]
        raw_times = netCDF4.num2date(tvar[:], units=tvar.units,
                                      calendar=getattr(tvar, "calendar", "standard"))
        data = {"Timestamp": pd.to_datetime([t.isoformat() for t in raw_times])}
        for vname in value_var_names:
            v = ds.variables[vname]
            if len(v.dimensions) != 1:
                raise ValueError(f"'{vname}' has {len(v.dimensions)} dimensions "
                                  f"{v.dimensions} - only 1-D (time-only) variables "
                                  f"are supported here.")
            arr = np.ma.filled(np.ma.masked_invalid(v[:]), np.nan).astype(float)
            fill = getattr(v, "_FillValue", None)
            if fill is not None:
                arr = np.where(np.isclose(arr, float(fill)), np.nan, arr)
            data[vname] = arr
        ds.close()
    finally:
        os.unlink(tmp_path)
    return pd.DataFrame(data)


# ==============================================================================
# HELPERS - SHEAR (PROFILE METHOD)
# ==============================================================================

@st.cache_data(show_spinner="Fitting shear profile...")
def compute_shear_data(df, height_map, min_availability=80.0):
    hm_sorted = sorted_heights(height_map)
    avail = {hm["height"]: overall_availability(df, hm["ws_col"]) for hm in hm_sorted}
    used = [hm for hm in hm_sorted if avail[hm["height"]] >= min_availability]
    excluded = {hm["height"]: round(avail[hm["height"]], 1) for hm in hm_sorted if hm not in used}

    if len(used) < 3:
        return None

    heights_used = [hm["height"] for hm in used]
    mean_ws = np.array([df[hm["ws_col"]].mean() for hm in used])

    log_z = np.log(heights_used)
    log_ws = np.log(mean_ws)
    slope, intercept, r, p, se = stats.linregress(log_z, log_ws)

    return {
        "alpha": slope, "intercept": intercept, "r2": r ** 2,
        "heights_used": heights_used, "mean_ws": mean_ws, "excluded": excluded,
    }


def render_shear_fig(shear_data):
    heights_used = shear_data["heights_used"]
    mean_ws = shear_data["mean_ws"]
    slope, intercept = shear_data["alpha"], shear_data["intercept"]

    z_min = max(1, min(heights_used) * 0.05)
    z_max = max(heights_used) * 1.3
    z_smooth = np.linspace(z_min, z_max, 200)
    ws_ref = np.exp(intercept) * heights_used[0] ** slope
    ws_smooth = ws_ref * (z_smooth / heights_used[0]) ** slope

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot(ws_smooth, z_smooth, color="#2f9e44", linewidth=2.2, label="Fitted profile")
    ax.plot(mean_ws, heights_used, "D", color="navy", markersize=8, label="Overall wind speed",
            zorder=5)
    ax.set_xlabel("Wind Speed [m/s]")
    ax.set_ylabel("Height [m]")
    ax.set_title("Predicted Vertical Wind Profile")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


def render_long_term_fig(shear_data, model_height, model_label, lt_ws_at_model_height,
                          interest_height, lt_ws_at_interest):
    """Shows the fitted shear profile with the long-term corrected wind speed
    highlighted at both the modelled dataset's height and the height of interest,
    so the final number has a clear visual anchor rather than just a metric card."""
    heights_used = shear_data["heights_used"]
    mean_ws = shear_data["mean_ws"]
    slope, intercept = shear_data["alpha"], shear_data["intercept"]

    all_heights = heights_used + [model_height, interest_height]
    z_min = max(1, min(all_heights) * 0.05)
    z_max = max(all_heights) * 1.2
    z_smooth = np.linspace(z_min, z_max, 200)
    ws_ref = np.exp(intercept) * heights_used[0] ** slope
    ws_smooth = ws_ref * (z_smooth / heights_used[0]) ** slope

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.plot(ws_smooth, z_smooth, color="#2f9e44", linewidth=2, label="Fitted shear profile",
            zorder=2)
    ax.plot(mean_ws, heights_used, "D", color="navy", markersize=7,
            label="Measured (period mean)", zorder=3)
    ax.plot(lt_ws_at_model_height, model_height, "o", color="#888888", markersize=10,
            label=f"Long-term at {model_label} height", zorder=4)
    ax.plot(lt_ws_at_interest, interest_height, "*", color=FLAG, markersize=22,
            label="Long-term at height of interest", zorder=5)
    ax.annotate(f"{lt_ws_at_interest:.2f} m/s @ {interest_height:.0f} m",
                xy=(lt_ws_at_interest, interest_height), xytext=(10, 8),
                textcoords="offset points", fontsize=10, fontweight="bold", color=FLAG)
    ax.set_xlabel("Wind Speed [m/s]")
    ax.set_ylabel("Height [m]")
    ax.set_title("Long-Term Wind Speed at Height of Interest")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner="Matching measurement height to the modelled dataset...")
def get_measurement_at_target_height(meas_series_by_height, shear_data, target_height, tolerance=2.0):
    for h, (series, _) in meas_series_by_height.items():
        if abs(h - target_height) <= tolerance:
            return series, f"direct measurement at {h:.0f} m", h

    if shear_data is None:
        return None, ("no measured height matches the target height and no shear exponent "
                       "is available to extrapolate (need >=3 heights at the chosen "
                       "availability threshold)"), None

    heights_used = shear_data["heights_used"]
    alpha = shear_data["alpha"]
    ref_h = min(heights_used, key=lambda h: abs(h - target_height))
    ref_series = meas_series_by_height[ref_h][0]
    extrapolated = ref_series * (target_height / ref_h) ** alpha
    desc = f"extrapolated from {ref_h:.0f} m using shear alpha={alpha:.3f}"
    return extrapolated, desc, ref_h


# ==============================================================================
# HELPERS - RESAMPLE / MERGE / CORRELATION
# ==============================================================================

@st.cache_data(show_spinner="Resampling to hourly...")
def resample_to_hourly(ws, samples_per_hour, min_fraction=0.5):
    hourly_mean = ws.resample("h").mean()
    hourly_count = ws.resample("h").count()
    hourly_mean[hourly_count < samples_per_hour * min_fraction] = np.nan
    return hourly_mean


@st.cache_data(show_spinner="Merging concurrent timestamps...")
def merge_concurrent(meas_hourly, model_series):
    merged = pd.concat([meas_hourly, model_series], axis=1, join="inner").dropna()
    merged.columns = ["Meas", "Model"]
    return merged


@st.cache_data(show_spinner="Building daily / monthly averages...")
def build_daily_monthly(merged, min_hours_per_day=18, min_month_fraction=0.5):
    daily = merged.resample("D").agg(["mean", "count"])
    daily_ok = daily[(daily[("Meas", "count")] >= min_hours_per_day) &
                      (daily[("Model", "count")] >= min_hours_per_day)]
    daily_avg = daily_ok.xs("mean", axis=1, level=1)

    monthly = merged.resample("ME").agg(["mean", "count"])
    thresh = 24 * 28 * min_month_fraction
    monthly_ok = monthly[(monthly[("Meas", "count")] >= thresh) &
                          (monthly[("Model", "count")] >= thresh)]
    monthly_avg = monthly_ok.xs("mean", axis=1, level=1)
    return daily_avg, monthly_avg


def correlation_fig(merged, label, model_label):
    if len(merged) < 2:
        return None, None
    slope, intercept, r, p, se = stats.linregress(merged["Model"], merged["Meas"])
    r2 = r ** 2

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.scatter(merged["Model"], merged["Meas"], s=12, alpha=0.45, color=ACCENT,
               edgecolor="none")
    xr = np.linspace(merged["Model"].min(), merged["Model"].max(), 50)
    ax.plot(xr, slope * xr + intercept, color=FLAG, linewidth=2, label=f"OLS fit (R2={r2:.2f})")
    ax.set_xlabel(f"{model_label} Wind Speed (m/s)")
    ax.set_ylabel("Measured Wind Speed (m/s)")
    ax.set_title(label)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    stats_dict = {"n": len(merged), "R": r, "R2": r2, "slope": slope, "intercept": intercept}
    return fig, stats_dict


def orthogonal_regression(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    xc, yc = x - x_mean, y - y_mean
    cov = np.cov(np.vstack([xc, yc]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    slope = principal[1] / principal[0]
    intercept = y_mean - slope * x_mean
    return slope, intercept


@st.cache_data(show_spinner="Running long-term regression...")
def long_term_correction(merged, model_full_series):
    x, y = merged["Model"].values, merged["Meas"].values
    slope_ols, intercept_ols, r, p, se = stats.linregress(x, y)
    slope_tls, intercept_tls = orthogonal_regression(x, y)

    lt_model = model_full_series.dropna()
    lt_ws_ols = slope_ols * lt_model + intercept_ols
    lt_ws_tls = slope_tls * lt_model + intercept_tls

    return {
        "concurrent_meas_mean": y.mean(), "concurrent_model_mean": x.mean(),
        "lt_model_mean": lt_model.mean(), "lt_model_start": lt_model.index.min(),
        "lt_model_end": lt_model.index.max(), "n_concurrent": len(merged),
        "ols": {"slope": slope_ols, "intercept": intercept_ols, "lt_mean": lt_ws_ols.mean()},
        "tls": {"slope": slope_tls, "intercept": intercept_tls, "lt_mean": lt_ws_tls.mean()},
    }


# ==============================================================================
# STREAMLIT UI
# ==============================================================================

# ==============================================================================
# MEASUREMENT CAMPAIGN PLANNING - HELPERS
# (geopandas / scikit-learn / plotly are imported lazily inside these functions,
# so the Long-Term Correction mode doesn't pay their import cost on every rerun)
# ==============================================================================

def read_ascii_grid(file_bytes):
    """
    Parses an ESRI ASCII grid (.asc) file - a 6-line header (ncols, nrows,
    xllcorner, yllcorner, cellsize, NODATA_value) followed by a nrows x ncols
    matrix of values. This is the format Vortex (and most GIS tools) export
    wind speed maps in.
    """
    text = file_bytes.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    header = {}
    for line in lines[:6]:
        parts = line.split()
        if len(parts) >= 2:
            header[parts[0].lower()] = float(parts[1])

    data_text = "\n".join(lines[6:])
    data = np.loadtxt(io.StringIO(data_text))
    if "nodata_value" in header:
        data[data == header["nodata_value"]] = np.nan

    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    x0 = header["xllcorner"]
    y0 = header["yllcorner"]
    cs = header["cellsize"]
    extent = [x0, x0 + ncols * cs, y0, y0 + nrows * cs]

    meta = {"ncols": ncols, "nrows": nrows, "xllcorner": x0, "yllcorner": y0,
            "cellsize": cs, "extent": extent}
    return data, meta


def sample_raster(data, meta, lat, lon):
    """Looks up the raster value at a given (lat, lon), or NaN if out of bounds
    / on a nodata cell. Mirrors the original tool's row/col indexing exactly
    (row is flipped since the raster is stored top-to-bottom, origin='upper')."""
    try:
        col = int((lon - meta["xllcorner"]) / meta["cellsize"])
        row = int((lat - meta["yllcorner"]) / meta["cellsize"])
        row = meta["nrows"] - 1 - row
        if row < 0 or row >= meta["nrows"] or col < 0 or col >= meta["ncols"]:
            return np.nan
        val = data[row, col]
        return val
    except Exception:
        return np.nan


def _kml_tag(elem):
    """Strips the XML namespace off a tag name (e.g. '{http://www.opengis.net/
    kml/2.2}Polygon' -> 'Polygon') so parsing works regardless of which KML
    namespace URI/version a given file declares, or whether it declares one
    at all."""
    return elem.tag.split("}", 1)[-1] if "}" in elem.tag else elem.tag


def parse_kml_polygon(file_bytes):
    """Extracts boundary polygon(s) from a KML file using Python's built-in
    XML parser rather than GDAL/pyogrio's KML/LIBKML driver. This is
    deliberate: pyogrio only added libkml to its bundled wheels recently and
    its coverage still varies by platform/version, so relying on it would
    make KML support silently dependent on exactly which pyogrio build ends
    up installed - a standard-library XML parser has no such dependency risk
    and needs nothing added to requirements.txt.

    Returns a GeoDataFrame (EPSG:4326, KML's native CRS) with one row.
    Handles multiple Polygons in the file (e.g. several Placemarks) by
    unioning them; if that union is disjoint (a genuine MultiPolygon), the
    largest piece is kept, since the rest of this tool assumes a single
    simple boundary."""
    import geopandas as gpd
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    root = ET.fromstring(file_bytes)

    polygons = []
    for poly_elem in root.iter():
        if _kml_tag(poly_elem) != "Polygon":
            continue
        outer_coords, holes = None, []
        for boundary_elem in poly_elem.iter():
            btag = _kml_tag(boundary_elem)
            if btag not in ("outerBoundaryIs", "innerBoundaryIs"):
                continue
            coord_elem = next((e for e in boundary_elem.iter() if _kml_tag(e) == "coordinates"),
                               None)
            if coord_elem is None or not coord_elem.text:
                continue
            pts = []
            for triplet in coord_elem.text.strip().split():
                lon_str, lat_str = triplet.split(",")[:2]
                pts.append((float(lon_str), float(lat_str)))
            if btag == "outerBoundaryIs":
                outer_coords = pts
            else:
                holes.append(pts)
        if outer_coords and len(outer_coords) >= 3:
            polygons.append(Polygon(outer_coords, holes) if holes else Polygon(outer_coords))

    if not polygons:
        raise ValueError("No <Polygon> geometry found in this KML file - only point/line "
                          "placemarks, or an unsupported KML structure.")
    geom = polygons[0] if len(polygons) == 1 else unary_union(polygons)
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda p: p.area)
    if not geom.is_valid:
        geom = geom.buffer(0)
    return gpd.GeoDataFrame({"name": ["boundary"]}, geometry=[geom], crs="EPSG:4326")


def read_geo_file_from_bytes(file_bytes, suffix):
    """
    Reads a geospatial file (GeoJSON/GeoPackage/KML) via geopandas, or the
    standard-library KML parser above for .kml specifically. Writes to a
    real temp file first rather than reading from memory - GeoPackage (SQLite-
    based) in particular needs an actual file on disk, and this is uniformly
    reliable across formats.
    """
    if suffix.lower() == ".kml":
        return parse_kml_polygon(file_bytes)
    import geopandas as gpd
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        gdf = gpd.read_file(tmp_path)
    finally:
        os.unlink(tmp_path)
    return gdf


def utm_epsg_code(zone, hemisphere):
    """WGS84 UTM EPSG code: 326xx for the Northern hemisphere, 327xx for the
    Southern, where xx is the UTM zone number (1-60) - the standard EPSG
    numbering scheme for WGS84 UTM projected CRSs."""
    return (32600 if hemisphere == "Northern" else 32700) + int(zone)


def easting_northing_to_latlon(eastings, northings, source_epsg):
    """Converts Easting/Northing arrays (in the projected CRS identified by
    source_epsg) to WGS84 longitude/latitude via pyproj - already a
    mandatory dependency of geopandas (not an optional extra), so this needs
    nothing new in requirements.txt. always_xy=True keeps input/output order
    as (x, y) = (easting, northing) / (lon, lat), avoiding pyproj's
    sometimes axis-swapped default for certain CRS definitions."""
    from pyproj import Transformer
    transformer = Transformer.from_crs(f"EPSG:{source_epsg}", "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(np.asarray(eastings, dtype=float),
                                        np.asarray(northings, dtype=float))
    return lons, lats


@st.cache_data(show_spinner="Converting Easting/Northing to Latitude/Longitude...")
def read_layout_from_en(file_bytes, filename, easting_col, northing_col, source_epsg):
    """Reads a turbine layout from a CSV/XLSX of Easting/Northing POINTS (row
    order doesn't matter - each row is an independent turbine position,
    unlike a boundary) in the given projected CRS, converting to WGS84
    lon/lat."""
    import geopandas as gpd
    df = (pd.read_excel(io.BytesIO(file_bytes)) if filename.lower().endswith(".xlsx")
          else pd.read_csv(io.BytesIO(file_bytes)))
    eastings = pd.to_numeric(df[easting_col], errors="coerce").to_numpy()
    northings = pd.to_numeric(df[northing_col], errors="coerce").to_numpy()
    valid = ~(np.isnan(eastings) | np.isnan(northings))
    if not valid.all():
        df = df.loc[valid].reset_index(drop=True)
        eastings, northings = eastings[valid], northings[valid]
    if len(eastings) == 0:
        raise ValueError("No valid Easting/Northing values found in the chosen columns.")
    lons, lats = easting_northing_to_latlon(eastings, northings, source_epsg)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326")


@st.cache_data(show_spinner="Converting Easting/Northing to Latitude/Longitude...")
def read_boundary_from_en(file_bytes, filename, easting_col, northing_col, source_epsg):
    """Reads a site boundary from a CSV/XLSX of Easting/Northing VERTEX
    points, in the given projected CRS, converting to WGS84 lon/lat and
    connecting them IN FILE ROW ORDER into a closed polygon. Unlike a
    turbine layout, order is load-bearing here - this assumes rows are
    already listed in boundary-walking order, which is how survey/CAD
    exports of a boundary point list are conventionally given (the UI warns
    about this explicitly, since connecting out-of-order points would
    silently produce a self-intersecting/wrong shape rather than an error)."""
    import geopandas as gpd
    from shapely.geometry import Polygon
    df = (pd.read_excel(io.BytesIO(file_bytes)) if filename.lower().endswith(".xlsx")
          else pd.read_csv(io.BytesIO(file_bytes)))
    eastings = pd.to_numeric(df[easting_col], errors="coerce").to_numpy()
    northings = pd.to_numeric(df[northing_col], errors="coerce").to_numpy()
    valid = ~(np.isnan(eastings) | np.isnan(northings))
    eastings, northings = eastings[valid], northings[valid]
    if len(eastings) < 3:
        raise ValueError("Need at least 3 valid Easting/Northing rows to form a boundary polygon.")
    lons, lats = easting_northing_to_latlon(eastings, northings, source_epsg)
    geom = Polygon(zip(lons, lats))
    if not geom.is_valid:
        geom = geom.buffer(0)
    return gpd.GeoDataFrame({"name": ["boundary"]}, geometry=[geom], crs="EPSG:4326")


def read_layout_file(file_bytes, filename, lat_col=None, lon_col=None):
    """Reads a turbine layout from .geojson/.gpkg (point geometries already
    present) or .xlsx (Latitude/Longitude columns, name configurable)."""
    import geopandas as gpd
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "xlsx":
        df = pd.read_excel(io.BytesIO(file_bytes))
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
        return gdf
    else:
        return read_geo_file_from_bytes(file_bytes, f".{ext}")


@st.cache_data(show_spinner="Building clickable grid inside the boundary...")
def generate_clickable_grid(boundary_wkt, bounds, n_grid=45):
    """
    A grid of points covering the boundary's interior, used purely to capture
    click coordinates - Plotly/Streamlit's on_select event only fires reliably
    on scatter markers, not on heatmap/image pixels (a confirmed Streamlit
    limitation), so this invisible marker layer is what makes "click on the
    map to place a point" work at all.
    """
    from shapely import wkt as shapely_wkt
    poly_shape = shapely_wkt.loads(boundary_wkt)
    poly_path = MplPath(np.array(poly_shape.exterior.coords))

    lons = np.linspace(bounds[0], bounds[2], n_grid)
    lats = np.linspace(bounds[1], bounds[3], n_grid)
    glon, glat = np.meshgrid(lons, lats)
    glon, glat = glon.ravel(), glat.ravel()
    inside = poly_path.contains_points(np.column_stack([glon, glat]))
    return glon[inside], glat[inside]


def boundary_path(boundary_gdf):
    return MplPath(np.array(boundary_gdf.geometry.iloc[0].exterior.coords))


def build_planning_map_fig(active_map, boundary_gdf, layout_gdf, show_layout_ws,
                            measurement_points, best_points, click_grid):
    """Builds the interactive map: wind map heatmap (visual only), boundary
    outline, layout markers, placed measurement points (A, B, C...), best
    points (stars), and an invisible clickable grid layer."""
    import plotly.graph_objects as go

    fig = go.Figure()

    if active_map is not None:
        data, meta = active_map
        x = np.linspace(meta["extent"][0], meta["extent"][1], meta["ncols"])
        y = np.linspace(meta["extent"][2], meta["extent"][3], meta["nrows"])
        fig.add_trace(go.Heatmap(x=x, y=y, z=data, colorscale="Viridis",
                                  hoverinfo="skip", showscale=True,
                                  colorbar=dict(title="m/s")))

    if boundary_gdf is not None:
        bx, by = boundary_gdf.geometry.iloc[0].exterior.coords.xy
        fig.add_trace(go.Scatter(x=list(bx), y=list(by), mode="lines",
                                  line=dict(color="black", width=2),
                                  showlegend=False, hoverinfo="skip"))

    if layout_gdf is not None:
        lx = [g.x for g in layout_gdf.geometry]
        ly = [g.y for g in layout_gdf.geometry]
        text = None
        if show_layout_ws and active_map is not None:
            data, meta = active_map
            ws_vals = [sample_raster(data, meta, la, lo) for lo, la in zip(lx, ly)]
            text = [f"{w:.2f} m/s" if not np.isnan(w) else "n/a" for w in ws_vals]
        fig.add_trace(go.Scatter(
            x=lx, y=ly, mode="markers+text" if text else "markers",
            marker=dict(symbol="circle-open", color="white", size=7),
            text=text, textposition="top center", textfont=dict(color="yellow", size=9),
            name="Layout", showlegend=False, hoverinfo="skip"))

    if measurement_points:
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        mx = [lon for lat, lon in measurement_points]
        my = [lat for lat, lon in measurement_points]
        mtxt = [labels[i] for i in range(len(measurement_points))]
        fig.add_trace(go.Scatter(x=mx, y=my, mode="markers+text",
                                  marker=dict(color="red", size=12),
                                  text=mtxt, textposition="top center",
                                  textfont=dict(color="red", size=12, weight="bold"),
                                  name="Measurement points", showlegend=False))

    if best_points:
        bx2 = [lon for lat, lon, info in best_points]
        by2 = [lat for lat, lon, info in best_points]
        btxt = [f"P{info['cluster']}" for lat, lon, info in best_points]
        hover = [f"P{info['cluster']}<br>WS={info['ws']:.2f} m/s<br>"
                 f"mean dev={info['mean_dev']:.1f}%<br>max dev={info['max_dev']:.1f}%"
                 for lat, lon, info in best_points]
        fig.add_trace(go.Scatter(x=bx2, y=by2, mode="markers+text",
                                  marker=dict(symbol="star", color="orange", size=18,
                                              line=dict(color="black", width=1)),
                                  text=btxt, textposition="bottom center",
                                  hovertext=hover, hoverinfo="text",
                                  name="Best points", showlegend=False))

    if click_grid is not None:
        glon, glat = click_grid
        fig.add_trace(go.Scattergl(x=glon, y=glat, mode="markers",
                                    marker=dict(size=16, color="rgba(0,0,0,0)"),
                                    name="_clickgrid", showlegend=False))

    fig.update_layout(
        height=650, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Longitude", yaxis_title="Latitude",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        dragmode="pan",
    )

    if boundary_gdf is not None:
        bx0, by0, bx1, by1 = boundary_gdf.total_bounds
        pad_x = (bx1 - bx0) * 0.20 or 0.01
        pad_y = (by1 - by0) * 0.20 or 0.01
        fig.update_xaxes(range=[bx0 - pad_x, bx1 + pad_x])
        fig.update_yaxes(range=[by0 - pad_y, by1 + pad_y])
        # uirevision keyed on the boundary itself: as long as the boundary hasn't
        # changed, Plotly preserves whatever pan/zoom the user has set instead of
        # resetting to the default range above on every rerun (e.g. clicking "Fix
        # points" or "Locate best points" would otherwise zoom back out every time).
        fig.update_layout(uirevision=boundary_gdf.geometry.iloc[0].wkt)

    return fig, (len(fig.data) - 1 if click_grid is not None else None)


@st.cache_data(show_spinner="Clustering turbines and searching for the best measurement points...")
def run_best_points_search(data, meta, boundary_wkt, bounds, layout_coords, n_clusters,
                            grid_spacing_m=100):
    """
    Ports the original K-Means + grid-search logic: partitions turbines into
    n_clusters groups, then for each cluster grid-searches inside the boundary
    for the point whose modelled wind speed best represents that cluster's
    turbines (weighted 70% wind-speed match / 30% proximity), same as before.
    """
    from sklearn.cluster import KMeans
    from shapely import wkt as shapely_wkt

    poly_shape = shapely_wkt.loads(boundary_wkt)
    poly_path = MplPath(np.array(poly_shape.exterior.coords))

    coords = np.array(layout_coords)  # (lat, lon) pairs
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(coords)
    labels = kmeans.labels_

    dx = grid_spacing_m / 111000
    dy = grid_spacing_m / 111000

    best_points = []
    for cluster_id in range(n_clusters):
        cluster_pts = coords[labels == cluster_id]
        layout_ws, valid_pts = [], []
        for lat, lon in cluster_pts:
            ws = sample_raster(data, meta, lat, lon)
            if not np.isnan(ws):
                layout_ws.append(ws)
                valid_pts.append((lat, lon))
        if not layout_ws:
            continue
        layout_ws = np.array(layout_ws)

        best_score, best_candidate = np.inf, None
        for lat in np.arange(bounds[1], bounds[3], dy):
            for lon in np.arange(bounds[0], bounds[2], dx):
                if not poly_path.contains_point((lon, lat)):
                    continue
                ws_ref = sample_raster(data, meta, lat, lon)
                if np.isnan(ws_ref):
                    continue
                pct_err = np.mean(np.abs((layout_ws - ws_ref) / layout_ws))
                dist = np.mean([np.sqrt((lat - la) ** 2 + (lon - lo) ** 2)
                                for la, lo in valid_pts])
                score = 0.7 * pct_err + 0.3 * dist
                if score < best_score:
                    best_score = score
                    best_candidate = (lat, lon, ws_ref, pct_err)

        if best_candidate:
            lat, lon, ws_ref, pct_err = best_candidate
            max_pct = np.max(np.abs((layout_ws - ws_ref) / layout_ws))
            best_points.append((lat, lon, {
                "cluster": cluster_id + 1, "ws": ws_ref,
                "mean_dev": pct_err * 100, "max_dev": max_pct * 100,
            }))
    return best_points


def render_comparison_fig(combo_labels, points, wind_maps):
    """Replaces the original tool's blocking plt.show() popup with an inline
    figure comparing wind speed at the chosen points across every loaded map."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for (source, height), (data, meta) in wind_maps.items():
        values = []
        for lbl in combo_labels:
            lat, lon = points[lbl]
            values.append(sample_raster(data, meta, lat, lon))
        ax.plot(combo_labels, values, marker="o", label=f"{source} {height} m")
    ax.set_title(f"Wind Speed Comparison - {', '.join(combo_labels)}")
    ax.set_ylabel("Wind Speed (m/s)")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def render_full_campaign_map_fig(active_map, boundary_gdf, layout_gdf, measurement_points,
                                  best_points):
    """A single static overview combining every layer: background wind map with its
    colour scale, site boundary, layout with per-turbine wind speed labels, the
    candidate measurement points, and the K-Means-recommended best points -
    intended as the one "everything" figure for the download package."""
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as pe

    outline = [pe.withStroke(linewidth=2.5, foreground="black")]
    outline_w = [pe.withStroke(linewidth=2.5, foreground="white")]

    fig, ax = plt.subplots(figsize=(11, 9))

    if active_map is not None:
        data, meta = active_map
        im = ax.imshow(data, extent=meta["extent"], origin="upper", cmap="viridis",
                        aspect="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label("Wind Speed (m/s)")

    if boundary_gdf is not None:
        boundary_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=2)

    handles = []
    if layout_gdf is not None:
        lx = [g.x for g in layout_gdf.geometry]
        ly = [g.y for g in layout_gdf.geometry]
        ax.scatter(lx, ly, facecolor="none", edgecolor="white", marker="o", s=45,
                   linewidth=1.3, zorder=4, path_effects=outline)
        if active_map is not None:
            data, meta = active_map
            for x, y in zip(lx, ly):
                ws = sample_raster(data, meta, y, x)
                if not np.isnan(ws):
                    txt = ax.annotate(f"{ws:.2f}", (x, y), color="yellow", fontsize=7,
                                       xytext=(3, 3), textcoords="offset points", zorder=5)
                    txt.set_path_effects(outline)
        handles.append(Line2D([0], [0], marker="o", color="none", markeredgecolor="white",
                               label="Layout turbine", markersize=8))

    if measurement_points:
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i, (lat, lon) in enumerate(measurement_points):
            ax.plot(lon, lat, "o", color="red", markersize=9, zorder=6,
                    path_effects=outline_w)
            txt = ax.annotate(labels[i], (lon, lat), color="white", fontsize=10,
                               fontweight="bold", xytext=(4, 4), textcoords="offset points",
                               zorder=7)
            txt.set_path_effects(outline)
        handles.append(Line2D([0], [0], marker="o", color="red", linestyle="",
                               label="Measurement point", markersize=8))

    if best_points:
        for lat, lon, info in best_points:
            ax.plot(lon, lat, "*", color="orange", markersize=18,
                    markeredgecolor="black", zorder=6, path_effects=outline_w)
            txt = ax.annotate(f"P{info['cluster']}", (lon, lat), color="black", fontsize=10,
                               fontweight="bold", xytext=(6, -10), textcoords="offset points",
                               zorder=7)
            txt.set_path_effects(outline_w)
        handles.append(Line2D([0], [0], marker="*", color="orange", markeredgecolor="black",
                               linestyle="", label="Best measurement point", markersize=13))

    if handles:
        ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=9)

    if boundary_gdf is not None:
        bx0, by0, bx1, by1 = boundary_gdf.total_bounds
        pad_x = (bx1 - bx0) * 0.20 or 0.01
        pad_y = (by1 - by0) * 0.20 or 0.01
        ax.set_xlim(bx0 - pad_x, bx1 + pad_x)
        ax.set_ylim(by0 - pad_y, by1 + pad_y)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Measurement Campaign Planning - Overview")
    fig.tight_layout()
    return fig


def build_campaign_excel(boundary_gdf, layout_gdf, wind_maps, measurement_points, best_points):
    """One workbook: site boundary vertices, layout, layout wind speed (one column
    per loaded wind map), candidate measurement points, and best measurement points."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        if boundary_gdf is not None:
            bx, by = boundary_gdf.geometry.iloc[0].exterior.coords.xy
            pd.DataFrame({"Latitude": list(by), "Longitude": list(bx)}).to_excel(
                writer, sheet_name="Site Boundary", index=False)

        if layout_gdf is not None:
            lat = [g.y for g in layout_gdf.geometry]
            lon = [g.x for g in layout_gdf.geometry]
            layout_df = pd.DataFrame({"Turbine": [f"T{i+1}" for i in range(len(lat))],
                                       "Latitude": lat, "Longitude": lon})
            layout_df.to_excel(writer, sheet_name="Layout", index=False)

            ws_df = layout_df.copy()
            for (source, height), (data, meta) in wind_maps.items():
                col = f"{source} @ {height}m (m/s)"
                ws_df[col] = [sample_raster(data, meta, la, lo) for la, lo in zip(lat, lon)]
            ws_df.to_excel(writer, sheet_name="Layout Wind Speeds", index=False)

        if measurement_points:
            labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            mp_df = pd.DataFrame([
                {"Label": labels[i], "Latitude": lat, "Longitude": lon}
                for i, (lat, lon) in enumerate(measurement_points)
            ])
            mp_df.to_excel(writer, sheet_name="Measurement Points", index=False)

        if best_points:
            bp_df = pd.DataFrame([
                {"Point": f"P{info['cluster']}", "Latitude": lat, "Longitude": lon,
                 "Wind Speed (m/s)": info["ws"], "Mean Deviation (%)": info["mean_dev"],
                 "Max Deviation (%)": info["max_dev"]}
                for lat, lon, info in best_points
            ])
            bp_df.to_excel(writer, sheet_name="Best Measurement Points", index=False)

    buf.seek(0)
    return buf.read()


# ==============================================================================
# PRELIMINARY WIND RESOURCE ASSESSMENT - HELPERS
# (reuses read_ascii_grid, sample_raster, read_geo_file_from_bytes, read_layout_file,
# boundary_path, show_fig, fig_to_png_bytes from the Measurement Campaign Planning
# section above)
# ==============================================================================

def detect_asc_height(filename):
    """Height in meters from a .asc filename. Primary pattern matches the
    '.M.<N>m' convention (e.g. 'site.M.100m.asc'); falls back to any trailing
    number-then-m pattern if that's not found."""
    m = re.search(r"\.M\.(\d+)m", filename, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*m(?:\.asc)?$", filename, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


@st.cache_data(show_spinner=False)
def cached_read_ascii_grid(file_bytes):
    """A folder can easily hold a dozen+ .asc files across both sources -
    without caching, every one would be fully re-parsed on every rerun
    (e.g. every keystroke elsewhere on the page)."""
    return read_ascii_grid(file_bytes)


def compute_shear_at_point(wind_maps_source, lat, lon):
    """wind_maps_source: {height: (data, meta)} for ONE source (ERA5 or CFSR).
    Fits ln(z) vs ln(WS) across every available valid height AT THIS EXACT
    LOCATION - one regression per position, rather than pooling every
    consecutive-height-pair alpha from every position into one site-wide
    number. The same code path naturally handles 2 heights (a direct slope)
    or 3+ (a proper least-squares fit). Returns (alpha, r2, n_heights_used)."""
    heights = sorted(wind_maps_source.keys())
    zs, ws = [], []
    for h in heights:
        data, meta = wind_maps_source[h]
        val = sample_raster(data, meta, lat, lon)
        if not np.isnan(val) and val > 0:
            zs.append(h)
            ws.append(val)
    if len(zs) < 2:
        return np.nan, np.nan, len(zs)
    slope, intercept, r, p, se = stats.linregress(np.log(zs), np.log(ws))
    return slope, r ** 2, len(zs)


def sample_at_height(wind_maps_source, lat, lon, target_height, alpha):
    """Extrapolates to target_height from the nearest available height at or
    below it (or the minimum available height if none are below), using the
    given location-specific shear exponent. Used consistently everywhere a
    height needs picking - unlike the original tool, which used this "nearest
    available" logic in one place but silently hardcoded height 100 elsewhere,
    breaking for any map set that didn't happen to include a 100m file."""
    heights = sorted(wind_maps_source.keys())
    if not heights or np.isnan(alpha):
        return np.nan
    lower = [h for h in heights if h <= target_height]
    base_h = max(lower) if lower else min(heights)
    data, meta = wind_maps_source[base_h]
    u_base = sample_raster(data, meta, lat, lon)
    if np.isnan(u_base) or u_base <= 0:
        return np.nan
    return u_base * (target_height / base_h) ** alpha


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km - used for calibration distance-weighting
    instead of the original's flat degree-difference approximation, which
    distorts longitude distances away from the equator."""
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2r - lat1r, lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


@st.cache_data(show_spinner="Computing shear at each turbine position...")
def compute_layout_shear(era5_maps, cfsr_maps, layout_coords):
    """Per-position ERA5 alpha, CFSR alpha, and their average, for every
    (lat, lon) in layout_coords."""
    results = []
    for lat, lon in layout_coords:
        a_e, r2_e, n_e = compute_shear_at_point(era5_maps, lat, lon) if era5_maps else (np.nan, np.nan, 0)
        a_c, r2_c, n_c = compute_shear_at_point(cfsr_maps, lat, lon) if cfsr_maps else (np.nan, np.nan, 0)
        valid = [a for a in (a_e, a_c) if not np.isnan(a)]
        avg = float(np.mean(valid)) if valid else np.nan
        results.append({"era5_alpha": a_e, "era5_r2": r2_e, "era5_n": n_e,
                         "cfsr_alpha": a_c, "cfsr_r2": r2_c, "cfsr_n": n_c,
                         "avg_alpha": avg})
    return results


@st.cache_data(show_spinner="Extrapolating to hub height and applying weights...")
def compute_hub_height_results(era5_maps, cfsr_maps, layout_coords, shear_results,
                                hub_height, w_era5):
    """Per-position ERA5/CFSR wind speed at hub height, plus the weighted blend.
    The weighted value requires BOTH sources to be valid at that position -
    that's the whole point of this tool - but the individual source values are
    still reported even where only one is valid, for diagnostic purposes."""
    w_cfsr = 1 - w_era5
    results = []
    for (lat, lon), sh in zip(layout_coords, shear_results):
        u_era5 = sample_at_height(era5_maps, lat, lon, hub_height, sh["era5_alpha"])
        u_cfsr = sample_at_height(cfsr_maps, lat, lon, hub_height, sh["cfsr_alpha"])
        weighted = (u_era5 * w_era5 + u_cfsr * w_cfsr
                    if not (np.isnan(u_era5) or np.isnan(u_cfsr)) else np.nan)
        results.append({"era5_hh_ws": u_era5, "cfsr_hh_ws": u_cfsr, "weighted_hh_ws": weighted})
    return results


@st.cache_data(show_spinner="Computing calibration factors...")
def compute_calibration_factors(era5_maps, cfsr_maps, cal_points, w_era5):
    """For each calibration point, computes the model wind speed AT THAT
    POINT'S OWN LOCATION AND HEIGHT directly - using local shear fitted at
    that exact (lat, lon), same as for turbines - then CF = measured / model.
    This is more direct than the original tool, which extrapolated to hub
    height first and then back down to the calibration height using a single
    site-wide shear number; going straight from the raster heights to the
    target height with the local shear avoids that indirection entirely."""
    w_cfsr = 1 - w_era5
    factors = []
    for pt in cal_points:
        lat, lon, h = pt["lat"], pt["lon"], pt["h"]
        a_e, _, _ = compute_shear_at_point(era5_maps, lat, lon) if era5_maps else (np.nan, np.nan, 0)
        a_c, _, _ = compute_shear_at_point(cfsr_maps, lat, lon) if cfsr_maps else (np.nan, np.nan, 0)
        u_era5 = sample_at_height(era5_maps, lat, lon, h, a_e)
        u_cfsr = sample_at_height(cfsr_maps, lat, lon, h, a_c)
        if np.isnan(u_era5) or np.isnan(u_cfsr):
            continue
        u_model = u_era5 * w_era5 + u_cfsr * w_cfsr
        if u_model <= 0:
            continue
        factors.append({"name": pt["name"], "lat": lat, "lon": lon, "h": h,
                         "meas_ws": pt["ws"], "model_ws": u_model, "cf": pt["ws"] / u_model,
                         "months": pt.get("months", 0), "manual_weight": pt.get("manual_weight"),
                         "r_squared": pt.get("r_squared")})
    return factors


def semivariogram_exponential(h, range_km, nugget=0.0, sill=1.0):
    """Exponential semivariogram model gamma(h). Weight solutions from kriging
    depend only on the SHAPE of this curve, not its absolute scale, so sill=1
    is fine for computing calibration weights."""
    h = np.asarray(h, dtype=float)
    return np.where(h <= 1e-9, 0.0, nugget + (sill - nugget) * (1 - np.exp(-h / range_km)))


def ordinary_kriging_weights(cal_lats, cal_lons, target_lat, target_lon, range_km,
                              nugget=0.05, point_nuggets=None):
    """Solves the ordinary kriging system for a single target location. Unlike
    plain inverse-distance/exponential weighting, kriging accounts for
    REDUNDANCY between calibration points that sit close to each other - two
    clustered points don't each get full independent weight the way they
    would under IDW, since they carry overlapping information. This is the
    key reason kriging outperforms IDW-style methods for exactly this kind of
    sparse point-to-surface interpolation problem (see literature note in the
    Calibration section).

    point_nuggets (optional): a per-point EXTRA nugget added only to that
    point's own diagonal entry, on top of the shared `nugget` used for the
    ordinary between-point semivariogram shape. This is how a calibration
    point's own data-quality problem (e.g. a long-term estimate built on a
    short measurement record) gets folded in: standard ordinary kriging is an
    exact interpolator (gamma(0)=0 for every point, so a point's own value is
    trusted completely), but if one point's own long-term estimate is itself
    uncertain, treating it as exact is exactly what should NOT happen. Adding
    that point's own error variance to its diagonal entry - "kriging with
    measurement error" / "filtered kriging" in the geostatistics literature
    (Christensen, 2011; Cressie, 1993/2015) - relaxes exactness in proportion
    to how much that specific point should be trusted, while leaving every
    other point's exactness and the overall spatial correlation structure
    alone."""
    n = len(cal_lats)
    if n == 1:
        return np.array([1.0])

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(cal_lats[i], cal_lons[i], cal_lats[j], cal_lons[j])
            D[i, j] = D[j, i] = d
    Gamma = semivariogram_exponential(D, range_km, nugget=nugget)
    if point_nuggets is not None:
        np.fill_diagonal(Gamma, Gamma.diagonal() + np.asarray(point_nuggets, dtype=float))

    A = np.ones((n + 1, n + 1))
    A[:n, :n] = Gamma
    A[n, n] = 0.0

    d0 = np.array([haversine_km(target_lat, target_lon, cal_lats[i], cal_lons[i])
                    for i in range(n)])
    gamma0 = semivariogram_exponential(d0, range_km, nugget=nugget)
    b = np.append(gamma0, 1.0)

    try:
        sol = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
    return sol[:n]


def dispersion_uncertainty_pct(months):
    """Estimates the MCP 'dispersion-based uncertainty' (%) - the relative
    standard deviation of the long-term-correction error, normalised by mean
    wind speed - as a function of on-site measurement duration, anchored to
    real measured values rather than an assumed curve shape.

    Anchors: Abascal Mendez et al. (2026, Inventions journal), analysing 30
    real meteorological masts worldwide with up to 27 months of concurrent
    data across three MCP methods (TLS/LR/GB), report this exact metric
    (their Eq. 13-14) for their Linear Regression model, All-Type-Terrain
    average: approximately 2.1% at 3 months, decreasing to approximately
    0.4% at 12 months, in a relationship they find to be approximately
    LINEAR across that range (their Section 3.3.1) - which is what's
    interpolated here between those two points. Below 3 months this
    extrapolates the same line rather than using a directly-measured value,
    since 3 months was the shortest campaign length in their study. Beyond
    12 months the uncertainty is held at the 12-month (best observed) value,
    matching their own finding that reduction levels off there.

    Deliberately NOT stratified by terrain complexity here, even though the
    source paper reports terrain-specific values (complex terrain runs
    higher, flat terrain lower) - only the All-Type-Terrain aggregate could
    be verified with confidence from the available source material, so
    terrain-specific breakdown is left for a future update rather than
    risk citing a number that wasn't verified as precisely."""
    if months is None or months != months or months <= 0:  # months != months catches NaN
        return 0.4  # unflagged -> treated as the best (12-month) case, same as before
    m = max(float(months), 1.0)
    slope_pct_per_month = (0.4 - 2.1) / (12 - 3)
    u = 2.1 + slope_pct_per_month * (m - 3)
    return float(np.clip(u, 0.4, None))


def confidence_weight_from_duration_and_r2(months, r_squared=None):
    """Converts measurement duration (and optionally the MCP regression's
    R2, if the engineer has it) into a 0-1 confidence weight, via inverse-
    variance weighting - the statistically standard way to combine
    estimates of differing precision (the same principle behind meta-
    analysis pooling and Gauss-Markov/BLUE estimation), rather than an
    assumed 0-1 curve shape.

    weight = (U_best / U)^2, where U is this point's estimated dispersion-
    uncertainty percentage (see dispersion_uncertainty_pct) and U_best=0.4%
    is the best value the cited study observed (at 12+ months) - so an
    unflagged or 12+-month point still gets exactly 1.0, identical to
    before. Squaring (rather than a linear or sqrt scaling of U) is not a
    stylistic choice - it's what inverse-VARIANCE weighting requires, since
    U is a standard deviation, not a variance. This makes the resulting
    penalty for a short record noticeably harsher than either of the two
    ad hoc curves used previously in this tool.

    R2, if supplied: the standard regression-theory fact that unexplained
    variance = (1 - R2) of total variance is used to scale U by
    sqrt((1-R2)/(1-R2_REF)). R2_REF=0.95 is a reasonable reference point
    chosen for this tool (roughly the middle of the 0.80-1.00 range the
    cited study examined) - it is NOT a value taken from the paper, and
    should be read as a considered approximation, not a precise citation.

    Weight is clipped to [0, 1] - even an excellent R2 can't push a point
    above the 12-month reference case, keeping 1.0 as a consistent ceiling
    for every calibration method that consumes this weight."""
    u = dispersion_uncertainty_pct(months)
    if r_squared is not None and r_squared == r_squared and 0 < r_squared < 1:  # excludes NaN
        r2_ref = 0.95
        u *= np.sqrt((1 - r_squared) / (1 - r2_ref))
    u_best = 0.4
    weight = (u_best / u) ** 2 if u > 0 else 1.0
    return float(np.clip(weight, 0.0, 1.0))


def resolve_confidence_weight(point):
    """Returns the confidence weight to actually use for a calibration point:
    a manual override if the engineer supplied one (full control, no formula
    involved), otherwise the measurement-duration-(and-optionally-R2)-
    derived value. Every consumer of confidence weights (Site Average CF,
    Distance Weighted CF, Kriging's per-point nugget) should call this
    rather than confidence_weight_from_duration_and_r2 directly, so a
    manual override is always respected."""
    mw = point.get("manual_weight")
    if mw is not None and mw == mw:  # mw == mw excludes NaN
        return float(mw)
    return confidence_weight_from_duration_and_r2(point.get("months"), point.get("r_squared"))


def uncertainty_nugget(weight):
    """Maps a confidence weight (1 = fully trusted) onto the EXTRA per-point
    kriging diagonal term described in ordinary_kriging_weights. Ordinary
    kriging's diagonal is always exactly 0 by definition (gamma(0)=0
    regardless of the shared `nugget` parameter, which only shapes the curve
    for points at nonzero distance from each other) - so 0 is the correct
    no-op baseline here, not the shared nugget value. At weight=1 this
    returns exactly 0 (no change from today's behaviour for an unflagged
    point); it rises toward 1.0 (the semivariogram's sill) as confidence
    drops toward 0, so a flagged low-confidence point contributes almost no
    spatially-structured information to the solve."""
    return 1.0 - weight


def clip_and_renormalize(weights):
    """Clips any negative kriging weight to zero and renormalizes the rest to
    sum to 1. Ordinary kriging is mathematically allowed to produce negative
    weights (it isn't a constrained weighted average), and that's often fine
    with enough calibration points - but with very few points, especially
    when one carries a much larger nugget than the others (e.g. a flagged
    short record), the system can become poorly constrained: solving the
    same kriging system with one point held out (as leave-one-out
    cross-validation does) can leave only 2 points and a skewed nugget
    between them, which is close to the most degenerate case kriging can
    face. Testing this against randomized small calibration sets shaped like
    that found negative weights in effectively all of them, with cross-
    validated error roughly 5-8x higher than the clipped version - so this
    is the recommended default, with clip_negative_weights left as an
    explicit toggle for anyone who wants pure, unclipped kriging instead."""
    w = np.clip(np.asarray(weights, dtype=float), 0, None)
    total = w.sum()
    return w / total if total > 0 else np.ones_like(w) / len(w)


def apply_calibration(weighted_hh_ws_list, layout_coords, factors, method, decay_km=250.0,
                       clip_negative_weights=True):
    """Site Average CF: one calibration factor applied uniformly everywhere.
    Distance Weighted CF: each turbine's factor is an exponential-decay-weighted
    blend of every calibration point's factor, using true great-circle distance.
    Kriging (Ordinary): a geostatistical blend using the same distance decay as
    a semivariogram range, but correctly down-weighting clustered/redundant
    calibration points rather than treating each as independent evidence.

    All three additionally weight by each point's confidence (see
    confidence_weight_from_months) - a point with a flagged short record
    contributes less, on top of (not instead of) its distance/redundancy
    standing. Points with no record length entered get full weight, so this
    changes nothing unless a point has actually been flagged.

    clip_negative_weights (Kriging only, default True): see
    clip_and_renormalize - strongly recommended with few calibration points.

    Returns (calibrated_ws_list, influence_list) or (None, None) if no factors."""
    n_cal = len(factors)
    if n_cal == 0:
        return None, None

    cf_vals = np.array([f["cf"] for f in factors])
    conf_weights = np.array([resolve_confidence_weight(f) for f in factors])

    if method == "Site Average CF":
        avg_cf = np.average(cf_vals, weights=conf_weights)
        calibrated = [w * avg_cf if not np.isnan(w) else np.nan for w in weighted_hh_ws_list]
        norm_conf = conf_weights / conf_weights.sum()
        return calibrated, norm_conf.tolist()

    calibrated = []
    weight_tracker = np.zeros(n_cal)
    point_nuggets = uncertainty_nugget(conf_weights)
    for (lat, lon), w_ws in zip(layout_coords, weighted_hh_ws_list):
        if np.isnan(w_ws):
            calibrated.append(np.nan)
            continue
        if method == "Kriging (Ordinary)":
            cal_lats = [f["lat"] for f in factors]
            cal_lons = [f["lon"] for f in factors]
            w_norm = ordinary_kriging_weights(cal_lats, cal_lons, lat, lon, decay_km,
                                               point_nuggets=point_nuggets)
            if clip_negative_weights:
                w_norm = clip_and_renormalize(w_norm)
        else:  # Distance Weighted CF
            dists = np.array([haversine_km(lat, lon, f["lat"], f["lon"]) for f in factors])
            w = np.exp(-dists / decay_km) * conf_weights
            w_norm = w / np.sum(w)
        weight_tracker += w_norm
        cf_local = np.sum(w_norm * cf_vals)
        calibrated.append(w_ws * cf_local)
    influence = (weight_tracker / len(layout_coords)).tolist()
    return calibrated, influence


@st.cache_data(show_spinner="Cross-validating calibration methods...")
def loocv_calibration_errors(factors, methods, decay_km=250.0, clip_negative_weights=True):
    """Leave-one-out cross-validation: for each calibration point, predict it
    using ONLY the other points, and compare to its actual measurement. This
    is the standard way the wind resource literature evaluates which spatial
    calibration approach actually performs best for a given set of points -
    rather than assuming one method is best, check empirically. Needs >=2
    points; more points make the comparison more informative."""
    results = {}
    for method in methods:
        errors = []
        for i, held_out in enumerate(factors):
            remaining = factors[:i] + factors[i + 1:]
            if not remaining:
                continue
            cf_vals = np.array([f["cf"] for f in remaining])
            conf_weights = np.array([resolve_confidence_weight(f) for f in remaining])
            if method == "Site Average CF":
                pred_cf = np.average(cf_vals, weights=conf_weights)
            elif method == "Kriging (Ordinary)":
                lats = [f["lat"] for f in remaining]
                lons = [f["lon"] for f in remaining]
                point_nuggets = uncertainty_nugget(conf_weights)
                w = ordinary_kriging_weights(lats, lons, held_out["lat"], held_out["lon"],
                                              decay_km, point_nuggets=point_nuggets)
                if clip_negative_weights:
                    w = clip_and_renormalize(w)
                pred_cf = np.sum(w * cf_vals)
            else:
                dists = np.array([haversine_km(held_out["lat"], held_out["lon"], f["lat"], f["lon"])
                                   for f in remaining])
                w = np.exp(-dists / decay_km) * conf_weights
                pred_cf = np.sum((w / np.sum(w)) * cf_vals)
            pred_ws = held_out["model_ws"] * pred_cf
            errors.append(pred_ws - held_out["meas_ws"])
        if errors:
            errors = np.array(errors)
            results[method] = {"rmse": float(np.sqrt(np.mean(errors ** 2))),
                                "mae": float(np.mean(np.abs(errors))), "n": len(errors)}
        else:
            results[method] = {"rmse": np.nan, "mae": np.nan, "n": 0}
    return results


def build_results_map_fig(boundary_gdf, layout_gdf, values, value_label, cal_points=None):
    """Layout coloured/labelled by a precomputed per-position value (hub-height
    or calibrated wind speed) rather than a raster - same padding/uirevision
    treatment as the Measurement Campaign Planning maps, so pan/zoom persists
    across reruns and the default view is the boundary, not whatever the
    layout's bounding box happens to be."""
    import plotly.graph_objects as go

    fig = go.Figure()

    if boundary_gdf is not None:
        bx, by = boundary_gdf.geometry.iloc[0].exterior.coords.xy
        fig.add_trace(go.Scatter(x=list(bx), y=list(by), mode="lines",
                                  line=dict(color="white", width=2),
                                  showlegend=False, hoverinfo="skip"))

    if layout_gdf is not None:
        lx = [g.x for g in layout_gdf.geometry]
        ly = [g.y for g in layout_gdf.geometry]
        text = [f"{v:.2f}" if not np.isnan(v) else "n/a" for v in values]
        fig.add_trace(go.Scatter(
            x=lx, y=ly, mode="markers+text",
            marker=dict(size=14, color=values, colorscale="Viridis", showscale=True,
                        colorbar=dict(title=value_label),
                        line=dict(color="white", width=1)),
            text=text, textposition="top center", textfont=dict(color="white", size=9),
            name="Layout", showlegend=False,
            hovertemplate="%{text} m/s<extra></extra>"))

    if cal_points:
        cx = [p["lon"] for p in cal_points]
        cy = [p["lat"] for p in cal_points]
        ctxt = [p["name"] for p in cal_points]
        chover = [f"{p['name']}<br>Measured: {p['ws']:.2f} m/s @ {p['h']:.0f} m"
                  for p in cal_points]
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="markers+text",
                                  marker=dict(symbol="star", size=16, color="red",
                                              line=dict(color="white", width=1)),
                                  text=ctxt, textposition="bottom center",
                                  textfont=dict(color="white", size=10),
                                  hovertext=chover, hoverinfo="text",
                                  name="Calibration point", showlegend=False))

    fig.update_layout(height=650, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis_title="Longitude", yaxis_title="Latitude",
                       yaxis=dict(scaleanchor="x", scaleratio=1), dragmode="pan")

    if boundary_gdf is not None:
        bx0, by0, bx1, by1 = boundary_gdf.total_bounds
        pad_x = (bx1 - bx0) * 0.20 or 0.01
        pad_y = (by1 - by0) * 0.20 or 0.01
        fig.update_xaxes(range=[bx0 - pad_x, bx1 + pad_x])
        fig.update_yaxes(range=[by0 - pad_y, by1 + pad_y])
        fig.update_layout(uirevision=boundary_gdf.geometry.iloc[0].wkt + value_label)

    return fig


def render_wra_static_map(boundary_gdf, layout_gdf, values, value_label, title,
                           cal_points=None):
    """Static matplotlib equivalent of build_results_map_fig, for the download
    package (avoids a kaleido/headless-Chrome dependency for exporting the
    interactive Plotly figure)."""
    import matplotlib.patheffects as pe
    outline = [pe.withStroke(linewidth=2.5, foreground="white")]

    fig, ax = plt.subplots(figsize=(10, 8))

    if boundary_gdf is not None:
        boundary_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=2)

    if layout_gdf is not None:
        lx = [g.x for g in layout_gdf.geometry]
        ly = [g.y for g in layout_gdf.geometry]
        sc = ax.scatter(lx, ly, c=values, cmap="viridis", s=80, edgecolor="black",
                        linewidth=0.8, zorder=4)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label(value_label)
        for x, y, v in zip(lx, ly, values):
            if not np.isnan(v):
                txt = ax.annotate(f"{v:.2f}", (x, y), fontsize=7, color="black",
                                   xytext=(3, 3), textcoords="offset points", zorder=5)
                txt.set_path_effects(outline)

    if cal_points:
        for p in cal_points:
            ax.plot(p["lon"], p["lat"], "*", color="red", markersize=16,
                    markeredgecolor="black", zorder=6)
            txt = ax.annotate(p["name"], (p["lon"], p["lat"]), fontsize=9,
                               fontweight="bold", color="black", xytext=(5, -10),
                               textcoords="offset points", zorder=7)
            txt.set_path_effects(outline)

    if boundary_gdf is not None:
        bx0, by0, bx1, by1 = boundary_gdf.total_bounds
        pad_x = (bx1 - bx0) * 0.20 or 0.01
        pad_y = (by1 - by0) * 0.20 or 0.01
        ax.set_xlim(bx0 - pad_x, bx1 + pad_x)
        ax.set_ylim(by0 - pad_y, by1 + pad_y)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _group_sources_by_location(sources, tolerance_km=0.5):
    """Groups sources whose locations are within tolerance_km of each other -
    the common case being several modelled sources (e.g. Vortex ERA5 and
    Vortex CFSR) extracted for the exact same site, which would otherwise
    render as overlapping markers with overlapping text labels. 0.5 km is
    tight enough that genuinely distinct nearby sites won't get merged, but
    loose enough to absorb small differences in how a coordinate got typed
    or rounded between exports."""
    groups, used = [], [False] * len(sources)
    for i, s in enumerate(sources):
        if used[i]:
            continue
        group = [s]
        used[i] = True
        for j in range(i + 1, len(sources)):
            if used[j]:
                continue
            if haversine_km(s["lat"], s["lon"], sources[j]["lat"], sources[j]["lon"]) <= tolerance_km:
                group.append(sources[j])
                used[j] = True
        groups.append(group)
    return groups


def build_model_locations_map_fig(sources):
    """Interactive world map of where each configured modelled source is
    actually located, using Plotly's Scattergeo trace - built-in vector
    coastline/country/land outlines bundled with plotly.js itself, so this
    needs no external map tiles, no API key or token, and no new
    dependency (unlike Scattermapbox, which needs a Mapbox token this
    environment doesn't have). resolution=50 is the finest built-in detail
    Plotly's geo maps support (a two-tier "110" or "50" setting - 50 is
    roughly twice as fine, at a 1:50,000,000 scale rather than 1:110M);
    there's no path to sharper than this without switching to real map
    tiles. Sources sharing (near enough) the same location - the common
    case for multiple reanalysis products pulled for one site - are
    grouped into a single marker rather than rendered as overlapping dots
    with overlapping text, since that's both clearer and more honest about
    what's actually being shown."""
    import plotly.graph_objects as go
    groups = _group_sources_by_location(sources)

    xs = [g[0]["lon"] for g in groups]
    ys = [g[0]["lat"] for g in groups]
    labels = []
    hover = []
    for g in groups:
        if len(g) == 1:
            s = g[0]
            labels.append(s["label"])
            hover.append(f"{s['label']}<br>{s['lat']:.5f}, {s['lon']:.5f}<br>"
                          f"Height: {s['height']:.0f} m")
        else:
            labels.append(f"{len(g)} sources (same location)")
            lines = "<br>".join(f"{s['label']} ({s['height']:.0f} m)" for s in g)
            hover.append(f"Same location - {len(g)} sources:<br>{lines}<br>"
                          f"{g[0]['lat']:.5f}, {g[0]['lon']:.5f}")

    fig = go.Figure(go.Scattergeo(
        lon=xs, lat=ys, mode="markers+text",
        marker=dict(symbol="circle", size=12, color="#2b6cb0", line=dict(color="white", width=1)),
        text=labels, textposition="top center", textfont=dict(color="black", size=10),
        hovertext=hover, hoverinfo="text", showlegend=False))
    fig.update_geos(
        resolution=50,
        showcountries=True, countrycolor="rgb(170,170,170)", countrywidth=1,
        showsubunits=True, subunitcolor="rgb(210,210,210)", subunitwidth=0.5,
        showcoastlines=True, coastlinecolor="rgb(110,110,110)", coastlinewidth=1,
        showland=True, landcolor="rgb(245,245,240)",
        showocean=True, oceancolor="rgb(220,235,245)",
        showlakes=True, lakecolor="rgb(220,235,245)",
        showrivers=True, rivercolor="rgb(220,235,245)",
        showframe=False)
    fig.update_layout(height=550, margin=dict(l=10, r=10, t=10, b=10))
    if xs and ys:
        # Auto-zoom only for a reasonably local cluster of sources (the realistic case here -
        # comparing several modelled sources for essentially the same site). For sources spread
        # across continents, a computed "zoomed" range can land far outside the valid lon/lat
        # domain entirely (confirmed by testing: two sources ~100 degrees of longitude apart
        # produced a requested range of roughly -240 to 150) - simplest correct fix is to just
        # leave the natural full-world view in that case rather than try to clamp a meaningless
        # zoom level.
        lon_span, lat_span = max(xs) - min(xs), max(ys) - min(ys)
        if lon_span < 60 and lat_span < 60:
            # Padding computed in true distance (km), not raw degrees - a degree of longitude
            # covers less real distance than a degree of latitude away from the equator, by a
            # factor of cos(latitude). Ignoring that (as an earlier version of this did) can
            # produce a window whose TRUE aspect ratio is badly mismatched to the wide
            # container it renders into - confirmed by calculation: at 55 degrees latitude, a
            # naive +/-2 degree box in both directions comes out at a 0.57:1 (portrait) true
            # aspect ratio against a ~2.2:1 (landscape) target, which is what caused the map to
            # render as a small, letterboxed rectangle inside a mostly-empty chart area. This
            # instead solves for lon/lat padding that hits TARGET_ASPECT once projected.
            TARGET_ASPECT = 2.2  # width:height, matching a typical wide Streamlit chart area
            MIN_PAD_KM = 15.0
            mean_lat = sum(ys) / len(ys)
            lat_cos = max(np.cos(np.radians(mean_lat)), 0.1)  # guards against 0 near the poles
            lon_span_km = lon_span * 111.0 * lat_cos
            lat_span_km = lat_span * 111.0
            half_lat_km = max(lat_span_km / 2, MIN_PAD_KM)
            half_lon_km = max(half_lat_km * TARGET_ASPECT, lon_span_km / 2, MIN_PAD_KM)
            half_lat_km = half_lon_km / TARGET_ASPECT  # keep the box at exactly TARGET_ASPECT
            lat_pad = half_lat_km / 111.0
            lon_pad = half_lon_km / (111.0 * lat_cos)
            mid_x, mid_y = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
            fig.update_geos(
                projection_type="mercator",  # designed for local zoom, unlike natural earth
                lonaxis_range=[max(mid_x - lon_pad, -180), min(mid_x + lon_pad, 180)],
                lataxis_range=[max(mid_y - lat_pad, -90), min(mid_y + lat_pad, 90)])
        else:
            # Whole-world view: natural earth avoids the severe high-latitude area distortion
            # (e.g. Greenland rendering far larger than it really is) that mercator is known for
            # at a global scale - the tradeoff that made mercator the right choice above doesn't
            # apply here, since there's no small local window being rendered.
            fig.update_geos(projection_type="natural earth")
    return fig


def build_wra_excel(boundary_gdf, layout_gdf, hub_height_results, calibrated_ws=None,
                     cal_points=None, cal_factors=None):
    """Site Boundary, Layout, per-position ERA5/CFSR/Weighted (and Calibrated,
    if available) hub-height wind speed, plus calibration diagnostics."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        if boundary_gdf is not None:
            bx, by = boundary_gdf.geometry.iloc[0].exterior.coords.xy
            pd.DataFrame({"Latitude": list(by), "Longitude": list(bx)}).to_excel(
                writer, sheet_name="Site Boundary", index=False)

        if layout_gdf is not None:
            lat = [g.y for g in layout_gdf.geometry]
            lon = [g.x for g in layout_gdf.geometry]
            rows = []
            for i in range(len(lat)):
                row = {"Turbine": f"T{i+1}", "Latitude": lat[i], "Longitude": lon[i],
                       "ERA5 Wind Speed (m/s)": hub_height_results[i]["era5_hh_ws"],
                       "CFSR Wind Speed (m/s)": hub_height_results[i]["cfsr_hh_ws"],
                       "Weighted Wind Speed (m/s)": hub_height_results[i]["weighted_hh_ws"]}
                if calibrated_ws is not None:
                    row["Calibrated Wind Speed (m/s)"] = calibrated_ws[i]
                rows.append(row)
            df_ws = pd.DataFrame(rows)
            df_ws.to_excel(writer, sheet_name="Wind Speed by Position", index=False)

            summary = {
                "Metric": ["Mean ERA5 Wind Speed (m/s)", "Mean CFSR Wind Speed (m/s)",
                           "Mean Weighted Wind Speed (m/s)"],
                "Value": [df_ws["ERA5 Wind Speed (m/s)"].mean(),
                          df_ws["CFSR Wind Speed (m/s)"].mean(),
                          df_ws["Weighted Wind Speed (m/s)"].mean()],
            }
            if calibrated_ws is not None:
                summary["Metric"].append("Mean Calibrated Wind Speed (m/s)")
                summary["Value"].append(df_ws["Calibrated Wind Speed (m/s)"].mean())
            pd.DataFrame(summary).to_excel(writer, sheet_name="Summary", index=False)

        if cal_points:
            pd.DataFrame(cal_points).to_excel(writer, sheet_name="Calibration Points",
                                               index=False)
        if cal_factors:
            pd.DataFrame(cal_factors).to_excel(writer, sheet_name="Calibration Factors",
                                                index=False)

    buf.seek(0)
    return buf.read()


def _guess_col_index(cols, keywords, default=0):
    for kw in keywords:
        for i, c in enumerate(cols):
            if kw.lower() in c.lower():
                return i
    return default


def render_crs_picker(key_prefix):
    """UTM zone/hemisphere (default) or a custom EPSG code (for non-UTM
    projected CRSs, e.g. a national grid like OSGB36) - returns the EPSG
    code to convert FROM, as an int. Shared by every Easting/Northing
    upload path in the tool."""
    crs_mode = st.radio("Coordinate system", ["UTM zone", "Custom EPSG code"],
                         key=f"{key_prefix}_crs_mode", horizontal=True)
    if crs_mode == "UTM zone":
        cc1, cc2 = st.columns(2)
        with cc1:
            zone = st.number_input("UTM zone (1-60)", min_value=1, max_value=60, value=31,
                                    key=f"{key_prefix}_utm_zone")
        with cc2:
            hemisphere = st.selectbox("Hemisphere", ["Northern", "Southern"],
                                       key=f"{key_prefix}_hemisphere")
        epsg = utm_epsg_code(zone, hemisphere)
        st.caption(f"Using EPSG:{epsg} (WGS84 / UTM zone {int(zone)}"
                   f"{'N' if hemisphere == 'Northern' else 'S'}). Getting the hemisphere wrong "
                   f"shifts everything by ~10,000 km, so check the result lands in the right "
                   f"place once loaded, on the map below.")
    else:
        epsg = st.number_input(
            "EPSG code", min_value=1000, max_value=999999, value=27700,
            key=f"{key_prefix}_custom_epsg",
            help="E.g. 27700 for OSGB36 British National Grid, or any other projected CRS your "
                 "Easting/Northing values are given in.")
    return int(epsg)


def render_boundary_uploader(key_prefix, gen):
    """Site Boundary upload: GeoJSON, KML, or a CSV/Excel of Easting/Northing
    boundary vertices. Writes the result straight to
    st.session_state[f'{key_prefix}_boundary'] (matching each mode's
    existing state key, e.g. camp_boundary / wra_boundary) and also returns
    it. Shared between Campaign Planning and Preliminary WRA."""
    state_key = f"{key_prefix}_boundary"
    fmt = st.radio("Boundary file type", ["GeoJSON / KML", "CSV or Excel (Easting/Northing)"],
                    key=f"{key_prefix}_boundary_fmt", horizontal=True)
    if fmt == "GeoJSON / KML":
        f = st.file_uploader("Site boundary file (.geojson or .kml)",
                              type=["geojson", "kml"], key=f"{key_prefix}_boundary_file_{gen}")
        if f is not None and st.session_state[state_key] is None:
            try:
                ext = f.name.lower().rsplit(".", 1)[-1]
                st.session_state[state_key] = read_geo_file_from_bytes(f.getvalue(), f".{ext}")
            except Exception as e:
                st.error(f"Could not read boundary file: {e}")
    else:
        f = st.file_uploader("Boundary vertex file (.csv or .xlsx)", type=["csv", "xlsx"],
                              key=f"{key_prefix}_boundary_en_file_{gen}")
        st.caption("Points are connected in the order they appear in the file, then closed "
                   "into a polygon - list them in boundary-walking order (as most survey/CAD "
                   "exports already do), not e.g. sorted by point ID.")
        if f is not None:
            fname = f.name
            raw = (pd.read_excel(io.BytesIO(f.getvalue())) if fname.lower().endswith(".xlsx")
                   else pd.read_csv(io.BytesIO(f.getvalue())))
            cols = list(raw.columns)
            bc1, bc2 = st.columns(2)
            with bc1:
                e_col = st.selectbox("Easting column", cols,
                                      index=_guess_col_index(cols, ["east"]),
                                      key=f"{key_prefix}_bnd_e_col")
            with bc2:
                n_col = st.selectbox("Northing column", cols,
                                      index=_guess_col_index(cols, ["north"]),
                                      key=f"{key_prefix}_bnd_n_col")
            epsg = render_crs_picker(f"{key_prefix}_bnd")
            if st.button("Load boundary from points", key=f"{key_prefix}_load_boundary_en"):
                try:
                    st.session_state[state_key] = read_boundary_from_en(
                        f.getvalue(), fname, e_col, n_col, epsg)
                except Exception as e:
                    st.error(f"Could not build boundary: {e}")
    if st.session_state[state_key] is not None:
        st.success("Boundary loaded.")
    return st.session_state[state_key]


def render_layout_uploader(key_prefix, gen):
    """Turbine Layout upload: GeoJSON/GPKG (points already present), Excel
    with Latitude/Longitude columns, or a CSV/Excel of Easting/Northing
    points. Writes the result to st.session_state[f'{key_prefix}_layout']
    and also returns it. Shared between Campaign Planning and Preliminary
    WRA."""
    state_key = f"{key_prefix}_layout"
    fmt = st.radio("Layout file type", ["GeoJSON / GPKG / Excel (Lat-Lon)",
                                         "CSV or Excel (Easting/Northing)"],
                    key=f"{key_prefix}_layout_fmt", horizontal=True)
    if fmt == "GeoJSON / GPKG / Excel (Lat-Lon)":
        f = st.file_uploader("Layout file", type=["geojson", "gpkg", "xlsx"],
                              key=f"{key_prefix}_layout_file_{gen}",
                              help="Not raw .shp, since that format is really several files "
                                   "bundled together - export or convert to one of these instead.")
        if f is not None:
            fname = f.name
            if fname.lower().endswith(".xlsx"):
                raw_preview = pd.read_excel(io.BytesIO(f.getvalue()))
                cols = list(raw_preview.columns)
                lc1, lc2, lc3 = st.columns([1, 1, 1])
                with lc1:
                    lat_col = st.selectbox(
                        "Latitude column", cols,
                        index=cols.index("Latitude") if "Latitude" in cols else 0,
                        key=f"{key_prefix}_lat_col")
                with lc2:
                    lon_col = st.selectbox(
                        "Longitude column", cols,
                        index=cols.index("Longitude") if "Longitude" in cols else 0,
                        key=f"{key_prefix}_lon_col")
                with lc3:
                    st.write("")
                    st.write("")
                    if st.button("Load layout", key=f"{key_prefix}_load_layout_xlsx"):
                        try:
                            st.session_state[state_key] = read_layout_file(
                                f.getvalue(), fname, lat_col, lon_col)
                        except Exception as e:
                            st.error(f"Could not read layout: {e}")
            else:
                try:
                    st.session_state[state_key] = read_layout_file(f.getvalue(), fname)
                except Exception as e:
                    st.error(f"Could not read layout: {e}")
    else:
        f = st.file_uploader("Layout points file (.csv or .xlsx)", type=["csv", "xlsx"],
                              key=f"{key_prefix}_layout_en_file_{gen}")
        if f is not None:
            fname = f.name
            raw = (pd.read_excel(io.BytesIO(f.getvalue())) if fname.lower().endswith(".xlsx")
                   else pd.read_csv(io.BytesIO(f.getvalue())))
            cols = list(raw.columns)
            lc1, lc2 = st.columns(2)
            with lc1:
                e_col = st.selectbox("Easting column", cols,
                                      index=_guess_col_index(cols, ["east"]),
                                      key=f"{key_prefix}_lyt_e_col")
            with lc2:
                n_col = st.selectbox("Northing column", cols,
                                      index=_guess_col_index(cols, ["north"]),
                                      key=f"{key_prefix}_lyt_n_col")
            epsg = render_crs_picker(f"{key_prefix}_lyt")
            if st.button("Load layout from points", key=f"{key_prefix}_load_layout_en"):
                try:
                    st.session_state[state_key] = read_layout_from_en(
                        f.getvalue(), fname, e_col, n_col, epsg)
                except Exception as e:
                    st.error(f"Could not build layout: {e}")
    return st.session_state.get(state_key)


if mode == "Long-Term Correction":
    st.title("Wind Resource Analysis Tool")
    st.caption("Upload your measurement data and a modelled wind dataset to get availability, "
               "monthly means, shear, wind roses, correlation and a long-term corrected wind speed.")

    if "ltc_uploader_gen" not in st.session_state:
        st.session_state.ltc_uploader_gen = 0
    _ltc_gen = st.session_state.ltc_uploader_gen

    if st.button("Reset Long-Term Correction"):
        st.session_state.ltc_uploader_gen += 1
        st.session_state.ltc_model_sources = []
        st.session_state.ltc_model_add_gen = st.session_state.get("ltc_model_add_gen", 0) + 1
        if "ltc_meas_df" in st.session_state:
            del st.session_state["ltc_meas_df"]
        if "plots_zip" in st.session_state:
            del st.session_state["plots_zip"]
        st.rerun()

    # ------------------------------------------------------------------ STEP 1 --
    st.header("1. Upload measurement data")
    st.caption("Upload every file you have - lidar or met mast, any number of files. Accepts "
               ".csv, Campbell Scientific TOA5 .dat/.sta, a second .dat/.sta layout seen from "
               "floating LiDAR buoy systems (a header row starting with 'timestamp'), and "
               "NetCDF .nc. Files are auto-detected by format, concatenated, and sorted by "
               "time; overlapping timestamps are de-duplicated.")
    ltc_meas_files = st.file_uploader(
        "Measurement files", type=["csv", "dat", "sta", "nc"], accept_multiple_files=True,
        key=f"ltc_meas_files_{_ltc_gen}")

    if ltc_meas_files:
        ltc_csv_files, ltc_toa5_files, ltc_generic_dat_files, ltc_nc_files, ltc_rejected = (
            [], [], [], [], [])
        for f in ltc_meas_files:
            ext = f.name.lower().rsplit(".", 1)[-1]
            if ext == "csv":
                ltc_csv_files.append(f)
            elif ext in ("dat", "sta"):
                fbytes = f.getvalue()
                if sniff_toa5(fbytes):
                    ltc_toa5_files.append(f)
                elif sniff_generic_dat(fbytes):
                    ltc_generic_dat_files.append(f)
                else:
                    ltc_rejected.append(f.name)
            elif ext == "nc":
                ltc_nc_files.append(f)

        if ltc_rejected:
            st.error(f"{len(ltc_rejected)} file(s) didn't match either supported .dat/.sta "
                     f"layout (Campbell Scientific TOA5, or a header row containing "
                     f"'timestamp') and were skipped: {', '.join(ltc_rejected[:5])}"
                     f"{' ...' if len(ltc_rejected) > 5 else ''}. Show me what one of these "
                     f"actually looks like and I'll add support for that format too.")

        ltc_parsed_frames = []

        # --- TOA5 files: parse directly, timestamp is unambiguous, no mapping needed ---
        if ltc_toa5_files:
            ltc_toa5_skipped_total, ltc_toa5_rows_total = 0, 0
            for f in ltc_toa5_files:
                try:
                    tdf, _units, skipped = parse_toa5(f.getvalue())
                    tdf["Timestamp"] = pd.to_datetime(tdf["TIMESTAMP"], errors="coerce")
                    tdf = (tdf.drop(columns=["TIMESTAMP"]).dropna(subset=["Timestamp"])
                              .set_index("Timestamp"))
                    ltc_parsed_frames.append(tdf)
                    ltc_toa5_skipped_total += skipped
                    ltc_toa5_rows_total += len(tdf)
                except Exception as e:
                    st.error(f"Could not parse {f.name}: {e}")
            st.success(f"Parsed {len(ltc_toa5_files)} TOA5 file(s) - {ltc_toa5_rows_total} rows.")
            if ltc_toa5_skipped_total:
                st.warning(f"{ltc_toa5_skipped_total} malformed row(s) skipped across TOA5 "
                           f"files (wrong number of fields for that file's header).")

        # --- Generic .dat/.sta files (e.g. floating LiDAR buoy exports): one shared
        # dayfirst/invalid-codes setting, applied to every file in this group ---
        if ltc_generic_dat_files:
            st.subheader("1a. Generic .dat/.sta settings (applied to every file in this group)")
            try:
                _preview_df, _detected_dayfirst, _ = parse_generic_dat(
                    ltc_generic_dat_files[0].getvalue())
                st.write(f"Preview of {ltc_generic_dat_files[0].name}:")
                st.dataframe(_preview_df.head(5), width="stretch")
            except Exception as e:
                _detected_dayfirst = True
                st.error(f"Could not preview {ltc_generic_dat_files[0].name}: {e}")
            gc1, gc2 = st.columns(2)
            with gc1:
                ltc_gd_dayfirst = st.checkbox(
                    "Date format is day-first (DD/MM/YYYY)", value=_detected_dayfirst,
                    key="ltc_gd_dayfirst",
                    help="Pre-filled from the units-row date-format descriptor, if this file "
                         "has one (e.g. 'DD-MM-YYYY hh:mm') - check it's right, since it's "
                         "inferred, not guaranteed.")
            with gc2:
                ltc_gd_invalid_text = st.text_input(
                    "Invalid/missing value codes (comma-separated)", value="9999, 9998, 999, -999",
                    key="ltc_gd_invalid_text",
                    help="9998 is included by default since it's a real sentinel value seen in "
                         "this kind of file (e.g. a LiDAR range gate with no valid return) - "
                         "edit if your files use something different.")
            ltc_gd_invalid_codes = parse_invalid_codes(ltc_gd_invalid_text)

            ltc_gd_rows_total, ltc_gd_skipped_total = 0, 0
            for f in ltc_generic_dat_files:
                try:
                    gdf, _df, skipped = parse_generic_dat(f.getvalue())
                    gdf["Timestamp"] = pd.to_datetime(gdf["timestamp"], dayfirst=ltc_gd_dayfirst,
                                                       errors="coerce")
                    gdf = (gdf.drop(columns=["timestamp"]).dropna(subset=["Timestamp"])
                               .set_index("Timestamp"))
                    for c in gdf.columns:
                        gdf[c] = pd.to_numeric(gdf[c], errors="coerce")
                    if ltc_gd_invalid_codes:
                        gdf = gdf.replace(ltc_gd_invalid_codes, np.nan)
                    ltc_parsed_frames.append(gdf)
                    ltc_gd_rows_total += len(gdf)
                    ltc_gd_skipped_total += skipped
                except Exception as e:
                    st.error(f"Could not parse {f.name}: {e}")
            st.success(f"Parsed {len(ltc_generic_dat_files)} file(s) - {ltc_gd_rows_total} rows.")
            if ltc_gd_skipped_total:
                st.warning(f"{ltc_gd_skipped_total} malformed row(s) skipped across these "
                           f"files (wrong number of fields for that file's header).")

        # --- CSV files: one shared column mapping, applied to every CSV uploaded ---
        if ltc_csv_files:
            st.subheader("1b. CSV column mapping (applied to every CSV uploaded)")
            ltc_first_csv_raw = read_raw_csv(ltc_csv_files[0].getvalue())
            st.write(f"Preview of {ltc_csv_files[0].name}:")
            st.dataframe(ltc_first_csv_raw.head(5), width="stretch")
            ltc_csv_cols = list(ltc_first_csv_raw.columns)
            cc1, cc2 = st.columns(2)
            with cc1:
                ltc_ts_col = st.selectbox("Timestamp column", ltc_csv_cols, key="ltc_ts_col")
                ltc_dayfirst = st.checkbox("Date format is day-first (DD/MM/YYYY)", value=True,
                                            key="ltc_dayfirst")
            with cc2:
                ltc_invalid_text = st.text_input(
                    "Invalid/missing value codes (comma-separated)", value="9999, 999, -999",
                    key="ltc_invalid_text")
            ltc_invalid_codes = parse_invalid_codes(ltc_invalid_text)
            timestamp_diagnostics_ui(ltc_first_csv_raw, ltc_ts_col, ltc_dayfirst,
                                      key_prefix="ltc_meas_csv")

            ltc_csv_rows_total = 0
            for f in ltc_csv_files:
                try:
                    raw = read_raw_csv(f.getvalue())
                    cdf = build_clean_df(raw, ltc_ts_col, ltc_dayfirst, tuple(ltc_invalid_codes))
                    ltc_parsed_frames.append(cdf)
                    ltc_csv_rows_total += len(cdf)
                except Exception as e:
                    st.error(f"Could not parse {f.name}: {e}")
            st.success(f"Parsed {len(ltc_csv_files)} CSV file(s) - {ltc_csv_rows_total} rows.")

        # --- NetCDF files: one shared variable mapping, applied to every .nc uploaded ---
        if ltc_nc_files:
            st.subheader("1c. NetCDF variable mapping (applied to every .nc file uploaded)")
            try:
                ltc_var_info = list_netcdf_variables(ltc_nc_files[0].getvalue())
            except Exception as e:
                ltc_var_info = {}
                st.error(f"Could not read {ltc_nc_files[0].name}: {e}")
            if ltc_var_info:
                ltc_var_names = list(ltc_var_info.keys())
                st.write(f"Variables found in {ltc_nc_files[0].name}:")
                st.dataframe(pd.DataFrame([
                    {"Variable": k, "Dims": str(v["dims"]), "Units": v["units"],
                     "Description": v["long_name"]} for k, v in ltc_var_info.items()
                ]), hide_index=True, width="stretch")
                ltc_time_guess = next((n for n in ltc_var_names if "time" in n.lower()),
                                       ltc_var_names[0])
                nc1, nc2 = st.columns(2)
                with nc1:
                    ltc_nc_time_var = st.selectbox(
                        "Time variable", ltc_var_names, index=ltc_var_names.index(ltc_time_guess),
                        key="ltc_nc_time_var")
                with nc2:
                    ltc_nc_value_vars = st.multiselect(
                        "Value variables to import (WS, WD, etc.)",
                        [n for n in ltc_var_names if n != ltc_nc_time_var],
                        key="ltc_nc_value_vars",
                        help="Only 1-D (time-indexed) variables are supported - a single-height "
                             "dataset. A multi-dimensional, height-resolved NetCDF would need a "
                             "different reader.")
                if ltc_nc_value_vars:
                    ltc_nc_rows_total = 0
                    for f in ltc_nc_files:
                        try:
                            ndf = read_netcdf_to_df(f.getvalue(), ltc_nc_time_var,
                                                     tuple(ltc_nc_value_vars))
                            ndf = ndf.set_index("Timestamp")
                            ltc_parsed_frames.append(ndf)
                            ltc_nc_rows_total += len(ndf)
                        except Exception as e:
                            st.error(f"Could not read {f.name}: {e}")
                    st.success(f"Parsed {len(ltc_nc_files)} NetCDF file(s) - "
                               f"{ltc_nc_rows_total} rows.")

        # --- Concatenate everything into one combined, de-duplicated, sorted dataset ---
        if ltc_parsed_frames:
            meas_df = pd.concat(ltc_parsed_frames, axis=0, join="outer", sort=False)
            _n_before_dedup = len(meas_df)
            meas_df = meas_df[~meas_df.index.duplicated(keep="first")].sort_index()
            for c in meas_df.columns:
                meas_df[c] = pd.to_numeric(meas_df[c], errors="coerce")
            if _n_before_dedup != len(meas_df):
                st.caption(f"Removed {_n_before_dedup - len(meas_df)} duplicate timestamp(s) "
                           f"found across files (kept the first occurrence of each).")
            st.session_state.ltc_meas_df = meas_df

    if st.session_state.get("ltc_meas_df") is not None:
        meas_df = st.session_state.ltc_meas_df
        all_cols = list(meas_df.columns)
        n_months = meas_df.index.to_series().dt.to_period("M").nunique()
        st.info(f"Combined dataset: {len(meas_df)} rows, {meas_df.index.min()} to "
                f"{meas_df.index.max()} ({n_months} distinct calendar month"
                f"{'s' if n_months != 1 else ''}).")

        st.subheader("1d. Map heights")
        n_heights = st.number_input("How many heights do you want to map?", min_value=1,
                                     max_value=20, value=3, step=1)

        st.caption("Heights don't need to be entered in order - they're automatically sorted "
                   "low-to-high everywhere in the results.")
        height_map = []
        for i in range(int(n_heights)):
            cols = st.columns([1, 2, 2])
            with cols[0]:
                h = st.number_input(f"Height {i+1} (m)", min_value=1.0, value=float(50 + i * 30),
                                     key=f"h_{i}")
            with cols[1]:
                ws_c = st.selectbox(f"WS column - height {i+1}", all_cols, key=f"ws_{i}")
            with cols[2]:
                wd_c = st.selectbox(f"WD column - height {i+1} (optional)",
                                     ["(none)"] + all_cols, key=f"wd_{i}")
            height_map.append({"height": h, "ws_col": ws_c,
                                "wd_col": None if wd_c == "(none)" else wd_c})

        _sanity_cols = [{"ws_col": hm["ws_col"], "wd_col": hm["wd_col"],
                          "label": f"{hm['height']:.0f} m"} for hm in height_map]
        meas_df, _sanity_report = apply_physical_sanity_filter(meas_df, _sanity_cols)
        if _sanity_report:
            st.warning("Automatically treated as invalid, beyond whatever invalid-value codes "
                       "were specified above (a physically impossible reading - e.g. a wind "
                       f"speed over {MAX_PLAUSIBLE_WS:.0f} m/s - is almost always a sensor "
                       "fault or placeholder value, not a real one, even if it doesn't match "
                       "any code you entered):\n\n" + "\n".join(f"- {line}" for line in _sanity_report))

        res_minutes = detect_resolution_minutes(meas_df.index)
        samples_per_hour = max(1, round(60 / res_minutes))
        samples_per_day = samples_per_hour * 24

        st.info(f"Detected measurement resolution: ~{res_minutes:.1f} min "
                f"({samples_per_hour} samples/hour).")

        st.subheader("1e. Measurement timezone")
        utc_offset = st.number_input(
            "Measurement timezone offset from UTC (hours). E.g. enter 8 if your "
            "measurement timestamps are UTC+8.", value=0.0, step=0.5, key="meas_utc_offset")
        st.caption("Everything is converted to UTC internally (each modelled source uses its "
                   "own offset, set when you configure it in Step 2) - so displayed timestamps "
                   "further down may look shifted from your raw file even if you enter the same "
                   "offset here and for a source. That's expected and doesn't change any "
                   "correlation or long-term result: shifting both sides by an equal amount "
                   "moves the reference frame, not which timestamps line up with which. UTC is "
                   "used as the one common frame because different sources can have different "
                   "offsets from each other.")

        meas_df_utc = meas_df.copy()
        meas_df_utc.index = meas_df_utc.index - pd.Timedelta(hours=utc_offset)

        meas_series_by_height = {}
        meas_series_by_height_native = {}
        for hm in sorted_heights(height_map):
            hourly = resample_to_hourly(meas_df_utc[hm["ws_col"]], samples_per_hour)
            meas_series_by_height[hm["height"]] = (hourly, hm["ws_col"])
            meas_series_by_height_native[hm["height"]] = (meas_df_utc[hm["ws_col"]], hm["ws_col"])

        st.divider()

        # -------------------------------------------------------------- STEP 2 --
        st.header("2. Upload modelled wind data")
        st.caption("Add one or more modelled/reanalysis sources - ERA5, CFSR, MERRA-2, Vortex, "
                   "or similar - to compare correlations across them before choosing which one "
                   "to use for the long-term result.")

        if "ltc_model_sources" not in st.session_state:
            st.session_state.ltc_model_sources = []
        if "ltc_model_add_gen" not in st.session_state:
            st.session_state.ltc_model_add_gen = 0

        if st.session_state.ltc_model_sources:
            st.write(f"**Configured sources ({len(st.session_state.ltc_model_sources)}):**")
            for _idx, _src in enumerate(st.session_state.ltc_model_sources):
                _rc1, _rc2 = st.columns([6, 1])
                with _rc1:
                    st.write(f"\u2713 **{_src['label']}** \u2014 {_src['height']:.0f} m \u2014 "
                             f"UTC{_src['utc_offset']:+.1f} \u2014 WS: `{_src['ws_col']}`"
                             + (f", WD: `{_src['wd_col']}`" if _src['wd_col']
                                else " (no WD mapped)"))
                with _rc2:
                    if st.button("Remove", key=f"ltc_remove_src_{_idx}"):
                        st.session_state.ltc_model_sources.pop(_idx)
                        st.rerun()

            with st.expander("Map: where each modelled source is located", expanded=True):
                st.plotly_chart(build_model_locations_map_fig(st.session_state.ltc_model_sources),
                                 width="stretch", key="ltc_model_locations_map")

        with st.expander("+ Add a modelled source",
                          expanded=(len(st.session_state.ltc_model_sources) == 0)):
            _magen = st.session_state.ltc_model_add_gen
            new_model_file = st.file_uploader(
                "Modelled wind dataset file (CSV or Vortex .txt)", type=["csv", "txt"],
                key=f"ltc_new_model_file_{_magen}")

            if new_model_file is not None:
                new_model_bytes = new_model_file.getvalue()
                new_is_vortex = sniff_vortex_format(new_model_bytes)

                if new_is_vortex:
                    (new_raw_model_df, new_vortex_height, new_vortex_tz,
                     new_vortex_lat, new_vortex_lon) = parse_vortex_txt(new_model_bytes)
                    st.success(f"Detected a Vortex-format file - Hub-Height="
                               f"{new_vortex_height:.0f} m, Timezone=UTC{new_vortex_tz:+.1f}.")
                    st.write("Preview:")
                    st.dataframe(new_raw_model_df.head(5), width="stretch")

                    new_model_cols = [c for c in new_raw_model_df.columns if c != "Timestamp"]
                    new_model_ts_col, new_model_dayfirst = "Timestamp", False

                    nmc1, nmc2 = st.columns(2)
                    with nmc1:
                        _ws_idx = guess_column(new_model_cols, ["m/s", "wspd", "speed"])
                        new_model_ws_col = st.selectbox("Wind speed column", new_model_cols,
                                                         index=_ws_idx,
                                                         key=f"ltc_new_ws_vortex_{_magen}")
                    with nmc2:
                        _wd_opts = ["(none)"] + new_model_cols
                        _wd_idx = guess_column(new_model_cols, ["deg", "wdir", "direction"])
                        _wd_choice = st.selectbox("Wind direction column (optional, for wind rose)",
                                                   _wd_opts, index=_wd_idx + 1,
                                                   key=f"ltc_new_wd_vortex_{_magen}")
                        new_model_wd_col = None if _wd_choice == "(none)" else _wd_choice

                    nmc3, nmc4 = st.columns(2)
                    with nmc3:
                        new_model_height = st.number_input(
                            "Height of the modelled dataset (m)", min_value=1.0,
                            value=new_vortex_height, step=1.0, key=f"ltc_new_h_vortex_{_magen}")
                    with nmc4:
                        new_model_label = st.text_input(
                            "Dataset name (for labeling charts)", value="Vortex",
                            key=f"ltc_new_label_vortex_{_magen}")

                    new_model_tz_default = new_vortex_tz
                    new_model_lat_default = new_vortex_lat if new_vortex_lat is not None else 0.0
                    new_model_lon_default = new_vortex_lon if new_vortex_lon is not None else 0.0
                else:
                    new_raw_model_df = read_raw_csv(new_model_bytes)
                    st.write("Preview:")
                    st.dataframe(new_raw_model_df.head(5), width="stretch")
                    new_model_cols = list(new_raw_model_df.columns)

                    nmc1, nmc2 = st.columns(2)
                    with nmc1:
                        new_model_ts_col = st.selectbox("Timestamp column", new_model_cols,
                                                         key=f"ltc_new_ts_{_magen}")
                        new_model_dayfirst = st.checkbox("Date format is day-first (DD/MM/YYYY)",
                                                          value=False, key=f"ltc_new_df_{_magen}")
                    with nmc2:
                        new_model_ws_col = st.selectbox("Wind speed column", new_model_cols,
                                                         key=f"ltc_new_ws_{_magen}")
                        _wd_choice = st.selectbox("Wind direction column (optional, for wind rose)",
                                                   ["(none)"] + new_model_cols,
                                                   key=f"ltc_new_wd_{_magen}")
                        new_model_wd_col = None if _wd_choice == "(none)" else _wd_choice

                    nmc3, nmc4 = st.columns(2)
                    with nmc3:
                        new_model_height = st.number_input(
                            "Height of the modelled dataset (m)", min_value=1.0,
                            value=detect_height_from_colname(new_model_ws_col, default=100.0),
                            step=1.0, key=f"ltc_new_h_{_magen}")
                    with nmc4:
                        new_model_label = st.text_input(
                            "Dataset name (for labeling charts, e.g. ERA5, CFSR)",
                            value=f"Source {len(st.session_state.ltc_model_sources) + 1}",
                            key=f"ltc_new_label_{_magen}")

                    new_model_tz_default = 0.0
                    new_model_lat_default, new_model_lon_default = 0.0, 0.0

                new_model_df = build_clean_df(new_raw_model_df, new_model_ts_col,
                                               new_model_dayfirst, ())
                if not new_is_vortex:
                    timestamp_diagnostics_ui(new_raw_model_df, new_model_ts_col,
                                              new_model_dayfirst, key_prefix=f"newmodel_{_magen}")

                ntz1, ntz2 = st.columns(2)
                with ntz1:
                    new_model_utc_offset = st.number_input(
                        "Timezone offset from UTC (hours)", value=new_model_tz_default, step=0.5,
                        key=f"ltc_new_tz_{_magen}")
                with ntz2:
                    st.caption("Most reanalysis products (ERA5, CFSR, MERRA-2) are already UTC "
                               "(0); Vortex files are typically local time and pre-filled above.")

                nloc1, nloc2 = st.columns(2)
                with nloc1:
                    new_model_lat = st.number_input(
                        "Latitude", min_value=-90.0, max_value=90.0,
                        value=new_model_lat_default, format="%.5f", key=f"ltc_new_lat_{_magen}")
                with nloc2:
                    new_model_lon = st.number_input(
                        "Longitude", min_value=-180.0, max_value=180.0,
                        value=new_model_lon_default, format="%.5f", key=f"ltc_new_lon_{_magen}")

                if st.button("Add this source", type="primary", key=f"ltc_add_src_{_magen}"):
                    st.session_state.ltc_model_sources.append({
                        "label": new_model_label, "is_vortex": new_is_vortex,
                        "model_df": new_model_df, "ws_col": new_model_ws_col,
                        "wd_col": new_model_wd_col, "height": new_model_height,
                        "utc_offset": new_model_utc_offset, "lat": new_model_lat,
                        "lon": new_model_lon,
                    })
                    st.session_state.ltc_model_add_gen += 1
                    st.rerun()

        if st.session_state.ltc_model_sources:
            # Resolve each configured source: UTC-align, hourly-resample, detect resolution -
            # measurement UTC offset (set right after Step 1) stays a single, global setting -
            # it doesn't vary by which model it's being correlated against; only the model
            # side is per-source.
            resolved_sources = []
            for src in st.session_state.ltc_model_sources:
                s_df_utc = src["model_df"].copy()
                s_df_utc.index = s_df_utc.index - pd.Timedelta(hours=src["utc_offset"])
                s_res_minutes = detect_resolution_minutes(src["model_df"].index)
                s_samples_per_hour = max(1, round(60 / s_res_minutes))
                s_is_subhourly = s_res_minutes < 55
                s_ws_hourly = resample_to_hourly(s_df_utc[src["ws_col"]], s_samples_per_hour)
                resolved_sources.append({
                    **src, "df_utc": s_df_utc, "res_minutes": s_res_minutes,
                    "samples_per_hour": s_samples_per_hour, "is_subhourly": s_is_subhourly,
                    "ws_hourly": s_ws_hourly,
                })

            with st.expander("Diagnostics: measurement vs modelled dataset coverage per source",
                              expanded=False):
                st.write(f"Measurement (UTC): {meas_df_utc.index.min()} to "
                         f"{meas_df_utc.index.max()}  ({len(meas_df_utc)} rows)")
                for rs in resolved_sources:
                    _valid_count = rs["df_utc"][rs["ws_col"]].notna().sum()
                    st.write(f"**{rs['label']}** (UTC): {rs['df_utc'].index.min()} to "
                             f"{rs['df_utc'].index.max()}  ({len(rs['df_utc'])} rows, "
                             f"{_valid_count} with a valid '{rs['ws_col']}' value)")
                    _overlap_start = max(meas_df_utc.index.min(), rs["df_utc"].index.min())
                    _overlap_end = min(meas_df_utc.index.max(), rs["df_utc"].index.max())
                    if _overlap_start > _overlap_end:
                        st.error(f"NO nominal overlap for {rs['label']} - check its timezone "
                                 f"offset and date range.")
                    elif _valid_count == 0:
                        st.error(f"{rs['label']}'s '{rs['ws_col']}' column has ZERO valid values.")

            st.divider()
            st.header("3. Analysis settings")
            st.caption("These settings feed the Shear, Correlation and Long-Term tabs below.")
            s1, s2 = st.columns(2)
            with s1:
                min_avail = st.slider("Minimum data availability to include a height in the "
                                       "shear fit (%)", 0, 100, 80)
            with s2:
                interest_height = st.number_input(
                    "Height of interest for the long-term result (m)", min_value=1.0,
                    value=float(sorted_heights(height_map)[0]["height"]))

            shear_data = compute_shear_data(meas_df, height_map, min_availability=min_avail)
            if shear_data is not None:
                st.success(f"Shear exponent (alpha) = {shear_data['alpha']:.3f}  |  "
                           f"heights used: {shear_data['heights_used']}")
            else:
                st.warning("Fewer than 3 heights meet the availability threshold - shear-dependent "
                           "results (extrapolation, long-term at non-matching heights) won't be "
                           "available until this is resolved.")

            # Correlate each source at ITS OWN height, computing Hourly/Daily/Monthly (+native
            # if sub-hourly) panels - basis for the Correlation tab's comparison and for picking
            # which source feeds the Long-Term Result.
            for rs in resolved_sources:
                rs["target_series"], rs["target_desc"], rs["target_ref_h"] = None, None, None
                rs["merged_hourly"], rs["daily_avg"], rs["monthly_avg"] = None, None, None
                rs["native_merged"], rs["native_label"] = None, None
                rs["lt"] = None
                rs["lt_ws_at_source_height"] = None
                rs["lt_ws_at_interest"] = None
                rs["hourly_r2"] = None
                if shear_data is None:
                    continue
                t_series, t_desc, t_ref_h = get_measurement_at_target_height(
                    meas_series_by_height, shear_data, rs["height"])
                rs["target_series"], rs["target_desc"], rs["target_ref_h"] = t_series, t_desc, t_ref_h
                if t_series is None:
                    continue
                merged_hourly = merge_concurrent(t_series, rs["ws_hourly"])
                rs["merged_hourly"] = merged_hourly
                if len(merged_hourly) >= 2:
                    rs["daily_avg"], rs["monthly_avg"] = build_daily_monthly(merged_hourly)
                    _, _hourly_stats = correlation_fig(merged_hourly, "Hourly", rs["label"])
                    if _hourly_stats is not None:
                        rs["hourly_r2"] = _hourly_stats["R2"]
                    rs["lt"] = long_term_correction(merged_hourly, rs["ws_hourly"])
                    rs["lt_ws_at_source_height"] = rs["lt"]["tls"]["lt_mean"]
                    rs["lt_ws_at_interest"] = (rs["lt_ws_at_source_height"] *
                                                (interest_height / rs["height"]) ** shear_data["alpha"])
                if rs["is_subhourly"]:
                    nt_series, nt_desc, _ = get_measurement_at_target_height(
                        meas_series_by_height_native, shear_data, rs["height"])
                    if nt_series is not None:
                        rs["native_merged"] = merge_concurrent(nt_series, rs["df_utc"][rs["ws_col"]])
                        rs["native_label"] = f"{rs['res_minutes']:.0f}-min (native)"

            valid_lt_sources = [rs for rs in resolved_sources if rs["lt"] is not None]
            lt_source = None
            if valid_lt_sources:
                _r2_key = lambda r: r["hourly_r2"] if r["hourly_r2"] is not None else -1
                _best_idx = int(np.argmax([_r2_key(rs) for rs in valid_lt_sources]))
                st.subheader("Which source should the Long-Term Result use?")
                st.caption(f"Defaulted to '{valid_lt_sources[_best_idx]['label']}', the highest "
                           f"hourly R\u00b2 among your configured sources - check the Correlation "
                           f"tab below to see the full comparison yourself before deciding.")
                lt_source_labels = [rs["label"] for rs in valid_lt_sources]
                lt_source_choice = st.selectbox("Long-term source", lt_source_labels,
                                                 index=_best_idx, key="ltc_lt_source_choice")
                lt_source = valid_lt_sources[lt_source_labels.index(lt_source_choice)]

            st.divider()
            st.header("4. Results")

            tabs = st.tabs(["Data Availability", "Monthly Means", "Wind Rose", "Shear Profile",
                             "Correlation", "Long-Term Result"])

            height_labels = [f"{hm['height']:.0f} m" for hm in sorted_heights(height_map)]

            # ---- Data availability ----
            with tabs[0]:
                st.subheader("Data availability by height and month")
                table = availability_table(meas_df, height_map)
                fig = plot_availability_bars(table, threshold=min_avail)
                show_fig(fig, width=WIDTH_AVAILABILITY)
                st.download_button(
                    "Download availability table (CSV)",
                    table.to_csv().encode(), file_name="data_availability.csv", mime="text/csv")

            # ---- Monthly means ----
            with tabs[1]:
                st.subheader("Monthly mean wind speed")
                for hm in sorted_heights(height_map):
                    mm, inc, om = monthly_mean_data(meas_df, hm["ws_col"],
                                                     samples_per_day=samples_per_day)
                    fig = render_monthly_fig(mm, inc, om, f"{hm['height']:.0f} m")
                    show_fig(fig, width=WIDTH_MONTHLY)

            # ---- Wind rose ----
            with tabs[2]:
                rose_heights = [hm for hm in sorted_heights(height_map)
                                 if hm["wd_col"] is not None]
                rose_source_candidates = [rs for rs in resolved_sources if rs["wd_col"] is not None]
                if not rose_heights:
                    st.warning("No wind direction column was mapped for any height - "
                               "add one in Step 1a to enable wind roses.")
                elif not rose_source_candidates:
                    st.warning("No wind direction column was mapped for any modelled source - "
                               "add one when configuring a source to enable wind roses.")
                else:
                    rose_height_label = st.selectbox(
                        "Measured height for wind rose", [f"{hm['height']:.0f} m"
                                                            for hm in rose_heights],
                        key="rose_height")
                    hm_r = next(hm for hm in rose_heights
                                if f"{hm['height']:.0f} m" == rose_height_label)
                    st.caption("Each source's panel uses its own native-resolution data over the "
                               "period where it overlaps with your measurement data - no hourly "
                               "averaging or concurrent-timestamp matching, since a wind rose "
                               "describes a distribution over a period rather than a point-by-"
                               "point comparison (that's only needed for correlation).")
                    for rs_r in rose_source_candidates:
                        st.subheader(f"Measured vs {rs_r['label']}")
                        _rose_start = max(meas_df_utc.index.min(), rs_r["df_utc"].index.min())
                        _rose_end = min(meas_df_utc.index.max(), rs_r["df_utc"].index.max())
                        if _rose_start > _rose_end:
                            st.warning(f"No overlapping period between the measurement data and "
                                       f"{rs_r['label']}.")
                            continue
                        meas_combined, model_combined = rose_source_data(
                            meas_df_utc.loc[_rose_start:_rose_end, hm_r["ws_col"]],
                            meas_df_utc.loc[_rose_start:_rose_end, hm_r["wd_col"]],
                            rs_r["df_utc"].loc[_rose_start:_rose_end, rs_r["ws_col"]],
                            rs_r["df_utc"].loc[_rose_start:_rose_end, rs_r["wd_col"]])
                        fig = render_rose_fig(meas_combined, model_combined, rose_height_label,
                                               f"{rs_r['label']} ({rs_r['height']:.0f} m)")
                        if fig is None:
                            st.warning(f"Not enough data for {rs_r['label']} over the "
                                       f"overlapping period.")
                        else:
                            show_fig(fig, width=WIDTH_ROSE)
                            if abs(hm_r["height"] - rs_r["height"]) > 2:
                                st.caption(f"Note: the measured panel is at {hm_r['height']:.0f} m "
                                           f"and the modelled panel is at {rs_r['label']}'s height "
                                           f"({rs_r['height']:.0f} m) - shown side by side but not "
                                           "height-matched, since wind rose is primarily about "
                                           "directional shape.")

            # ---- Shear ----
            with tabs[3]:
                st.subheader("Shear exponent (profile method)")
                st.caption(f"Using the availability threshold set above ({min_avail}%).")
                if shear_data is None:
                    st.warning("Fewer than 3 heights meet the availability threshold - "
                               "lower the threshold above or check your data.")
                else:
                    st.metric("Shear exponent (alpha)", f"{shear_data['alpha']:.3f}",
                              help=f"R2 = {shear_data['r2']:.3f}")
                    st.write(f"Heights used (low to high): {shear_data['heights_used']}")
                    if shear_data["excluded"]:
                        st.write(f"Heights excluded (availability %): {shear_data['excluded']}")
                    fig = render_shear_fig(shear_data)
                    show_fig(fig, width=WIDTH_SHEAR)

            # ---- Correlation ----
            with tabs[4]:
                st.subheader("Correlation comparison across modelled sources")
                st.caption("Each source is correlated at its own height, so measurement and "
                           "model are compared like-for-like within each source. Organised by "
                           "panel type, with every source side by side, so the same statistic "
                           "is directly comparable across sources.")
                _valid_sources = [rs for rs in resolved_sources
                                   if rs["merged_hourly"] is not None
                                   and len(rs["merged_hourly"]) >= 2]
                if not _valid_sources:
                    st.warning("No concurrent overlap found between measurement and any "
                               "modelled source. Check the timezone offsets and date ranges.")
                else:
                    for rs in _valid_sources:
                        st.write(f"**{rs['label']}** ({rs['height']:.0f} m): using measurement "
                                 f"at {rs['height']:.0f} m ({rs['target_desc']}).")

                    panel_groups = [("Hourly", "merged_hourly", None),
                                     ("Daily Average", "daily_avg", None),
                                     ("Monthly Average", "monthly_avg", None)]
                    if any(rs["native_merged"] is not None for rs in _valid_sources):
                        panel_groups.insert(0, ("Native Resolution", "native_merged", "native_label"))

                    _N_COLS = 4  # always reserve this many columns, even if fewer sources are
                    # configured - so every panel's column is always page_width/4, never the
                    # full page width (which happened with 1 source before) and never narrower
                    # than what width="stretch" can safely fill (which caused overflow with a
                    # hardcoded pixel width when more sources made the real columns narrower)
                    for group_label, data_key, label_key in panel_groups:
                        _sources_with_data = [rs for rs in _valid_sources
                                               if rs.get(data_key) is not None
                                               and len(rs[data_key]) >= 2]
                        if not _sources_with_data:
                            continue
                        st.markdown(f"**{group_label}**")
                        for _row_start in range(0, len(_sources_with_data), _N_COLS):
                            _row_sources = _sources_with_data[_row_start:_row_start + _N_COLS]
                            cols = st.columns(_N_COLS)
                            for col, rs in zip(cols, _row_sources):
                                with col:
                                    _panel_label = rs[label_key] if label_key else group_label
                                    fig, stats_dict = correlation_fig(rs[data_key], _panel_label,
                                                                       rs["label"])
                                    if fig is None:
                                        st.warning(f"Not enough data for {rs['label']}.")
                                    else:
                                        show_fig(fig, width="stretch")
                                        st.caption(f"{rs['label']}: n={stats_dict['n']}, "
                                                   f"R2={stats_dict['R2']:.3f}")

            # ---- Long-term result ----
            with tabs[5]:
                st.subheader("Long-term wind speed at your height of interest")

                if shear_data is None:
                    st.warning("Shear exponent could not be computed (need >=3 heights at the "
                               "chosen availability threshold) - cannot extrapolate to the "
                               "height of interest.")
                elif not valid_lt_sources:
                    st.warning("No modelled source has enough concurrent overlap with the "
                               "measurement data yet.")
                else:
                    alpha = shear_data["alpha"]
                    if len(valid_lt_sources) > 1:
                        st.write("**Comparison across all configured sources:**")
                        _lt_rows = [
                            {"Source": r["label"], "Height (m)": f"{r['height']:.0f}",
                             "Hourly R\u00b2": (f"{r['hourly_r2']:.3f}"
                                                if r["hourly_r2"] is not None else "-"),
                             f"LT WS @ {interest_height:.0f} m (m/s)": (
                                 f"{r['lt_ws_at_interest']:.3f}"
                                 if r["lt_ws_at_interest"] is not None else "-"),
                             "Selected": "\u2713" if r is lt_source else ""}
                            for r in valid_lt_sources]
                        st.dataframe(pd.DataFrame(_lt_rows), hide_index=True, width="stretch")
                        st.caption("Every configured source gets the full breakdown below, not "
                                   "just the one selected above the tabs - since you're "
                                   "analysing measurements, it's worth seeing everything before "
                                   "deciding what to rely on.")

                    for rs in valid_lt_sources:
                        st.divider()
                        _is_selected = rs is lt_source
                        st.markdown(f"### {rs['label']}" +
                                    ("  \u2b50 *Selected for headline reporting*"
                                     if _is_selected else ""))
                        st.write(f"Correlation basis: measurement at {rs['height']:.0f} m "
                                 f"({rs['target_desc']}).")
                        st.write(f"Concurrent period: {rs['lt']['n_concurrent']} hours "
                                 f"(concurrent mean measured = "
                                 f"{rs['lt']['concurrent_meas_mean']:.3f} m/s, "
                                 f"concurrent mean {rs['label']} = "
                                 f"{rs['lt']['concurrent_model_mean']:.3f} m/s)")
                        st.write(f"Full {rs['label']} record: "
                                 f"{rs['lt']['lt_model_start'].date()} to "
                                 f"{rs['lt']['lt_model_end'].date()}, long-term mean "
                                 f"{rs['label']} = {rs['lt']['lt_model_mean']:.3f} m/s")

                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric(f"Long-term wind speed at {rs['height']:.0f} m "
                                      f"({rs['label']} height)",
                                      f"{rs['lt_ws_at_source_height']:.3f} m/s")
                        with m2:
                            st.metric(f"Long-term wind speed at {interest_height:.0f} m "
                                      f"(shear-extrapolated, alpha={alpha:.3f})",
                                      f"{rs['lt_ws_at_interest']:.3f} m/s")

                        fig = render_long_term_fig(shear_data, rs["height"], rs["label"],
                                                    rs["lt_ws_at_source_height"], interest_height,
                                                    rs["lt_ws_at_interest"])
                        show_fig(fig, width=WIDTH_SHEAR + 100)

                        st.caption(f"For reference, OLS fit gives a long-term mean of "
                                   f"{rs['lt']['ols']['lt_mean']:.3f} m/s at "
                                   f"{rs['height']:.0f} m. The orthogonal (TLS) result above is "
                                   "used as the primary estimate since OLS understates slope "
                                   "when both series carry noise.")

            st.divider()
            st.header("5. Download")
            st.caption("Bundles every chart above into a single ZIP, generated at the settings "
                       "currently selected (availability threshold, heights, etc) - covers "
                       "every configured source, not just the one selected for headline "
                       "reporting.")
            if st.button("Prepare all plots for download"):
                with st.spinner("Rendering all plots..."):
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        # 1. Data availability
                        table = availability_table(meas_df, height_map)
                        zf.writestr("01_data_availability.png",
                                    fig_to_png_bytes(plot_availability_bars(table,
                                                                             threshold=min_avail)))
                        zf.writestr("01_data_availability.csv", table.to_csv())

                        # 2. Monthly means - one per mapped height
                        for hm in sorted_heights(height_map):
                            mm, inc, om = monthly_mean_data(meas_df, hm["ws_col"],
                                                             samples_per_day=samples_per_day)
                            fig = render_monthly_fig(mm, inc, om, f"{hm['height']:.0f} m")
                            zf.writestr(f"02_monthly_mean_{hm['height']:.0f}m.png",
                                        fig_to_png_bytes(fig))

                        # 3. Wind rose - the height currently selected on screen (falls back to
                        # the first WD-mapped height if the tab was never reached), vs every
                        # source that has a direction column mapped - independent native-
                        # resolution periods, not concurrent-matched (matching the tab).
                        rose_heights = [hm for hm in sorted_heights(height_map)
                                        if hm["wd_col"] is not None]
                        if rose_heights:
                            _selected_label = st.session_state.get("rose_height")
                            hm_r = next((hm for hm in rose_heights
                                         if f"{hm['height']:.0f} m" == _selected_label),
                                        rose_heights[0])
                            for rs in resolved_sources:
                                if rs["wd_col"] is None:
                                    continue
                                _rs_start = max(meas_df_utc.index.min(), rs["df_utc"].index.min())
                                _rs_end = min(meas_df_utc.index.max(), rs["df_utc"].index.max())
                                if _rs_start > _rs_end:
                                    continue
                                meas_combined, model_combined = rose_source_data(
                                    meas_df_utc.loc[_rs_start:_rs_end, hm_r["ws_col"]],
                                    meas_df_utc.loc[_rs_start:_rs_end, hm_r["wd_col"]],
                                    rs["df_utc"].loc[_rs_start:_rs_end, rs["ws_col"]],
                                    rs["df_utc"].loc[_rs_start:_rs_end, rs["wd_col"]])
                                fig = render_rose_fig(meas_combined, model_combined,
                                                       f"{hm_r['height']:.0f} m",
                                                       f"{rs['label']} ({rs['height']:.0f} m)")
                                if fig is not None:
                                    _safe_label = rs["label"].replace(" ", "_")
                                    zf.writestr(f"03_wind_rose_vs_{_safe_label}.png",
                                                fig_to_png_bytes(fig))

                        # 4. Shear profile
                        if shear_data is not None:
                            zf.writestr("04_shear_profile.png",
                                        fig_to_png_bytes(render_shear_fig(shear_data)))

                        # 5. Correlation - every source, every available panel
                        for rs in resolved_sources:
                            if rs["merged_hourly"] is None or len(rs["merged_hourly"]) < 2:
                                continue
                            _safe_label = rs["label"].replace(" ", "_")
                            _panels = [("Hourly", rs["merged_hourly"]),
                                       ("Daily Average", rs["daily_avg"]),
                                       ("Monthly Average", rs["monthly_avg"])]
                            if rs["native_merged"] is not None:
                                _panels.insert(0, (rs["native_label"], rs["native_merged"]))
                            for _pname, _pdata in _panels:
                                if _pdata is None or len(_pdata) < 2:
                                    continue
                                fig, _ = correlation_fig(_pdata, _pname, rs["label"])
                                if fig is not None:
                                    _fname_part = _pname.lower().replace(" ", "_").replace("-", "_")
                                    zf.writestr(f"05_correlation_{_safe_label}_{_fname_part}.png",
                                                fig_to_png_bytes(fig))

                        # 6. Long-term wind speed at height of interest - EVERY configured
                        # source gets its own chart, not just the one selected for headline
                        # reporting, since the point of comparing sources is to have everything
                        # available to look back on, not just the final pick.
                        if valid_lt_sources:
                            for rs in valid_lt_sources:
                                fig = render_long_term_fig(shear_data, rs["height"], rs["label"],
                                                            rs["lt_ws_at_source_height"],
                                                            interest_height,
                                                            rs["lt_ws_at_interest"])
                                _safe_label = rs["label"].replace(" ", "_")
                                _selected_tag = "_SELECTED" if rs is lt_source else ""
                                zf.writestr(f"06_long_term_{_safe_label}{_selected_tag}.png",
                                            fig_to_png_bytes(fig))

                            _comparison_lines = "\n".join(
                                f"  {r['label']}: Hourly R2={r['hourly_r2']:.3f}, "
                                f"LT WS @ {interest_height:.0f} m = {r['lt_ws_at_interest']:.3f} m/s"
                                f"{'  <-- selected for headline reporting' if r is lt_source else ''}"
                                for r in valid_lt_sources)
                            rs = lt_source
                            summary = (
                                f"Wind Resource Analysis - Summary\n"
                                f"=================================\n"
                                f"Shear exponent (alpha): {shear_data['alpha']:.3f}\n"
                                f"Heights used in shear fit: {shear_data['heights_used']}\n\n"
                                f"Modelled sources compared (every source, full detail also in "
                                f"the individual 06_long_term_*.png files):\n{_comparison_lines}\n\n"
                                f"Selected for headline reporting: {rs['label']} at "
                                f"{rs['height']:.0f} m\n"
                                f"Correlation basis: {rs['target_desc']}\n"
                                f"Concurrent hours: {rs['lt']['n_concurrent']}\n"
                                f"Long-term mean at {rs['height']:.0f} m (TLS): "
                                f"{rs['lt_ws_at_source_height']:.3f} m/s\n"
                                f"Long-term mean at {interest_height:.0f} m (shear-extrapolated): "
                                f"{rs['lt_ws_at_interest']:.3f} m/s\n"
                                f"Long-term mean at {rs['height']:.0f} m (OLS, reference only): "
                                f"{rs['lt']['ols']['lt_mean']:.3f} m/s\n"
                            )
                            zf.writestr("00_summary.txt", summary)

                    buf.seek(0)
                    st.session_state["plots_zip"] = buf.read()

            if "plots_zip" in st.session_state:
                st.download_button(
                    "Download all plots (ZIP)", st.session_state["plots_zip"],
                    file_name="wind_analysis_plots.zip", mime="application/zip")
                st.caption("Reflects the settings selected at the moment you clicked "
                           "'Prepare' - click it again after changing anything above.")
    else:
        st.info("Upload a measurement CSV to begin.")

elif mode == "Measurement Campaign Planning":
    st.title("Measurement Campaign Planning")
    st.caption("Preliminary wind resource look-up from modelled maps, and LiDAR/FLiDAR siting "
               "based on your site boundary, turbine layout, and modelled wind maps.")

    for _k, _v in [("camp_wind_maps", {}), ("camp_active_map", None), ("camp_boundary", None),
                   ("camp_layout", None), ("camp_points", []), ("camp_points_fixed", False),
                   ("camp_best_points", []), ("camp_last_click", None),
                   ("camp_uploader_gen", 0), ("camp_wm_uploader_gen", 0),
                   ("camp_wm_sources", ["ERA5", "CFSR"])]:
        if _k not in st.session_state:
            st.session_state[_k] = _v
    _cgen = st.session_state.camp_uploader_gen

    # ------------------------------------------------------------------ STEP 1 --
    st.header("1. Site boundary")
    render_boundary_uploader("camp", _cgen)

    st.divider()

    # ------------------------------------------------------------------ STEP 2 --
    st.header("2. Wind maps (modelled data)")
    st.caption("Upload every .asc file for each source - drag-and-drop the whole folder "
               "(Chrome/Edge expand a dropped folder automatically) or select all the files "
               "at once. Height is read from each filename (e.g. 'site.M.100m.asc'). Rename, "
               "remove, or add a source as needed - unlike Preliminary WRA's fixed ERA5/CFSR "
               "pair, any number of sources with any names are supported here.")

    _wm_cols = st.columns(len(st.session_state.camp_wm_sources))
    for _i, _wm_col in enumerate(_wm_cols):
        with _wm_col:
            _label = st.text_input("Source name", value=st.session_state.camp_wm_sources[_i],
                                    key=f"camp_wm_source_name_{_i}")
            st.session_state.camp_wm_sources[_i] = _label
            _resolved_label = _label.strip() or f"Source {_i + 1}"
            _files = st.file_uploader(
                f"{_resolved_label} .asc files", type=["asc"], accept_multiple_files=True,
                key=f"camp_wm_files_{_i}_{st.session_state.camp_wm_uploader_gen}")
            if _files:
                _skipped = []
                for _f in _files:
                    _h = detect_asc_height(_f.name)
                    if _h is None:
                        _skipped.append(_f.name)
                        continue
                    _data, _meta = cached_read_ascii_grid(_f.getvalue())
                    _wm_key = (_resolved_label, f"{_h:.0f}")
                    st.session_state.camp_wind_maps[_wm_key] = (_data, _meta)
                    st.session_state.camp_active_map = _wm_key
                if _skipped:
                    st.warning(f"Could not detect height from {len(_skipped)} file(s): "
                               f"{', '.join(_skipped[:5])}{' ...' if len(_skipped) > 5 else ''}")
            if len(st.session_state.camp_wm_sources) > 1:
                if st.button("Remove this source", key=f"camp_wm_remove_src_{_i}"):
                    _removed_label = st.session_state.camp_wm_sources.pop(_i)
                    st.session_state.camp_wind_maps = {
                        k: v for k, v in st.session_state.camp_wind_maps.items()
                        if k[0] != (_removed_label.strip() or f"Source {_i + 1}")}
                    if (st.session_state.camp_active_map is not None and
                            st.session_state.camp_active_map not in st.session_state.camp_wind_maps):
                        st.session_state.camp_active_map = (
                            next(iter(st.session_state.camp_wind_maps), None))
                    st.rerun()

    if st.button("+ Add another source"):
        st.session_state.camp_wm_sources.append(
            f"Source {len(st.session_state.camp_wm_sources) + 1}")
        st.rerun()

    if st.session_state.camp_wind_maps:
        map_keys = list(st.session_state.camp_wind_maps.keys())
        map_labels = [f"{s} @ {h} m" for s, h in map_keys]

        st.write(f"**Loaded wind maps ({len(map_keys)}):**")
        loaded_df = pd.DataFrame([{"Source": s, "Height (m)": h} for s, h in map_keys])
        st.dataframe(loaded_df, hide_index=True, width="stretch")

        default_idx = (map_keys.index(st.session_state.camp_active_map)
                       if st.session_state.camp_active_map in map_keys else 0)
        mapc1, mapc2 = st.columns([3, 1])
        with mapc1:
            chosen_label = st.selectbox("Active map (shown on the map below)", map_labels,
                                         index=default_idx, key="camp_active_map_select")
            st.session_state.camp_active_map = map_keys[map_labels.index(chosen_label)]
        with mapc2:
            st.write("")
            st.write("")
            if st.button("Remove this map"):
                del st.session_state.camp_wind_maps[st.session_state.camp_active_map]
                st.session_state.camp_active_map = (list(st.session_state.camp_wind_maps.keys())[0]
                                                     if st.session_state.camp_wind_maps else None)
                st.rerun()

    st.divider()

    # ------------------------------------------------------------------ STEP 3 --
    st.header("3. Turbine layout (optional)")
    st.caption("Used for the best-measurement-point search (step 6) and optional wind speed "
               "labels on the map.")
    render_layout_uploader("camp", _cgen)

    show_layout_ws = False
    if st.session_state.camp_layout is not None:
        st.success(f"Layout loaded ({len(st.session_state.camp_layout)} turbines).")
        show_layout_ws = st.checkbox("Show wind speed at each turbine on the map",
                                      key="camp_show_layout_ws")

    st.divider()

    # ------------------------------------------------------------------ STEP 4 --
    st.header("4. Interactive map")
    if st.session_state.camp_boundary is None:
        st.info("Upload a site boundary above to see the map.")
    else:
        boundary_gdf = st.session_state.camp_boundary
        bounds = tuple(boundary_gdf.total_bounds)
        active = (st.session_state.camp_wind_maps.get(st.session_state.camp_active_map)
                  if st.session_state.camp_active_map else None)

        with st.expander("Click-grid density (advanced)"):
            n_grid = st.slider("Grid points per side inside the boundary", 15, 80, 45,
                                key="camp_grid_density",
                                help="Higher = finer click resolution when placing points, "
                                     "but slower to build.")

        click_grid = None
        if not st.session_state.camp_points_fixed:
            click_grid = generate_clickable_grid(
                boundary_gdf.geometry.iloc[0].wkt, bounds, n_grid=n_grid)

        fig, grid_curve_idx = build_planning_map_fig(
            active, boundary_gdf, st.session_state.camp_layout, show_layout_ws,
            st.session_state.camp_points, st.session_state.camp_best_points, click_grid)

        st.caption("Click anywhere inside the boundary to drop a measurement point." if
                   not st.session_state.camp_points_fixed else
                   "Points are fixed - click 'Clear points' below to place new ones.")
        event = st.plotly_chart(fig, width="stretch", on_select="rerun",
                                 selection_mode=("points",), key="camp_map_chart")

        if not st.session_state.camp_points_fixed and grid_curve_idx is not None:
            sel_points = []
            if event is not None:
                sel = getattr(event, "selection", None) or {}
                sel_points = sel.get("points", []) if hasattr(sel, "get") else getattr(sel, "points", [])
            if sel_points:
                p = sel_points[0]
                if p.get("curve_number") == grid_curve_idx:
                    clicked = (round(p["y"], 6), round(p["x"], 6))
                    if clicked != st.session_state.camp_last_click:
                        st.session_state.camp_points.append(clicked)
                        st.session_state.camp_last_click = clicked
                        st.rerun()

        pc1, pc2, pc3 = st.columns([1, 1, 2])
        with pc1:
            if st.button("Fix measurement points"):
                st.session_state.camp_points_fixed = True
                st.rerun()
        with pc2:
            if st.button("Clear points"):
                st.session_state.camp_points = []
                st.session_state.camp_points_fixed = False
                st.session_state.camp_last_click = None
                st.rerun()

        if st.session_state.camp_points:
            if st.session_state.camp_points_fixed:
                st.success("Comparison Points Fixed. See location in the table below.")
            labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            pts_df = pd.DataFrame([
                {"Label": labels[i], "Latitude": lat, "Longitude": lon}
                for i, (lat, lon) in enumerate(st.session_state.camp_points)
            ])
            st.dataframe(pts_df, width="stretch", hide_index=True)

    st.divider()

    # ------------------------------------------------------------------ STEP 5 --
    st.header("5. Compare wind speed at chosen points")
    if "camp_comparison_combos" not in st.session_state:
        st.session_state.camp_comparison_combos = []

    if not st.session_state.camp_points:
        st.info("Place at least one measurement point on the map above.")
    elif not st.session_state.camp_wind_maps:
        st.info("Load at least one wind map above.")
    else:
        combo_text = st.text_input("Combinations to compare, semicolon-separated "
                                    "(e.g. A,B;A,C)", value="", key="camp_combo")
        if st.button("Generate comparison plots"):
            labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            points = {labels[i]: pt for i, pt in enumerate(st.session_state.camp_points)}
            combos = [c.strip() for c in combo_text.split(";") if c.strip()]
            valid_combos = []
            if not combos:
                st.error("Enter at least one combination.")
            for combo in combos:
                combo_labels = [c.strip() for c in combo.split(",")]
                missing = [l for l in combo_labels if l not in points]
                if missing:
                    st.error(f"Unknown point label(s) in '{combo}': {missing}")
                    continue
                valid_combos.append(combo)
            st.session_state.camp_comparison_combos = valid_combos

        # Rendered unconditionally from session_state (not gated behind the button
        # above) so these stay visible across reruns triggered by other buttons,
        # e.g. "Locate best measurement points" below.
        if st.session_state.camp_comparison_combos:
            labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            points = {labels[i]: pt for i, pt in enumerate(st.session_state.camp_points)}
            for combo in st.session_state.camp_comparison_combos:
                combo_labels = [c.strip() for c in combo.split(",")]
                if any(l not in points for l in combo_labels):
                    continue  # points changed since this combo was generated
                fig = render_comparison_fig(combo_labels, points, st.session_state.camp_wind_maps)
                show_fig(fig, width=WIDTH_SHEAR)

    st.divider()

    # ------------------------------------------------------------------ STEP 6 --
    st.header("6. Locate best measurement points (K-Means)")
    st.caption("Groups your turbines into clusters, then searches inside the boundary for the "
               "point in each cluster whose modelled wind speed best represents that cluster "
               "(weighted against distance to the turbines it represents).")
    if (st.session_state.camp_layout is None or st.session_state.camp_active_map is None
            or st.session_state.camp_boundary is None):
        st.info("Load a boundary, a layout, and at least one wind map to use this.")
    else:
        n_clusters = st.number_input("Number of measurement locations", min_value=1, max_value=20,
                                      value=1, step=1, key="camp_n_clusters")
        if st.button("Locate best measurement points", type="primary"):
            data, meta = st.session_state.camp_wind_maps[st.session_state.camp_active_map]
            boundary_gdf = st.session_state.camp_boundary
            bounds = tuple(boundary_gdf.total_bounds)
            coords = tuple((g.y, g.x) for g in st.session_state.camp_layout.geometry)
            best_points = run_best_points_search(
                data, meta, boundary_gdf.geometry.iloc[0].wkt, bounds, coords, int(n_clusters))
            st.session_state.camp_best_points = best_points
            if not best_points:
                st.warning("No valid points found - check that the wind map covers the "
                           "boundary area.")
            st.rerun()

        if st.session_state.camp_best_points:
            src, h = st.session_state.camp_active_map
            st.success(f"Found {len(st.session_state.camp_best_points)} recommended point(s), "
                       f"based on {src} @ {h} m.")
            bp_df = pd.DataFrame([
                {"Point": f"P{info['cluster']}", "Latitude": lat, "Longitude": lon,
                 "Wind Speed (m/s)": round(info["ws"], 3),
                 "Mean Deviation (%)": round(info["mean_dev"], 2),
                 "Max Deviation (%)": round(info["max_dev"], 2)}
                for lat, lon, info in st.session_state.camp_best_points
            ])
            st.dataframe(bp_df, width="stretch", hide_index=True)
            st.download_button("Download best points (CSV)",
                                bp_df.to_csv(index=False).encode("utf-8"),
                                file_name="best_measurement_points.csv", mime="text/csv")

    st.divider()

    # ------------------------------------------------------------------ STEP 7 --
    st.header("7. Download all statistics")
    st.caption("Bundles the comparison plots, an Excel workbook (site boundary, layout, "
               "layout wind speeds, and best measurement points), and one combined overview "
               "figure into a single ZIP.")
    if st.button("Prepare all statistics for download"):
        with st.spinner("Building download package..."):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                points = {labels[i]: pt for i, pt in enumerate(st.session_state.camp_points)}
                for combo in st.session_state.camp_comparison_combos:
                    combo_labels = [c.strip() for c in combo.split(",")]
                    if any(l not in points for l in combo_labels):
                        continue
                    fig = render_comparison_fig(combo_labels, points,
                                                 st.session_state.camp_wind_maps)
                    fname = f"comparison_{'_'.join(combo_labels)}.png"
                    zf.writestr(fname, fig_to_png_bytes(fig))

                excel_bytes = build_campaign_excel(
                    st.session_state.camp_boundary, st.session_state.camp_layout,
                    st.session_state.camp_wind_maps, st.session_state.camp_points,
                    st.session_state.camp_best_points)
                zf.writestr("campaign_data.xlsx", excel_bytes)

                active = (st.session_state.camp_wind_maps.get(st.session_state.camp_active_map)
                          if st.session_state.camp_active_map else None)
                overview_fig = render_full_campaign_map_fig(
                    active, st.session_state.camp_boundary, st.session_state.camp_layout,
                    st.session_state.camp_points, st.session_state.camp_best_points)
                zf.writestr("campaign_overview.png", fig_to_png_bytes(overview_fig))

            buf.seek(0)
            st.session_state["camp_zip"] = buf.read()

    if "camp_zip" in st.session_state:
        st.download_button("Download all statistics (ZIP)", st.session_state["camp_zip"],
                            file_name="measurement_campaign_planning.zip",
                            mime="application/zip")
        st.caption("Reflects the settings selected at the moment you clicked 'Prepare' - "
                   "click it again after changing anything above.")

    st.divider()
    if st.button("Reset planning tool"):
        _next_cgen = st.session_state.get("camp_uploader_gen", 0) + 1
        _next_wmgen = st.session_state.get("camp_wm_uploader_gen", 0) + 1
        for _k in list(st.session_state.keys()):
            if _k.startswith("camp_"):
                del st.session_state[_k]
        # Bump AFTER wiping, so every file_uploader above gets a brand-new key and
        # genuinely resets - deleting the session_state value alone isn't enough, since
        # the browser-side widget still holds its selected file(s) and would otherwise
        # silently re-populate everything right back on the very next rerun.
        st.session_state.camp_uploader_gen = _next_cgen
        st.session_state.camp_wm_uploader_gen = _next_wmgen
        st.rerun()

elif mode == "Preliminary Wind Resource Assessment":
    st.title("Preliminary Wind Resource Assessment")
    st.caption("Multi-source (ERA5 + CFSR) shear, hub-height extrapolation, weighting, and "
               "long-term calibration across a turbine layout.")

    for _k, _v in [("wra_boundary", None), ("wra_layout", None),
                   ("wra_wind_maps", {"ERA5": {}, "CFSR": {}}), ("wra_active_map", None),
                   ("wra_shear_results", None), ("wra_hh_results", None),
                   ("wra_cal_points", []), ("wra_cal_factors", None),
                   ("wra_calibrated_ws", None), ("wra_cal_influence", None),
                   ("wra_uploader_gen", 0)]:
        if _k not in st.session_state:
            st.session_state[_k] = _v
    _gen = st.session_state.wra_uploader_gen

    # ------------------------------------------------------------------ STEP 1 --
    st.header("1. Site Boundary")
    render_boundary_uploader("wra", _gen)

    st.divider()

    # ------------------------------------------------------------------ STEP 2 --
    st.header("2. Layout")
    render_layout_uploader("wra", _gen)

    if st.session_state.wra_layout is not None:
        st.success(f"Layout loaded ({len(st.session_state.wra_layout)} turbines).")

    st.divider()

    # ------------------------------------------------------------------ STEP 3 --
    st.header("3. Wind Maps")
    st.caption("Upload every .asc file for each source - drag-and-drop the whole folder "
               "(Chrome/Edge expand a dropped folder automatically) or select all the files "
               "at once. Height is read from each filename (e.g. 'site.M.100m.asc').")

    wmc1, wmc2 = st.columns(2)
    for _source, _col in [("ERA5", wmc1), ("CFSR", wmc2)]:
        with _col:
            st.subheader(_source)
            _files = st.file_uploader(f"{_source} .asc files", type=["asc"],
                                       accept_multiple_files=True,
                                       key=f"wra_{_source.lower()}_files_{_gen}")
            if _files:
                _maps, _skipped = {}, []
                for _f in _files:
                    _h = detect_asc_height(_f.name)
                    if _h is None:
                        _skipped.append(_f.name)
                        continue
                    _data, _meta = cached_read_ascii_grid(_f.getvalue())
                    _maps[_h] = (_data, _meta)
                st.session_state.wra_wind_maps[_source] = _maps
                if _maps:
                    st.success(f"{len(_maps)} height(s) loaded: {sorted(_maps.keys())} m")
                if _skipped:
                    st.warning(f"Could not detect height from {len(_skipped)} file(s): "
                               f"{', '.join(_skipped[:5])}{' ...' if len(_skipped) > 5 else ''}")

    era5_maps = st.session_state.wra_wind_maps["ERA5"]
    cfsr_maps = st.session_state.wra_wind_maps["CFSR"]

    if era5_maps or cfsr_maps:
        st.write("**Loaded wind maps:**")
        _rows = ([{"Source": "ERA5", "Height (m)": h} for h in sorted(era5_maps.keys())] +
                 [{"Source": "CFSR", "Height (m)": h} for h in sorted(cfsr_maps.keys())])
        st.dataframe(pd.DataFrame(_rows), hide_index=True, width="stretch")

    st.divider()

    # ------------------------------------------------------------------ STEP 4 --
    st.header("4. Interactive Map")
    all_maps = {("ERA5", h): v for h, v in era5_maps.items()}
    all_maps.update({("CFSR", h): v for h, v in cfsr_maps.items()})

    if st.session_state.wra_boundary is None:
        st.info("Upload a site boundary above to see the map.")
    elif not all_maps:
        st.info("Load at least one wind map above to browse it here.")
    else:
        map_keys = sorted(all_maps.keys())
        map_labels = [f"{s} @ {h} m" for s, h in map_keys]
        default_idx = (map_keys.index(st.session_state.wra_active_map)
                       if st.session_state.wra_active_map in map_keys else 0)
        chosen_label = st.selectbox("Map to display", map_labels, index=default_idx,
                                     key="wra_active_map_select")
        st.session_state.wra_active_map = map_keys[map_labels.index(chosen_label)]

        wra_show_ws = False
        if st.session_state.wra_layout is not None:
            wra_show_ws = st.checkbox("Show wind speed at each turbine on the map",
                                       key="wra_show_ws")

        active = all_maps[st.session_state.wra_active_map]
        fig, _ = build_planning_map_fig(active, st.session_state.wra_boundary,
                                         st.session_state.wra_layout, wra_show_ws, [], [], None)
        st.plotly_chart(fig, width="stretch", key="wra_map_chart")

    st.divider()

    # ------------------------------------------------------------------ STEP 5 --
    st.header("5. Shear Calculation")
    st.caption("For every turbine, fits a power-law shear exponent to ERA5's own heights and "
               "CFSR's own heights separately (not one number pooled across the whole site), "
               "then averages the two per position. Needs at least 2 heights per source.")

    if st.session_state.wra_layout is None:
        st.info("Load a layout above to compute shear.")
    elif len(era5_maps) < 2 and len(cfsr_maps) < 2:
        st.info("Need at least 2 heights loaded for ERA5 or CFSR to compute shear.")
    else:
        if st.button("Compute Shear"):
            layout_coords = tuple((g.y, g.x) for g in st.session_state.wra_layout.geometry)
            st.session_state.wra_shear_results = compute_layout_shear(era5_maps, cfsr_maps,
                                                                        layout_coords)

        if st.session_state.wra_shear_results:
            sr = st.session_state.wra_shear_results
            era5_alphas = [r["era5_alpha"] for r in sr if not np.isnan(r["era5_alpha"])]
            cfsr_alphas = [r["cfsr_alpha"] for r in sr if not np.isnan(r["cfsr_alpha"])]
            avg_alphas = [r["avg_alpha"] for r in sr if not np.isnan(r["avg_alpha"])]
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Mean ERA5 shear", f"{np.mean(era5_alphas):.3f}" if era5_alphas else "n/a")
            with m2:
                st.metric("Mean CFSR shear", f"{np.mean(cfsr_alphas):.3f}" if cfsr_alphas else "n/a")
            with m3:
                st.metric("Mean combined shear", f"{np.mean(avg_alphas):.3f}" if avg_alphas else "n/a")

            with st.expander("Per-position shear detail"):
                shear_df = pd.DataFrame(sr)
                shear_df.insert(0, "Turbine", [f"T{i+1}" for i in range(len(sr))])
                st.dataframe(shear_df, hide_index=True, width="stretch")

    st.divider()

    # ------------------------------------------------------------------ STEP 6 --
    st.header("6. Hub Height & Weighting")
    if st.session_state.wra_shear_results is None:
        st.info("Compute shear above first.")
    elif not era5_maps or not cfsr_maps:
        st.info("Load both ERA5 and CFSR maps to blend a weighted result.")
    else:
        hc1, hc2 = st.columns(2)
        with hc1:
            wra_hub_height = st.number_input("Hub height (m)", min_value=1.0, value=100.0,
                                              step=1.0, key="wra_hub_height")
        with hc2:
            wra_era5_weight_pct = st.slider("ERA5 weight (%)", 0, 100, 67, key="wra_era5_weight_pct")
            st.caption(f"CFSR weight: {100 - wra_era5_weight_pct}%")

        if st.button("Compute Hub Height Wind Speed", type="primary"):
            layout_coords = tuple((g.y, g.x) for g in st.session_state.wra_layout.geometry)
            st.session_state.wra_hh_results = compute_hub_height_results(
                era5_maps, cfsr_maps, layout_coords, st.session_state.wra_shear_results,
                wra_hub_height, wra_era5_weight_pct / 100)
            # Any previous calibration was computed against the old hub-height/weights - clear it
            st.session_state.wra_cal_factors = None
            st.session_state.wra_calibrated_ws = None
            st.session_state.wra_cal_influence = None

        if st.session_state.wra_hh_results:
            hhr = st.session_state.wra_hh_results
            era5_vals = [r["era5_hh_ws"] for r in hhr if not np.isnan(r["era5_hh_ws"])]
            cfsr_vals = [r["cfsr_hh_ws"] for r in hhr if not np.isnan(r["cfsr_hh_ws"])]
            weighted_vals = [r["weighted_hh_ws"] for r in hhr if not np.isnan(r["weighted_hh_ws"])]

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Mean ERA5 WS", f"{np.mean(era5_vals):.2f} m/s" if era5_vals else "n/a")
            with m2:
                st.metric("Mean CFSR WS", f"{np.mean(cfsr_vals):.2f} m/s" if cfsr_vals else "n/a")
            with m3:
                st.metric("Average Modelled Wind Speed at the Windfarm",
                          f"{np.mean(weighted_vals):.2f} m/s" if weighted_vals else "n/a")

            fig = build_results_map_fig(st.session_state.wra_boundary, st.session_state.wra_layout,
                                         [r["weighted_hh_ws"] for r in hhr], "Weighted WS (m/s)")
            st.plotly_chart(fig, width="stretch", key="wra_hh_map_chart")

    st.divider()

    # ------------------------------------------------------------------ STEP 7 --
    st.header("7. Calibration")
    st.caption("Adjusts the modelled result toward long-term-corrected measurements at known "
               "locations. For each point, the model's wind speed is computed directly at that "
               "point's own location and height, compared to the measured value to get a "
               "calibration factor (CF), applied either as one site-wide average, a "
               "distance-weighted blend (nearer points count more), or via kriging.")
    with st.expander("Why three methods, and which should I use?"):
        st.markdown(
            "With a single calibration point, all three methods give the same result. With "
            "multiple points, the literature is fairly consistent: a plain average of the "
            "nearest predictor is usually beaten by distance-weighting several predictors "
            "(Bechmann et al., *Wind Energy Science*, 2020, validated across 185 met masts at "
            "40 sites), and several recent comparisons of spatial interpolation methods for "
            "wind speed - including a 2025 offshore-buoy study over the Mediterranean and a "
            "2025 station-network study - found ordinary kriging consistently gave the lowest "
            "prediction error among the methods tested, including inverse-distance weighting. "
            "\n\nThe reason is that kriging accounts for *redundancy*: two calibration points "
            "sitting close together carry mostly the same information, but plain distance "
            "weighting still counts them as two independent votes, while kriging correctly "
            "down-weights that overlap. This matters most once you have 3 or more points, "
            "especially if any are clustered.\n\n"
            "Rather than picking a method by assumption, use the **cross-validation table** "
            "below (available once you have 2+ points) - it leaves each point out in turn, "
            "predicts it from the rest, and reports the actual error for each method on "
            "*your* data. That is the same leave-one-out approach the literature itself uses "
            "to compare calibration methods, and it will tell you directly which method is "
            "performing best for this specific site rather than in general.\n\n"
            "All three methods also account for each point's own data quality, not just its "
            "location - see the '?' next to **Point quality** below for how a short "
            "measurement record gets down-weighted.")

    if st.session_state.wra_hh_results is None:
        st.info("Compute Hub Height Wind Speed above first.")
    else:
        st.subheader("7a. Calibration points")
        cpc1, cpc2, cpc3, cpc4, cpc5 = st.columns(5)
        with cpc1:
            cp_name = st.text_input("Name", value=f"M{len(st.session_state.wra_cal_points)+1}",
                                     key="wra_cp_name")
        with cpc2:
            cp_lat = st.number_input("Latitude", format="%.5f", key="wra_cp_lat")
        with cpc3:
            cp_lon = st.number_input("Longitude", format="%.5f", key="wra_cp_lon")
        with cpc4:
            cp_h = st.number_input("Height (m)", min_value=1.0, value=100.0, key="wra_cp_h")
        with cpc5:
            cp_ws = st.number_input("Long-term WS (m/s)", min_value=0.0, key="wra_cp_ws")

        CAL_QUALITY_HELP = (
            "**How calibration point quality is weighted**\n\n"
            "By default every calibration point counts equally. Flagging a point here means "
            "it counts *less* wherever it's used - Site Average CF, Distance Weighted CF, and "
            "Kriging's per-point uncertainty term - on top of (not instead of) its distance/"
            "redundancy standing.\n\n"
            "**Measurement Duration -> uncertainty.** Rather than an assumed curve shape, "
            "duration is converted to an actual estimated uncertainty percentage anchored to "
            "real measured values: Abascal Mendez et al. (2026, *Inventions* journal), "
            "analysing 30 real meteorological masts worldwide with up to 27 months of "
            "concurrent data, measured this exact quantity (dispersion-based MCP uncertainty) "
            "directly and found it falls in an approximately linear relationship with "
            "duration across 3-12 months - about 2.1% at 3 months down to about 0.4% at 12 "
            "months (their Linear Regression model, all-terrain average). That percentage is "
            "then converted to a confidence weight via **inverse-variance weighting** - the "
            "statistically standard way to combine estimates of differing precision (the same "
            "principle behind meta-analysis pooling) - rather than a linear or sqrt scale. "
            "Because this squares the ratio (variance, not standard deviation), a short record "
            "is penalized considerably harder than a simpler linear scale would suggest: a "
            "5-month record works out to roughly 5% confidence relative to a 12+ month one, "
            "not roughly 40%.\n\n"
            "**Correlation R² (optional).** If you have the R² from your own MCP regression "
            "for a point, entering it applies an additional adjustment using the standard "
            "regression-theory relationship (unexplained variance = 1 - R²) relative to a "
            "reference of R²=0.95 - a reasonable middle-of-the-range choice for this tool, not "
            "a value taken from the cited study. A stronger correlation raises confidence "
            "beyond what duration alone implies; a weaker one lowers it further.\n\n"
            "**Manual weight override** always takes full precedence over both of the above "
            "for that point - use it if you have your own judgement or your own independently "
            "computed uncertainty that you trust more than this formula.\n\n"
            "Caveats worth knowing: the cited study's numbers are for a Linear Regression MCP "
            "model averaged across all terrain types (not stratified by your specific terrain, "
            "and not the exact TLS method this tool's Long-Term Correction mode defaults to, "
            "though the study found the two comparable); below 3 months the relationship is "
            "extrapolated rather than directly measured; and this is one study, not a "
            "consensus figure. Use the leave-one-out cross-validation table below to check "
            "whether this weighting actually improves accuracy on *your* points.")

        st.markdown("**Point quality (optional)**", help=CAL_QUALITY_HELP)
        cqc1, cqc2 = st.columns(2)
        with cqc1:
            cp_months = st.number_input(
                "Measurement Duration (months)", min_value=0, value=0, step=1, key="wra_cp_months",
                help="Leave at 0 if you don't want to flag this point - treated as full-"
                     "confidence, same as before this feature existed. See the '?' above for "
                     "how this is converted to a weight.")
        with cqc2:
            cp_r2 = st.number_input(
                "Correlation R\u00b2 (optional)", min_value=0.0, max_value=0.999, value=0.0,
                step=0.01, key="wra_cp_r2",
                help="Leave at 0 to skip this adjustment. From your own MCP regression for "
                     "this point, if you have it - see the '?' above for how it's used.")

        cp_override = st.checkbox("Override with a manual weight instead", key="wra_cp_override")
        cp_manual_weight = None
        if cp_override:
            cp_manual_weight = st.number_input(
                "Manual confidence weight (0-1)", min_value=0.0, max_value=1.0, value=1.0,
                step=0.05, key="wra_cp_manual_weight",
                help="Takes full precedence over Measurement Duration and R\u00b2 for this point.")

        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("Add Point"):
                st.session_state.wra_cal_points.append(
                    {"name": cp_name, "lat": cp_lat, "lon": cp_lon, "h": cp_h, "ws": cp_ws,
                     "months": cp_months, "r_squared": cp_r2 if cp_r2 > 0 else None,
                     "manual_weight": cp_manual_weight})
        with bc2:
            if st.button("Clear Points"):
                st.session_state.wra_cal_points = []

        st.caption("Or import a CSV of calibration points (any column names - you map them below).")
        cal_csv = st.file_uploader("Calibration CSV", type=["csv"], key=f"wra_cal_csv_{_gen}")
        if cal_csv is not None:
            cal_raw = pd.read_csv(cal_csv)
            cal_cols = list(cal_raw.columns)

            def _guess(cols, keywords, default=0):
                for kw in keywords:
                    for i, c in enumerate(cols):
                        if kw.lower() in c.lower():
                            return i
                return default

            NO_MONTHS_COL = "(not provided - assume good quality)"
            NO_R2_COL = "(not provided - skip this adjustment)"
            NO_WEIGHT_COL = "(not provided - use Measurement Duration instead)"
            cal_cols_opt = [NO_MONTHS_COL] + cal_cols
            cal_cols_r2_opt = [NO_R2_COL] + cal_cols
            cal_cols_wt_opt = [NO_WEIGHT_COL] + cal_cols

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            with mc1:
                col_name = st.selectbox("Name col", cal_cols,
                                         index=_guess(cal_cols, ["name", "site"]), key="wra_csv_name")
            with mc2:
                col_lat = st.selectbox("Lat col", cal_cols,
                                        index=_guess(cal_cols, ["lat"]), key="wra_csv_lat")
            with mc3:
                col_lon = st.selectbox("Lon col", cal_cols,
                                        index=_guess(cal_cols, ["lon"]), key="wra_csv_lon")
            with mc4:
                col_h = st.selectbox("Height col", cal_cols,
                                      index=_guess(cal_cols, ["height", "h "]), key="wra_csv_h")
            with mc5:
                col_ws = st.selectbox("WS col", cal_cols,
                                       index=_guess(cal_cols, ["wind speed", "ws"]), key="wra_csv_ws")

            st.markdown("**Point quality columns (all optional)**", help=CAL_QUALITY_HELP)
            mq1, mq2, mq3 = st.columns(3)
            with mq1:
                _months_guess = _guess(cal_cols, ["duration", "month", "record length",
                                                   "record_length"], default=-1)
                col_months = st.selectbox(
                    "Measurement Duration col", cal_cols_opt,
                    index=(_months_guess + 1) if _months_guess >= 0 else 0, key="wra_csv_months")
            with mq2:
                _r2_guess = _guess(cal_cols, ["r2", "r-squared", "r squared", "correlation"],
                                    default=-1)
                col_r2 = st.selectbox(
                    "Correlation R\u00b2 col", cal_cols_r2_opt,
                    index=(_r2_guess + 1) if _r2_guess >= 0 else 0, key="wra_csv_r2")
            with mq3:
                _weight_guess = _guess(cal_cols, ["manual weight", "weight override", "weight"],
                                        default=-1)
                col_weight = st.selectbox(
                    "Manual weight col", cal_cols_wt_opt,
                    index=(_weight_guess + 1) if _weight_guess >= 0 else 0, key="wra_csv_weight")

            if st.button("Import CSV"):
                imported = []
                for _, row in cal_raw.iterrows():
                    _months = 0
                    if col_months != NO_MONTHS_COL:
                        try:
                            _months = float(row[col_months])
                        except (TypeError, ValueError):
                            _months = 0
                    _r2 = None
                    if col_r2 != NO_R2_COL:
                        try:
                            _r2 = float(row[col_r2])
                        except (TypeError, ValueError):
                            _r2 = None
                    _manual_weight = None
                    if col_weight != NO_WEIGHT_COL:
                        try:
                            _manual_weight = float(row[col_weight])
                        except (TypeError, ValueError):
                            _manual_weight = None
                    imported.append({"name": str(row[col_name]), "lat": float(row[col_lat]),
                                      "lon": float(row[col_lon]), "h": float(row[col_h]),
                                      "ws": float(row[col_ws]), "months": _months,
                                      "r_squared": _r2, "manual_weight": _manual_weight})
                st.session_state.wra_cal_points.extend(imported)

        if st.session_state.wra_cal_points:
            _cal_display = pd.DataFrame(st.session_state.wra_cal_points)
            for _col, _default in [("months", 0), ("r_squared", None), ("manual_weight", None)]:
                if _col not in _cal_display.columns:
                    _cal_display[_col] = _default
            _cal_display["confidence weight"] = _cal_display.apply(resolve_confidence_weight,
                                                                     axis=1).round(3)
            st.dataframe(_cal_display, hide_index=True, width="stretch")
            if (_cal_display["confidence weight"] < 1.0).any():
                st.caption("Points below 1.0 have a flagged Measurement Duration (or a manual "
                           "override) and are being down-weighted in the calibration accordingly "
                           "- see 7b.")

        if len(st.session_state.wra_cal_points) >= 1:
            st.subheader("7b. Run calibration")
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                wra_cal_method = st.radio(
                    "Calibration method",
                    ["Site Average CF", "Distance Weighted CF", "Kriging (Ordinary)"],
                    key="wra_cal_method",
                    help="Kriging accounts for redundancy between calibration points that "
                         "sit close together (they don't each get full independent weight "
                         "the way Distance Weighted CF would give them) - generally the more "
                         "statistically sound choice once you have 3+ points. See the "
                         "cross-validation comparison below to check which fits your data best.")
            with mcol2:
                wra_decay_km = st.number_input(
                    "Decay / range length (km)", min_value=1.0, value=250.0, step=10.0,
                    key="wra_decay_km", disabled=(wra_cal_method == "Site Average CF"))

            wra_clip_negative = True
            if wra_cal_method == "Kriging (Ordinary)":
                wra_clip_negative = st.checkbox(
                    "Prevent negative kriging weights (recommended)", value=True,
                    key="wra_clip_negative_weights",
                    help="Ordinary kriging can mathematically produce a negative weight for a "
                         "point - most often with few calibration points and/or one flagged "
                         "with a much shorter record than the others. Testing against randomized "
                         "small calibration sets in that situation found this happens on nearly "
                         "every one of them, with cross-validated error 5-8x worse than clipping "
                         "negative weights to zero and renormalizing the rest. Turn this off only "
                         "if you specifically want pure, unclipped kriging.")

            # Factors computed eagerly (cheap, cached) so a cross-validation comparison can be
            # shown before the user commits to a method.
            w_era5 = st.session_state.get("wra_era5_weight_pct", 67) / 100
            preview_factors = compute_calibration_factors(
                era5_maps, cfsr_maps, st.session_state.wra_cal_points, w_era5)

            if len(preview_factors) >= 2:
                with st.expander("Cross-validation: which method fits your points best?",
                                  expanded=True):
                    st.caption(
                        "Leave-one-out: each calibration point is predicted using only the "
                        "OTHER points, then compared to its actual measurement. Lower RMSE/MAE "
                        "means that method fits your specific points better - this is the "
                        "standard way the wind resource literature evaluates spatial "
                        "calibration methods rather than assuming one is universally best.")
                    loocv = loocv_calibration_errors(
                        preview_factors,
                        ["Site Average CF", "Distance Weighted CF", "Kriging (Ordinary)"],
                        decay_km=wra_decay_km, clip_negative_weights=wra_clip_negative)
                    loocv_df = pd.DataFrame([
                        {"Method": m, "RMSE (m/s)": round(r["rmse"], 3),
                         "MAE (m/s)": round(r["mae"], 3), "Points used": r["n"]}
                        for m, r in loocv.items()
                    ])
                    st.dataframe(loocv_df, hide_index=True, width="stretch")
            elif len(preview_factors) == 1:
                st.caption("Add at least one more calibration point to enable a "
                           "cross-validation comparison between methods.")

            if st.button("Start Calibration", type="primary"):
                if not preview_factors:
                    st.error("No valid calibration factors - check the points fall within the "
                             "wind map coverage.")
                else:
                    layout_coords = tuple((g.y, g.x) for g in st.session_state.wra_layout.geometry)
                    weighted_list = [r["weighted_hh_ws"] for r in st.session_state.wra_hh_results]
                    calibrated, influence = apply_calibration(
                        weighted_list, layout_coords, preview_factors, wra_cal_method, wra_decay_km,
                        clip_negative_weights=wra_clip_negative)
                    st.session_state.wra_cal_factors = preview_factors
                    st.session_state.wra_calibrated_ws = calibrated
                    st.session_state.wra_cal_influence = influence

        if st.session_state.wra_cal_factors:
            st.write("**Calibration factors:**")
            st.dataframe(pd.DataFrame(st.session_state.wra_cal_factors), hide_index=True,
                         width="stretch")

            avg_cf = np.mean([f["cf"] for f in st.session_state.wra_cal_factors])
            cal_vals = [c for c in st.session_state.wra_calibrated_ws if not np.isnan(c)]
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Average CF", f"{avg_cf:.3f}")
            with m2:
                st.metric("Calibrated Average Wind Speed at the Windfarm",
                          f"{np.mean(cal_vals):.2f} m/s" if cal_vals else "n/a")

            st.write("**Average influence of each calibration point:**")
            infl_df = pd.DataFrame([
                {"Point": f["name"], "Influence": f"{inf*100:.1f}%"}
                for f, inf in zip(st.session_state.wra_cal_factors, st.session_state.wra_cal_influence)
            ])
            st.dataframe(infl_df, hide_index=True, width="stretch")

            fig = build_results_map_fig(st.session_state.wra_boundary, st.session_state.wra_layout,
                                         st.session_state.wra_calibrated_ws, "Calibrated WS (m/s)",
                                         cal_points=st.session_state.wra_cal_points)
            st.plotly_chart(fig, width="stretch", key="wra_cal_map_chart")

    st.divider()

    # ------------------------------------------------------------------ STEP 8 --
    st.header("8. Download Plots and Table")
    if st.session_state.wra_hh_results is None:
        st.info("Compute Hub Height Wind Speed above first.")
    else:
        wra_file_basename = st.text_input(
            "File name (e.g. project or site name)", value="project",
            key=f"wra_file_basename_{_gen}",
            help="The download will be saved as '<this>_prelim_wra.zip'.")
        _safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", wra_file_basename.strip()) or "project"
        _download_filename = f"{_safe_name}_prelim_wra.zip"

        if st.button("Prepare download package"):
            with st.spinner("Building download package..."):
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    excel_bytes = build_wra_excel(
                        st.session_state.wra_boundary, st.session_state.wra_layout,
                        st.session_state.wra_hh_results,
                        calibrated_ws=st.session_state.wra_calibrated_ws,
                        cal_points=st.session_state.wra_cal_points or None,
                        cal_factors=st.session_state.wra_cal_factors)
                    zf.writestr("wra_results.xlsx", excel_bytes)

                    weighted_fig = render_wra_static_map(
                        st.session_state.wra_boundary, st.session_state.wra_layout,
                        [r["weighted_hh_ws"] for r in st.session_state.wra_hh_results],
                        "Weighted WS (m/s)", "Weighted Hub-Height Wind Speed")
                    zf.writestr("weighted_hub_height_map.png", fig_to_png_bytes(weighted_fig))

                    if st.session_state.wra_calibrated_ws is not None:
                        cal_fig = render_wra_static_map(
                            st.session_state.wra_boundary, st.session_state.wra_layout,
                            st.session_state.wra_calibrated_ws, "Calibrated WS (m/s)",
                            "Calibrated Wind Speed", cal_points=st.session_state.wra_cal_points)
                        zf.writestr("calibrated_map.png", fig_to_png_bytes(cal_fig))

                buf.seek(0)
                st.session_state["wra_zip"] = buf.read()
                st.session_state["wra_zip_filename"] = _download_filename

        if "wra_zip" in st.session_state:
            st.download_button("Download Plots and Table (ZIP)", st.session_state["wra_zip"],
                                file_name=st.session_state.get("wra_zip_filename", _download_filename),
                                mime="application/zip")
            st.caption("Reflects the settings selected at the moment you clicked 'Prepare' - "
                       "click it again after changing anything above.")

    st.divider()
    if st.button("Reset Wind Resource Assessment"):
        _next_gen = st.session_state.get("wra_uploader_gen", 0) + 1
        for _k in list(st.session_state.keys()):
            if _k.startswith("wra_"):
                del st.session_state[_k]
        # Bump AFTER wiping, so every file_uploader above gets a brand-new key and
        # genuinely resets - deleting the session_state value alone isn't enough, since
        # the browser-side widget still holds its selected file(s) and would otherwise
        # silently re-populate everything right back on the very next rerun.
        st.session_state.wra_uploader_gen = _next_gen
        st.rerun()

elif mode == "Wind Measurement Tracker":
    st.title("Wind Measurement Tracker")
    st.caption("Upload monthly measurement files as they arrive, and track wind speed, "
               "direction, turbulence intensity, and data availability over time.")

    if "track_uploader_gen" not in st.session_state:
        st.session_state.track_uploader_gen = 0
    _tgen = st.session_state.track_uploader_gen

    if st.button("Reset Wind Measurement Tracker"):
        for _k in list(st.session_state.keys()):
            if _k.startswith("track_"):
                del st.session_state[_k]
        st.session_state.track_uploader_gen = _tgen + 1
        st.rerun()

    st.divider()
    track_workflow = st.radio(
        "Start fresh, or continue tracking from a previous package?",
        ["Start fresh", "Continue tracking (load a previous package)"],
        key="track_workflow_mode", horizontal=True,
        help="A transferable package (downloaded from Step 5 of a previous session) carries "
             "forward everything accumulated so far - the combined measurement history, height "
             "mapping, and analysis settings - so you don't need to re-upload old raw files or "
             "redo column selections every time a new month of data arrives.")

    if track_workflow == "Continue tracking (load a previous package)":
        track_package_file = st.file_uploader(
            "Transferable package (.zip) from a previous session", type=["zip"],
            key=f"track_package_file_{_tgen}")
        if track_package_file is not None and st.session_state.get("track_package_df") is None:
            try:
                pkg_df, pkg_config = load_tracker_package(track_package_file.getvalue())
                st.session_state.track_package_df = pkg_df
                st.session_state.track_package_config = pkg_config
                st.session_state.track_package_cutoff = pkg_df.index.max()
                hm_list = pkg_config.get("height_map", [])
                if hm_list:
                    st.session_state.track_n_heights = len(hm_list)
                if "meas_utc_offset" in pkg_config:
                    st.session_state.track_meas_utc_offset = pkg_config["meas_utc_offset"]
                if "interest_height" in pkg_config:
                    st.session_state.track_interest_height = pkg_config["interest_height"]
                st.success(f"Loaded package: {len(pkg_df)} rows, {pkg_df.index.min()} to "
                           f"{pkg_df.index.max()}. Upload the new month's file(s) below to "
                           f"continue - or leave it here to just review this package's own "
                           f"results without adding anything new yet.")
            except Exception as e:
                st.error(f"Could not read this package: {e}")
        elif st.session_state.get("track_package_df") is not None:
            _pkg_df = st.session_state.track_package_df
            st.info(f"Package already loaded: {len(_pkg_df)} rows, {_pkg_df.index.min()} to "
                    f"{_pkg_df.index.max()}. Upload additional new month(s) below, or use Reset "
                    f"to load a different package instead.")
    st.divider()

    _is_continuing = (track_workflow.startswith("Continue")
                       and st.session_state.get("track_package_df") is not None)
    _pkg_cutoff = st.session_state.get("track_package_cutoff")

    # ------------------------------------------------------------------ STEP 1 --
    st.header("1. Upload New Measurement Data" if _is_continuing else "1. Upload Measurement Data")
    st.caption("Upload every file you have - from the first month through the most recent. "
               "Accepts .csv, Campbell Scientific TOA5 .dat/.sta, a second .dat/.sta layout "
               "seen from floating LiDAR buoy systems (a header row starting with "
               "'timestamp'), and NetCDF .nc. Files are auto-detected by format, "
               "concatenated, and sorted by time; overlapping timestamps are de-duplicated.")
    track_files = st.file_uploader(
        "Measurement files", type=["csv", "dat", "sta", "nc"], accept_multiple_files=True,
        key=f"track_files_{_tgen}")

    if track_files or st.session_state.get("track_package_df") is not None:
        csv_files, toa5_files, generic_dat_files, nc_files, rejected = [], [], [], [], []
        for f in (track_files or []):
            ext = f.name.lower().rsplit(".", 1)[-1]
            if ext == "csv":
                csv_files.append(f)
            elif ext in ("dat", "sta"):
                fbytes = f.getvalue()
                if sniff_toa5(fbytes):
                    toa5_files.append(f)
                elif sniff_generic_dat(fbytes):
                    generic_dat_files.append(f)
                else:
                    rejected.append(f.name)
            elif ext == "nc":
                nc_files.append(f)

        if rejected:
            st.error(f"{len(rejected)} file(s) didn't match either supported .dat/.sta layout "
                     f"(Campbell Scientific TOA5, or a header row containing 'timestamp') and "
                     f"were skipped: {', '.join(rejected[:5])}"
                     f"{' ...' if len(rejected) > 5 else ''}. Show me what one of these actually "
                     f"looks like and I'll add support for that format too.")

        parsed_frames = []
        if st.session_state.get("track_package_df") is not None:
            parsed_frames.append(st.session_state.track_package_df)

        # --- TOA5 files: parse directly, timestamp is unambiguous, no mapping needed ---
        if toa5_files:
            toa5_skipped_total = 0
            toa5_rows_total = 0
            for f in toa5_files:
                try:
                    tdf, _units, skipped = parse_toa5(f.getvalue())
                    tdf["Timestamp"] = pd.to_datetime(tdf["TIMESTAMP"], errors="coerce")
                    tdf = (tdf.drop(columns=["TIMESTAMP"]).dropna(subset=["Timestamp"])
                              .set_index("Timestamp"))
                    parsed_frames.append(tdf)
                    toa5_skipped_total += skipped
                    toa5_rows_total += len(tdf)
                except Exception as e:
                    st.error(f"Could not parse {f.name}: {e}")
            st.success(f"Parsed {len(toa5_files)} TOA5 file(s) - {toa5_rows_total} rows.")
            if toa5_skipped_total:
                st.warning(f"{toa5_skipped_total} malformed row(s) skipped across TOA5 files "
                           f"(wrong number of fields for that file's header).")

        # --- Generic .dat/.sta files (e.g. floating LiDAR buoy exports): one shared
        # dayfirst/invalid-codes setting, applied to every file in this group ---
        if generic_dat_files:
            st.subheader("1a. Generic .dat/.sta settings (applied to every file in this group)")
            try:
                _preview_df, _detected_dayfirst, _ = parse_generic_dat(generic_dat_files[0].getvalue())
                st.write(f"Preview of {generic_dat_files[0].name}:")
                st.dataframe(_preview_df.head(5), width="stretch")
            except Exception as e:
                _detected_dayfirst = True
                st.error(f"Could not preview {generic_dat_files[0].name}: {e}")
            gc1, gc2 = st.columns(2)
            with gc1:
                track_gd_dayfirst = st.checkbox(
                    "Date format is day-first (DD/MM/YYYY)", value=_detected_dayfirst,
                    key="track_gd_dayfirst",
                    help="Pre-filled from the units-row date-format descriptor, if this file "
                         "has one (e.g. 'DD-MM-YYYY hh:mm') - check it's right, since it's "
                         "inferred, not guaranteed.")
            with gc2:
                track_gd_invalid_text = st.text_input(
                    "Invalid/missing value codes (comma-separated)", value="9999, 9998, 999, -999",
                    key="track_gd_invalid_text",
                    help="9998 is included by default since it's a real sentinel value seen in "
                         "this kind of file (e.g. a LiDAR range gate with no valid return) - "
                         "edit if your files use something different.")
            track_gd_invalid_codes = parse_invalid_codes(track_gd_invalid_text)

            gd_rows_total, gd_skipped_total = 0, 0
            for f in generic_dat_files:
                try:
                    gdf, _df, skipped = parse_generic_dat(f.getvalue())
                    gdf["Timestamp"] = pd.to_datetime(gdf["timestamp"], dayfirst=track_gd_dayfirst,
                                                       errors="coerce")
                    gdf = (gdf.drop(columns=["timestamp"]).dropna(subset=["Timestamp"])
                               .set_index("Timestamp"))
                    for c in gdf.columns:
                        gdf[c] = pd.to_numeric(gdf[c], errors="coerce")
                    if track_gd_invalid_codes:
                        gdf = gdf.replace(track_gd_invalid_codes, np.nan)
                    parsed_frames.append(gdf)
                    gd_rows_total += len(gdf)
                    gd_skipped_total += skipped
                except Exception as e:
                    st.error(f"Could not parse {f.name}: {e}")
            st.success(f"Parsed {len(generic_dat_files)} file(s) - {gd_rows_total} rows.")
            if gd_skipped_total:
                st.warning(f"{gd_skipped_total} malformed row(s) skipped across these files "
                           f"(wrong number of fields for that file's header).")

        # --- CSV files: one shared column mapping, applied to every CSV uploaded ---
        if csv_files:
            st.subheader("1b. CSV column mapping (applied to every CSV uploaded)")
            first_csv_raw = read_raw_csv(csv_files[0].getvalue())
            st.write(f"Preview of {csv_files[0].name}:")
            st.dataframe(first_csv_raw.head(5), width="stretch")
            csv_cols = list(first_csv_raw.columns)
            cc1, cc2 = st.columns(2)
            with cc1:
                track_ts_col = st.selectbox("Timestamp column", csv_cols, key="track_ts_col")
                track_dayfirst = st.checkbox("Date format is day-first (DD/MM/YYYY)", value=True,
                                              key="track_dayfirst")
            with cc2:
                track_invalid_text = st.text_input(
                    "Invalid/missing value codes (comma-separated)", value="9999, 999, -999",
                    key="track_invalid_text")
            track_invalid_codes = parse_invalid_codes(track_invalid_text)

            csv_rows_total = 0
            for f in csv_files:
                try:
                    raw = read_raw_csv(f.getvalue())
                    cdf = build_clean_df(raw, track_ts_col, track_dayfirst,
                                          tuple(track_invalid_codes))
                    parsed_frames.append(cdf)
                    csv_rows_total += len(cdf)
                except Exception as e:
                    st.error(f"Could not parse {f.name}: {e}")
            st.success(f"Parsed {len(csv_files)} CSV file(s) - {csv_rows_total} rows.")

        # --- NetCDF files: one shared variable mapping, applied to every .nc uploaded ---
        if nc_files:
            st.subheader("1c. NetCDF variable mapping (applied to every .nc file uploaded)")
            try:
                var_info = list_netcdf_variables(nc_files[0].getvalue())
            except Exception as e:
                var_info = {}
                st.error(f"Could not read {nc_files[0].name}: {e}")
            if var_info:
                var_names = list(var_info.keys())
                st.write(f"Variables found in {nc_files[0].name}:")
                st.dataframe(pd.DataFrame([
                    {"Variable": k, "Dims": str(v["dims"]), "Units": v["units"],
                     "Description": v["long_name"]} for k, v in var_info.items()
                ]), hide_index=True, width="stretch")
                time_guess = next((n for n in var_names if "time" in n.lower()), var_names[0])
                nc1, nc2 = st.columns(2)
                with nc1:
                    track_nc_time_var = st.selectbox(
                        "Time variable", var_names, index=var_names.index(time_guess),
                        key="track_nc_time_var")
                with nc2:
                    track_nc_value_vars = st.multiselect(
                        "Value variables to import (WS, WD, TI, etc.)",
                        [n for n in var_names if n != track_nc_time_var],
                        key="track_nc_value_vars",
                        help="Only 1-D (time-indexed) variables are supported - a single-height "
                             "dataset. A multi-dimensional, height-resolved NetCDF would need a "
                             "different reader.")
                if track_nc_value_vars:
                    nc_rows_total = 0
                    for f in nc_files:
                        try:
                            ndf = read_netcdf_to_df(f.getvalue(), track_nc_time_var,
                                                     tuple(track_nc_value_vars))
                            ndf = ndf.set_index("Timestamp")
                            parsed_frames.append(ndf)
                            nc_rows_total += len(ndf)
                        except Exception as e:
                            st.error(f"Could not read {f.name}: {e}")
                    st.success(f"Parsed {len(nc_files)} NetCDF file(s) - {nc_rows_total} rows.")

        # --- Concatenate everything into one combined, de-duplicated, sorted dataset ---
        if parsed_frames:
            combined = pd.concat(parsed_frames, axis=0, join="outer", sort=False)
            n_before_dedup = len(combined)
            combined = combined[~combined.index.duplicated(keep="first")].sort_index()
            for c in combined.columns:
                combined[c] = pd.to_numeric(combined[c], errors="coerce")
            st.session_state.track_combined_df = combined
            if n_before_dedup != len(combined):
                st.caption(f"Removed {n_before_dedup - len(combined)} duplicate timestamp(s) "
                           f"found across files (kept the first occurrence of each).")

    if st.session_state.get("track_combined_df") is not None:
        combined = st.session_state.track_combined_df
        n_months = combined.index.to_series().dt.to_period("M").nunique()
        st.info(f"Combined dataset: {len(combined)} rows, "
                f"{combined.index.min()} to {combined.index.max()} "
                f"({n_months} distinct calendar month{'s' if n_months != 1 else ''}).")

        st.divider()

        # -------------------------------------------------------------- STEP 2 --
        st.header("2. Map heights")
        _hm_table_placeholder = st.empty() if _is_continuing else None
        if _is_continuing:
            st.caption("Restored from your loaded package. Expand below only if something "
                       "needs changing - e.g. a new height was added, or a column got renamed.")
        else:
            st.caption("For each height, choose the wind speed column (required), and optionally "
                       "wind direction and turbulence intensity - either a direct TI column, or "
                       "computed from a wind-speed standard-deviation column alongside the mean "
                       "(the standard TOA5 convention: a '_Std' column next to '_Avg').")

        with st.expander("Height mapping" + (" (edit if needed)" if _is_continuing else ""),
                          expanded=not _is_continuing):
            n_heights = st.number_input("How many heights?", min_value=1, max_value=20, value=1,
                                         step=1, key="track_n_heights")
            all_cols = list(combined.columns)

            _pkg_config = st.session_state.get("track_package_config")
            _pkg_hm_list = _pkg_config.get("height_map", []) if _pkg_config else []
            _should_seed_hm = (bool(_pkg_hm_list) and
                                not st.session_state.get("track_package_hm_seeded"))

            track_height_map = []
            for i in range(int(n_heights)):
                if _should_seed_hm and i < len(_pkg_hm_list):
                    _pkg_hm = _pkg_hm_list[i]
                    st.session_state[f"track_h_{i}"] = _pkg_hm.get("height", 50.0 + i * 30)
                    if _pkg_hm.get("ws_col") in all_cols:
                        st.session_state[f"track_ws_{i}"] = _pkg_hm["ws_col"]
                    _wd_val = _pkg_hm.get("wd_col")
                    st.session_state[f"track_wd_{i}"] = _wd_val if _wd_val in all_cols else "(none)"
                    if _pkg_hm.get("std_col") in all_cols:
                        st.session_state[f"track_ti_mode_{i}"] = "Compute from Std/Avg"
                        st.session_state[f"track_std_{i}"] = _pkg_hm["std_col"]
                    elif _pkg_hm.get("ti_col") in all_cols:
                        st.session_state[f"track_ti_mode_{i}"] = "Direct TI column"
                        st.session_state[f"track_ti_col_{i}"] = _pkg_hm["ti_col"]
                    else:
                        st.session_state[f"track_ti_mode_{i}"] = "(none)"

                st.markdown(f"**Height {i + 1}**")
                r1 = st.columns([1, 2, 2])
                with r1[0]:
                    h = st.number_input("Height (m)", min_value=1.0, value=float(50 + i * 30),
                                         key=f"track_h_{i}")
                with r1[1]:
                    ws_c = st.selectbox("WS column", all_cols,
                                         index=_guess_col_index(all_cols, ["ws", "wind speed"]),
                                         key=f"track_ws_{i}")
                with r1[2]:
                    wd_c = st.selectbox("WD column (optional)", ["(none)"] + all_cols,
                                         index=1 + _guess_col_index(all_cols, ["wd", "direction"],
                                                                     default=-1),
                                         key=f"track_wd_{i}")
                r2 = st.columns([2, 2])
                with r2[0]:
                    ti_mode = st.selectbox("TI source", ["(none)", "Direct TI column",
                                                           "Compute from Std/Avg"],
                                            key=f"track_ti_mode_{i}")
                ti_c, std_c = None, None
                with r2[1]:
                    if ti_mode == "Direct TI column":
                        ti_c = st.selectbox("TI column", all_cols, key=f"track_ti_col_{i}")
                    elif ti_mode == "Compute from Std/Avg":
                        std_c = st.selectbox("WS Std Dev column", all_cols,
                                              index=_guess_col_index(all_cols, ["std"]),
                                              key=f"track_std_{i}")
                track_height_map.append({"height": h, "ws_col": ws_c,
                                          "wd_col": None if wd_c == "(none)" else wd_c,
                                          "ti_col": ti_c, "std_col": std_c})
                st.divider()

            if _should_seed_hm:
                st.session_state.track_package_hm_seeded = True

        if _hm_table_placeholder is not None:
            _hm_table_df = pd.DataFrame([
                {"Height (m)": f"{hm['height']:.0f}", "WS column": hm["ws_col"],
                 "WD column": hm["wd_col"] or "-",
                 "TI source": hm["ti_col"] or (f"computed from {hm['std_col']}"
                                                if hm["std_col"] else "-")}
                for hm in sorted_heights(track_height_map)
            ])
            _hm_table_placeholder.dataframe(_hm_table_df, hide_index=True, width="stretch")

        _sanity_cols = [{"ws_col": hm["ws_col"], "wd_col": hm["wd_col"],
                          "label": f"{hm['height']:.0f} m"} for hm in track_height_map]
        combined, _sanity_report = apply_physical_sanity_filter(combined, _sanity_cols)
        if _sanity_report:
            st.warning("Automatically treated as invalid, beyond whatever invalid-value codes "
                       "were specified above (a physically impossible reading - e.g. a wind "
                       f"speed over {MAX_PLAUSIBLE_WS:.0f} m/s - is almost always a sensor "
                       "fault or placeholder value, not a real one, even if it doesn't match "
                       "any code you entered):\n\n" + "\n".join(f"- {line}" for line in _sanity_report))

        plot_df = combined.copy()
        for hm in track_height_map:
            if hm["std_col"]:
                derived_col = f"_TI_derived_{hm['height']:.0f}m"
                ws_safe = plot_df[hm["ws_col"]].replace(0, np.nan)
                plot_df[derived_col] = plot_df[hm["std_col"]] / ws_safe * 100
                hm["ti_col"] = derived_col

        if _is_continuing and _pkg_cutoff is not None:
            st.divider()
            st.subheader("What's new this update")
            _before_df = plot_df[plot_df.index <= _pkg_cutoff]
            _new_df = plot_df[plot_df.index > _pkg_cutoff]
            if len(_new_df) == 0:
                st.caption("No new data added yet this session - upload the new month's "
                           "file(s) in Step 1 to see a before/after comparison here.")
            else:
                _new_months = _new_df.index.to_period("M").nunique()
                st.caption(f"{len(_new_df)} new hour(s) added, spanning "
                           f"{_new_df.index.min()} to {_new_df.index.max()} "
                           f"({_new_months} calendar month{'s' if _new_months != 1 else ''}).")
                _rows = []
                for hm in sorted_heights(track_height_map):
                    ws_c = hm["ws_col"]
                    _before_valid = _before_df[ws_c].notna().any() if len(_before_df) else False
                    _new_valid = _new_df[ws_c].notna().any() if len(_new_df) else False
                    _rows.append({
                        "Height (m)": f"{hm['height']:.0f}", "Metric": "Mean WS (m/s)",
                        "Before this update": (round(_before_df[ws_c].mean(), 2)
                                                if _before_valid else "-"),
                        "New month(s)": round(_new_df[ws_c].mean(), 2) if _new_valid else "-",
                        "Combined (after)": round(plot_df[ws_c].mean(), 2),
                    })
                    _rows.append({
                        "Height (m)": f"{hm['height']:.0f}", "Metric": "Data availability (%)",
                        "Before this update": (round(overall_availability(_before_df, ws_c), 1)
                                                if len(_before_df) else "-"),
                        "New month(s)": (round(overall_availability(_new_df, ws_c), 1)
                                          if len(_new_df) else "-"),
                        "Combined (after)": round(overall_availability(plot_df, ws_c), 1),
                    })
                st.dataframe(pd.DataFrame(_rows), hide_index=True, width="stretch")

        # -------------------------------------------------------------- STEP 3 --
        st.header("3. Monthly metrics")
        tabs = st.tabs(["Data Availability", "Monthly Mean WS", "TI vs Wind Speed", "Wind Rose"])

        with tabs[0]:
            st.subheader("Data availability by height and month")
            table = availability_table(plot_df, track_height_map)
            st.dataframe(table.style.format("{:.1f}%"), width="stretch")
            fig = plot_availability_bars(table)
            show_fig(fig, width=WIDTH_AVAILABILITY)

        with tabs[1]:
            st.subheader("Monthly mean wind speed")
            if _is_continuing and _pkg_cutoff is not None:
                st.caption("Amber bars are the month(s) just added this update.")
            for hm in sorted_heights(track_height_map):
                mm, incomplete, overall = monthly_mean_data(plot_df, hm["ws_col"])
                fig = render_monthly_fig(mm, incomplete, overall, f"{hm['height']:.0f} m",
                                          highlight_after=_pkg_cutoff if _is_continuing else None)
                show_fig(fig, width=WIDTH_MONTHLY)

        with tabs[2]:
            st.subheader("Monthly turbulence intensity vs wind speed")
            st.caption("Turbulence intensity is conventionally characterised as a function of "
                       "wind speed - typically higher and noisier at low speeds, decreasing and "
                       "flattening out at higher ones - rather than as a single monthly average, "
                       "which would hide that relationship. Both axes share the same scale "
                       "across every month for a fair comparison.")
            ti_heights = [hm for hm in sorted_heights(track_height_map) if hm["ti_col"]]
            if not ti_heights:
                st.info("No turbulence intensity source mapped for any height.")
            for hm in ti_heights:
                ti_png = render_monthly_ti_vs_ws_grid_png(plot_df, hm["ws_col"], hm["ti_col"],
                                                            f"{hm['height']:.0f} m")
                if ti_png is not None:
                    st.image(ti_png, width=WIDTH_ROSE_GRID)
                else:
                    st.caption(f"Not enough data at {hm['height']:.0f} m to build this.")

        with tabs[3]:
            st.subheader("Wind rose by month")
            st.caption("One rose per calendar month, on a shared scale, so directional shifts "
                       "across the tracked period are visible at a glance.")
            rose_heights = [hm for hm in sorted_heights(track_height_map) if hm["wd_col"]]
            if not rose_heights:
                st.info("No wind direction column mapped for any height.")
            else:
                rose_height_labels = [f"{hm['height']:.0f} m" for hm in rose_heights]
                rose_choice = st.selectbox("Height", rose_height_labels, key="track_rose_height")
                hm = rose_heights[rose_height_labels.index(rose_choice)]
                rose_png = render_monthly_rose_grid_png(plot_df[hm["ws_col"]], plot_df[hm["wd_col"]],
                                                          f"{hm['height']:.0f} m")
                if rose_png is not None:
                    st.image(rose_png, width=WIDTH_ROSE_GRID)
                else:
                    st.caption(f"Not enough concurrent WS/WD data at {hm['height']:.0f} m for a rose.")

        st.divider()

        # -------------------------------------------------------------- STEP 4 --
        st.header("4. Long-Term Analysis")
        st.caption("Upload a modelled wind dataset (Vortex or any hourly reanalysis - ERA5, "
                   "CFSR, MERRA-2) to see whether the running measured average is converging "
                   "toward the long-term estimate as more months of data are added. The "
                   "long-term estimate itself is computed once, fit against every concurrent "
                   "hour currently available - it's a fixed reference line, not something "
                   "recomputed at each month.")
        track_model_file = st.file_uploader(
            "Modelled wind dataset file (CSV or Vortex .txt)", type=["csv", "txt"],
            key=f"track_model_file_{_tgen}")

        if track_model_file is not None:
            track_model_bytes = track_model_file.getvalue()
            track_is_vortex = sniff_vortex_format(track_model_bytes)

            if track_is_vortex:
                (track_raw_model_df, track_vortex_height, track_vortex_tz,
                 _track_vlat, _track_vlon) = parse_vortex_txt(track_model_bytes)
                st.success(f"Detected a Vortex-format file - Hub-Height="
                           f"{track_vortex_height:.0f} m, Timezone=UTC{track_vortex_tz:+.1f}.")
                track_model_cols = [c for c in track_raw_model_df.columns if c != "Timestamp"]
                track_model_ts_col, track_model_dayfirst = "Timestamp", False
                tm1, tm2 = st.columns(2)
                with tm1:
                    _ws_idx = guess_column(track_model_cols, ["m/s", "wspd", "speed"])
                    track_model_ws_col = st.selectbox(
                        "Wind speed column", track_model_cols, index=_ws_idx,
                        key="track_model_ws_col_vortex")
                with tm2:
                    track_model_height = st.number_input(
                        "Height of the modelled dataset (m)", min_value=1.0,
                        value=track_vortex_height, step=1.0, key="track_model_height_vortex")
                track_model_tz_default = track_vortex_tz
            else:
                track_raw_model_df = read_raw_csv(track_model_bytes)
                st.dataframe(track_raw_model_df.head(5), width="stretch")
                track_model_cols = list(track_raw_model_df.columns)
                tm1, tm2 = st.columns(2)
                with tm1:
                    track_model_ts_col = st.selectbox("Timestamp column", track_model_cols,
                                                        key="track_model_ts_col")
                    track_model_dayfirst = st.checkbox(
                        "Date format is day-first (DD/MM/YYYY)", value=False,
                        key="track_model_dayfirst")
                    track_model_ws_col = st.selectbox("Wind speed column", track_model_cols,
                                                        key="track_model_ws_col")
                with tm2:
                    track_model_height = st.number_input(
                        "Height of the modelled dataset (m)", min_value=1.0,
                        value=detect_height_from_colname(track_model_ws_col, default=100.0),
                        step=1.0, key="track_model_height")
                track_model_tz_default = 0.0

            track_model_df = build_clean_df(track_raw_model_df, track_model_ts_col,
                                             track_model_dayfirst, ())

            tzc1, tzc2 = st.columns(2)
            with tzc1:
                track_meas_utc_offset = st.number_input(
                    "Measurement timezone offset from UTC (hours)", value=0.0, step=0.5,
                    key="track_meas_utc_offset")
            with tzc2:
                track_model_utc_offset = st.number_input(
                    "Modelled dataset timezone offset from UTC (hours)",
                    value=track_model_tz_default, step=0.5, key="track_model_utc_offset")

            track_interest_height = st.number_input(
                "Height of interest for the long-term result (m)", min_value=1.0,
                value=float(sorted_heights(track_height_map)[0]["height"]),
                key="track_interest_height")

            plot_df_utc = plot_df.copy()
            plot_df_utc.index = plot_df_utc.index - pd.Timedelta(hours=track_meas_utc_offset)
            track_model_df_utc = track_model_df.copy()
            track_model_df_utc.index = (track_model_df_utc.index -
                                         pd.Timedelta(hours=track_model_utc_offset))

            track_samples_per_hour = max(1, round(60 / detect_resolution_minutes(plot_df.index)))
            track_model_samples_per_hour = max(
                1, round(60 / detect_resolution_minutes(track_model_df.index)))
            track_model_ws_hourly = resample_to_hourly(
                track_model_df_utc[track_model_ws_col], track_model_samples_per_hour)

            track_meas_series_by_height = {}
            for hm in sorted_heights(track_height_map):
                hourly = resample_to_hourly(plot_df_utc[hm["ws_col"]], track_samples_per_hour)
                track_meas_series_by_height[hm["height"]] = (hourly, hm["ws_col"])

            track_shear_data = compute_shear_data(plot_df, track_height_map,
                                                   min_availability=80.0)
            if track_shear_data is None:
                st.warning("Fewer than 3 heights meet the 80% availability threshold, and no "
                           "measured height matches the modelled dataset's height closely - "
                           "can't compute a long-term result without shear to extrapolate. Map "
                           "more heights, or make sure one is close to the modelled dataset's "
                           "height.")
            else:
                st.success(f"Shear exponent (alpha) = {track_shear_data['alpha']:.3f}  |  "
                           f"heights used: {track_shear_data['heights_used']}")

                lt_corr_series, lt_corr_desc, _lt_corr_ref_h = get_measurement_at_target_height(
                    track_meas_series_by_height, track_shear_data, track_model_height)

                if lt_corr_series is None:
                    st.error(f"Could not establish a correlation height: {lt_corr_desc}")
                else:
                    lt_merged = merge_concurrent(lt_corr_series, track_model_ws_hourly)
                    if len(lt_merged) < 2:
                        st.error("No concurrent overlap between measurement and modelled "
                                 "dataset - check the timezone offsets and date ranges.")
                    else:
                        lt_result = long_term_correction(lt_merged, track_model_ws_hourly)
                        lt_ws_at_model_height = lt_result["tls"]["lt_mean"]
                        lt_ws_at_interest = (
                            lt_ws_at_model_height *
                            (track_interest_height / track_model_height) **
                            track_shear_data["alpha"])
                        st.info(f"Long-term wind speed at {track_interest_height:.0f} m, fit "
                                f"once using all {lt_result['n_concurrent']} concurrent hours "
                                f"currently available: **{lt_ws_at_interest:.2f} m/s**")

                        interest_series, interest_desc, _ = get_measurement_at_target_height(
                            track_meas_series_by_height, track_shear_data, track_interest_height)
                        if interest_series is not None:
                            cum_means = running_cumulative_mean_by_month(interest_series)
                            if len(cum_means) >= 1:
                                conv_fig = render_convergence_fig(
                                    cum_means, lt_ws_at_interest,
                                    f"{track_interest_height:.0f} m")
                                show_fig(conv_fig, width=WIDTH_MONTHLY)
                                st.caption(f"Measured series: {interest_desc}.")

        st.divider()

        # -------------------------------------------------------------- STEP 5 --
        st.header("5. Download")
        if st.button("Prepare Download", type="primary"):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                table = availability_table(plot_df, track_height_map)
                zf.writestr("01_data_availability.png",
                            fig_to_png_bytes(plot_availability_bars(table)))
                zf.writestr("01_data_availability.csv", table.to_csv())

                for hm in sorted_heights(track_height_map):
                    mm, incomplete, overall = monthly_mean_data(plot_df, hm["ws_col"])
                    fig = render_monthly_fig(mm, incomplete, overall, f"{hm['height']:.0f} m")
                    zf.writestr(f"02_monthly_ws_{hm['height']:.0f}m.png", fig_to_png_bytes(fig))

                for hm in sorted_heights(track_height_map):
                    if hm["ti_col"]:
                        ti_png = render_monthly_ti_vs_ws_grid_png(plot_df, hm["ws_col"],
                                                                    hm["ti_col"],
                                                                    f"{hm['height']:.0f} m")
                        if ti_png is not None:
                            zf.writestr(f"03_monthly_ti_vs_ws_{hm['height']:.0f}m.png", ti_png)

                for hm in sorted_heights(track_height_map):
                    if hm["wd_col"]:
                        rose_png = render_monthly_rose_grid_png(plot_df[hm["ws_col"]],
                                                                 plot_df[hm["wd_col"]],
                                                                 f"{hm['height']:.0f} m")
                        if rose_png is not None:
                            zf.writestr(f"04_wind_rose_by_month_{hm['height']:.0f}m.png", rose_png)

            buf.seek(0)
            st.session_state["track_zip"] = buf.read()

        if "track_zip" in st.session_state:
            st.download_button("Download Plots (ZIP)", st.session_state["track_zip"],
                                file_name="wind_measurement_tracker_plots.zip",
                                mime="application/zip")
            st.caption("Reflects the settings selected at the moment you clicked "
                       "'Prepare Download' - click it again after changing anything above.")

        st.divider()
        _pkg_verb = "Update" if _is_continuing else "Create"
        st.subheader(f"Transferable package ({'update it every time a new month arrives'
                                                if _is_continuing else
                                                'to continue tracking next month'})")
        if _is_continuing:
            st.caption("This carries forward everything so far, including the month(s) you "
                       "just added - download the updated package now, and load *this* one "
                       "back in next month rather than the one you started with, so each "
                       "month's package always reflects the full history to date.")
        else:
            st.caption("Unlike the plots ZIP above, this carries the combined measurement data "
                       "and your height-mapping/analysis settings forward - download it now, "
                       "and next month, load it back in under 'Continue tracking' at the top of "
                       "this page along with just the new month's raw file(s), rather than "
                       "re-uploading and re-mapping everything from scratch. You only need to "
                       "start fresh like this once, when setting up a new project.")
        if st.button(f"{_pkg_verb} Transferable Package", type="primary"):
            _pkg_extra_config = {
                "meas_utc_offset": st.session_state.get("track_meas_utc_offset", 0.0),
                "interest_height": st.session_state.get(
                    "track_interest_height", float(sorted_heights(track_height_map)[0]["height"])),
                "n_months_at_export": n_months,
                "date_range": [str(combined.index.min()), str(combined.index.max())],
            }
            st.session_state["track_package_bytes"] = build_tracker_package(
                combined, track_height_map, _pkg_extra_config)

        if "track_package_bytes" in st.session_state:
            _download_label = ("Download Updated Transferable Package (.zip)" if _is_continuing
                                else "Download Transferable Package (.zip)")
            st.download_button(_download_label, st.session_state["track_package_bytes"],
                                file_name=f"wind_tracker_package_{combined.index.max().strftime('%Y-%m')}.zip",
                                mime="application/zip")
            st.caption(f"Reflects the data and settings at the moment you clicked "
                       f"'{_pkg_verb} Transferable Package' - click it again after changing "
                       f"anything above.")
