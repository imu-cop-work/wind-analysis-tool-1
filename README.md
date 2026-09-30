# Wind Resource Analysis Tool

A Streamlit app with four separate tools, switchable from the sidebar:

1. **Long-Term Correction** - measurement + modelled data correlation and
   long-term wind speed pipeline. The user maps their
   own CSV columns interactively.
2. **Measurement Campaign Planning** - preliminary wind resource look-up and
   LiDAR/FLiDAR siting: upload a site boundary, a turbine layout, and one or
   more modelled wind maps (ESRI ASCII grid `.asc`, e.g. Vortex map exports),
   click on an interactive map to place candidate measurement points, compare
   wind speed across points and maps, and run an automated K-Means-based
   search for the best measurement locations relative to your layout.
3. **Preliminary Wind Resource Assessment** - multi-source (ERA5 + CFSR) wind
   resource assessment across a turbine layout: upload every `.asc` map for
   each source (height auto-detected from the filename), compute a per-position
   shear exponent for each source, extrapolate and blend to a hub height with
   adjustable ERA5/CFSR weighting, then calibrate the result against
   long-term-corrected measurements at known locations (site-average,
   distance-weighted, or kriging calibration factor, weighted by each
   calibration point's own data quality).
4. **Wind Measurement Tracker** - upload monthly measurement files as they
   arrive (`.csv`, Campbell Scientific TOA5 `.dat`/`.sta`, a second
   `.dat`/`.sta` layout seen from floating LiDAR buoy systems, or NetCDF
   `.nc`) and track wind speed, direction, turbulence intensity, and data
   availability over time as the record grows.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run wind_streamlit_app.py
```

Then open the local URL it prints (usually http://localhost:8501).

## Deployment

The app is hosted on **Azure App Service** (Linux, Python 3.12) behind
Entra ID sign-in, and deploys automatically from this repo.

**To ship a change: commit, push to `main`.** That's the whole workflow.
`azure-pipelines.yml` is a single job that:

1. **Validates** - installs `requirements.txt` on Linux/Python 3.12, checks
   every library actually imports, and syntax-checks the app.
2. **Packages** the source into a zip.
3. **Deploys** it to App Service, which runs `pip install -r requirements.txt`
   itself and starts the app via `startup.sh`. First boot after a dependency
   change takes a few minutes while that install runs.

Pull requests run steps 1 and 2 but not the deploy, so a dependency that
doesn't work on Linux is caught before it can reach production.

There is deliberately no manual approval gate - push to `main` goes straight
to production. If one is ever wanted, the deploy step has to be split into its
own pipeline *stage* first: Azure DevOps environment approvals attach to
stages, not to individual steps.

### Adding a dependency

Add a line to `requirements.txt` with an `==` pin and push. If it can't
install on Linux, the pipeline fails and nothing is deployed.

Versions are pinned deliberately. Two matter beyond reproducibility:
`streamlit` must be >= 1.49 (the app uses `width="stretch"` and
`st.pyplot(..., width=, dpi=)`), and `pyogrio` is the only place native GDAL
enters the stack, via `geopandas`. To refresh the pins, resolve them in a
clean environment rather than editing by hand:

```powershell
python -m venv .venv-fresh
.venv-fresh\Scripts\activate
pip install -r requirements.txt
pip freeze | Out-File -Encoding utf8 requirements.txt
```

`Out-File -Encoding utf8`, not `>`: PowerShell's `pip freeze > requirements.txt`
writes UTF-16, which pip cannot read.

### When something breaks

- **App won't start, or errors at runtime** - App Service > Log stream.
- **Change didn't deploy** - the pipeline run page in Azure DevOps.

### Notes

- `.streamlit/config.toml` sets a 500 MB upload limit, needed because the
  Preliminary WRA tool expects whole folders of `.asc` maps to be dropped in.
- All state lives in the Streamlit session, in memory. The app runs on a
  single instance; if it is ever scaled out, session affinity must be enabled
  or users will lose uploaded maps mid-analysis.
- `startup.sh` must keep LF line endings (enforced by `.gitattributes`) -
  with Windows CRLF, bash fails to start it.
- KML boundaries are parsed with Python's built-in `xml.etree.ElementTree`,
  not `gpd.read_file`/pyogrio's KML driver - pyogrio only added `libkml` to
  its bundled wheels recently and coverage still varies by platform/version,
  so a standard-library parser avoids KML support silently depending on
  exactly which pyogrio build is installed. Only extracts `<Polygon>`
  geometry (with holes, if any); point/line placemarks are ignored.
- Easting/Northing conversion uses `pyproj` directly (UTM zone/hemisphere or
  a custom EPSG code) - `pyproj` is already a mandatory dependency of
  `geopandas`, not an optional one, so this needed no new
  `requirements.txt` entry.
- `netCDF4` was added specifically for the Wind Measurement Tracker's `.nc`
  support - unlike everything above, this genuinely is a new dependency,
  since nothing else in the stack pulls it in transitively. Used directly
  rather than via `xarray`, since a generic "list variables, let the user
  map them" reader doesn't need `xarray`'s higher-level Dataset abstraction,
  and `netCDF4` alone keeps the dependency footprint smaller.
- TOA5 (`.dat`/`.sta`) files are parsed with Python's `csv` module rather
  than a dedicated logger library, matching the same "avoid an extra
  dependency where a well-documented format can be parsed directly"
  philosophy as the KML parser above. The format is well-documented and
  stable (a fixed 4-line header, then comma-delimited data), so this isn't
  the same kind of version-fragility risk the KML driver had - it's simply
  more direct than adding a dependency for a format this simple.

## How to use - Long-Term Correction

1. **Upload measurement data** - any number of files, any mix of `.csv`,
   Campbell Scientific TOA5 `.dat`/`.sta` (a 4-line ASCII header, then
   comma-delimited data - the standard format for any contemporary Campbell
   Scientific datalogger), a second, distinct `.dat`/`.sta` layout seen in
   practice from floating LiDAR buoy systems (a short title line, then an
   unquoted header row whose first field is literally `timestamp`, usually
   a units row, then unquoted comma-delimited data), or NetCDF `.nc`
   (self-describing - the tool lists whatever variables a given file
   contains for you to map). Files are auto-detected by format,
   concatenated, and sorted by time; any overlapping timestamps across
   files are de-duplicated (first occurrence kept). `.dat`/`.sta` files
   that match neither layout are skipped with a warning rather than
   silently mis-parsed.
   - CSV files share one column mapping (timestamp column, day-first
     setting, invalid-value codes) applied to every CSV uploaded. The app
     checks your date-format choice against the first CSV's data and will
     warn you if it looks wrong (e.g. a huge chunk of rows failing to parse).
   - The LiDAR-buoy-style `.dat`/`.sta` files share one dayfirst/invalid-
     codes setting, applied to every file in that group - dayfirst is
     pre-filled from the units row's date-format descriptor when present,
     and the invalid-codes default includes `9998`, a real sentinel value
     seen in this kind of export.
   - NetCDF files share one variable mapping (which variable is time,
     which are the values to import), applied to every `.nc` uploaded.
     Only 1-D (time-indexed) variables are supported.
   - TOA5 files need no mapping - the timestamp format is unambiguous, and
     missing values (the literal string `NAN` in that format) convert to
     real NaNs automatically.
2. **Map heights**: pick, for each height you care about, the wind speed
   column (required) and wind direction column (optional - needed only for
   wind roses), from whatever columns the combined dataset ended up with.
   Heights can be entered in any order - they're sorted low-to-high
   automatically everywhere in the results.

   Beyond whatever invalid-value codes were specified during upload, any
   wind speed reading over 100 m/s (or negative) or wind direction outside
   0-360° is automatically treated as invalid too, for every mapped
   height in the measurement data. This exists because a sensor fault or
   logging glitch can produce a garbage value that isn't one of the usual
   sentinels (9999, -999, NaN, etc.) - and even a couple of such rows can
   quietly wreck an entire correlation, the same way one real outlier
   wouldn't: an R² near 0 dragged down entirely by 2 rows out of tens of
   thousands looks like a genuinely poor correlation, but isn't one.
   Verified directly - injecting two ~1,470 m/s readings into an otherwise
   well-correlated synthetic dataset dropped hourly R² from 0.88 to 0.03;
   filtering those two rows out restored it to 0.88. A message shows
   exactly what was caught and where, whenever this filter actually finds
   something, so it never silently changes data without you seeing it.
   Modelled datasets aren't checked this way, since they're assumed clean -
   matching the same reasoning behind not asking for invalid-value codes
   when configuring one.
3. **Set your measurement's timezone offset** - right after mapping columns,
   set once regardless of how many modelled sources you go on to compare
   against, since it doesn't change depending on which one it's being
   compared to.
4. **Upload one or more modelled wind datasets** - any hourly modelled/
   reanalysis wind time series works (ERA5, CFSR, MERRA-2, Vortex, or
   similar), and you're not limited to one: add a Vortex ERA5 extraction,
   a Vortex CFSR one, and a plain MERRA2 series side by side, to see how
   each actually correlates with your measurement before committing to one.
   Each source is configured once (upload, column mapping, height, its own
   timezone offset, a label for charts) and then collapses to a one-line
   summary; "+ Add a modelled source" opens the same form again for the
   next one. Vortex `.txt` exports are auto-detected and parsed
   automatically (timestamp, height, and timezone all read from the file
   header, pre-filling that source's own offset); other formats are mapped
   the same way as the measurement file (timestamp, wind speed, wind
   direction columns, plus the dataset's height).
5. Browse the result tabs: Data Availability, Monthly Means, Wind Rose,
   Shear Profile, Correlation, and Long-Term Result.
   - Shear is calculated using only heights above your chosen data-availability
     threshold (default 80%), fit once on the time-averaged profile (robust
     "profile method", not noisy per-timestamp averaging) - unaffected by how
     many modelled sources are configured, since it only uses measurement data.
   - **Wind Rose**: pick a measured height once, and every configured source's
     rose for that height is shown - each one using its own native-resolution
     data over the period it overlaps with your measurement, not hourly-
     averaged or matched to concurrent timestamps with the measurement side.
     A wind rose describes the distribution of direction and speed over a
     period, which doesn't need paired, simultaneous samples the way a
     correlation does - and there's no reason to average a circular quantity
     like direction down to hourly when the native resolution is already
     available and more representative. If your modelled dataset spans 2002
     to 2026 but your measurement only covers Jun 2024 to Dec 2025, the
     modelled panel reflects Jun 2024 to Dec 2025, not the full record.
     Concurrent-hour matching and hourly averaging are still used exactly
     where they're actually needed: Correlation and Long-Term Result.
   - Correlation is done at each source's own height, using a direct
     measurement match if you mapped one close enough, or shear-extrapolating
     from your nearest mapped height otherwise. Sub-hourly modelled data is
     automatically averaged to hourly first. The tab is organised by panel
     type (Hourly/Daily/Monthly, plus native resolution if any source is
     sub-hourly), with every configured source shown side by side within
     each panel type, at a fixed display width regardless of how many
     sources there are - so the same statistic (e.g. Monthly R²) is directly
     comparable across sources without hunting through separate sections,
     and a single configured source doesn't render oversized.
   - **Long-Term Result**: every configured source gets its own full
     breakdown here - concurrent-period stats, both TLS and OLS estimates,
     and its own chart - not just a single pick hidden behind a selector.
     A selector still defaults to whichever configured source has the
     highest hourly R², marked with a star for headline reporting, but nothing
     about the other sources' results is hidden behind that choice. The
     long-term result itself uses orthogonal (total least-squares) regression
     against the full modelled record as the primary estimate, with the OLS
     result shown alongside for reference; the result at your chosen "height
     of interest" is obtained by shear-extrapolating from each source's own
     height.
6. **Download**: package every chart (plus a summary and the availability
   table) into a single ZIP from the Download section at the bottom - covers
   every configured source's correlation and long-term result, not just the
   one selected for headline reporting; the wind rose export uses whichever
   height is currently selected on screen.

## How to use - Measurement Campaign Planning

1. **Site boundary**: `.geojson`, `.kml`, or a CSV/Excel list of
   Easting/Northing boundary vertices - pick a UTM zone/hemisphere or enter
   a custom EPSG code (e.g. 27700 for OSGB36) to convert them. Vertices are
   connected in the order they appear in the file, so list them in
   boundary-walking order (as most survey/CAD exports already do).
2. **Wind maps**: each source gets its own named column, side by side (start
   with ERA5 and CFSR, rename either, remove one, or add more - any number
   of sources with any names are supported). Drop that source's `.asc`
   files straight into its column - e.g. a Vortex map export at 100m and
   another at 150m - no separate step to add them; height is read from
   each filename.
3. **Turbine layout** (optional but needed for step 6): `.geojson`, `.gpkg`,
   `.xlsx` (with Latitude/Longitude columns - column names are
   configurable), or a CSV/Excel of Easting/Northing turbine positions
   (same UTM zone/EPSG conversion as the boundary). Raw `.shp` isn't
   supported directly since it's really several files bundled together -
   convert or export to one of the above.
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
   cluster's modelled wind speed (weighted against distance to the turbines),
   reporting mean/max deviation for each recommended point, downloadable as CSV.

## How to use - Preliminary Wind Resource Assessment

1. **Site Boundary** and **Layout**: same as Measurement Campaign Planning.
2. **Wind Maps**: upload every `.asc` file for ERA5 and for CFSR separately -
   either drag-and-drop the whole folder (Chrome/Edge expand a dropped folder
   automatically) or multi-select all the files. Height is read from each
   filename (matching a `.M.<N>m` pattern, e.g. `site.M.100m.asc`, with a
   looser fallback pattern if that's not found).
3. **Interactive Map**: browse any loaded map with the layout overlaid.
4. **Shear Calculation**: fits a power-law shear exponent to ERA5's own
   heights and CFSR's own heights *separately, at every turbine position*
   (not one number pooled across the whole site), then averages the two per
   position - needs at least 2 heights loaded for a source to use it.
5. **Hub Height & Weighting**: set the hub height and an ERA5/CFSR weight
   slider (CFSR is automatically 100% minus ERA5 - no manual balancing
   needed); extrapolates each source to hub height per position using that
   position's own shear, then blends them, shown on an interactive map
   coloured by wind speed, plus the site-average result.
6. **Calibration**: add long-term-corrected wind speeds at known locations,
   manually or via CSV (any column names - you map them), then choose:
   - **Site Average CF**: one calibration factor applied everywhere.
   - **Distance Weighted CF**: each turbine gets a blend of every
     calibration point's factor, weighted by true great-circle distance and
     a decay length you set - nearer points count more.
   - **Kriging (Ordinary)**: a geostatistical blend using the same distance
     decay ("range") as the Distance Weighted method, but it additionally
     accounts for redundancy between calibration points - two points sitting
     close together carry mostly overlapping information, and kriging
     down-weights that overlap instead of counting each point as fully
     independent evidence the way Distance Weighted CF does. Generally the
     more statistically sound choice once you have 3 or more calibration
     points, particularly if any are clustered.

   Each calibration point also has an optional **Measurement Duration (months)**
   field. Leave it at 0 for a point you don't want to flag - it's treated as
   full-confidence, identical to not having this field at all. If a point's
   long-term estimate is built on a short measurement record, enter how many
   months it's based on; it will be down-weighted in the calibration
   accordingly (on top of, not instead of, its distance/redundancy
   standing) rather than being trusted as fully as a point backed by a
   longer record.

   Duration is converted to a confidence weight via **inverse-variance
   weighting** - the statistically standard way to combine estimates of
   differing precision (the same principle behind meta-analysis pooling) -
   rather than an assumed curve shape. The underlying uncertainty percentage
   is anchored to real measured values from Abascal Mendez et al. (2026,
   *Inventions* journal): 30 real meteorological masts worldwide, up to 27
   months of concurrent data, ~2.1% dispersion-based MCP uncertainty at 3
   months down to ~0.4% at 12 months, in a relationship they found to be
   approximately linear across that range. Because inverse-variance
   weighting squares that ratio (variance, not standard deviation), a short
   record is penalized considerably harder than a simple proportional scale
   would suggest - a 5-month record works out to roughly 5% confidence
   relative to a 12+ month one, not roughly 40%. An optional **Correlation
   R² (0-1)** field applies a further adjustment if you have it from your
   own MCP regression, using the standard regression-theory relationship
   between R² and unexplained variance, relative to a reference of R²=0.95
   chosen for this tool (not a value taken from the cited study).

   Caveats worth knowing: the cited study's percentages are for a Linear
   Regression MCP model averaged across all terrain types (not stratified
   by site-specific terrain, and not the exact TLS method this tool's
   Long-Term Correction mode defaults to, though the study found the two
   comparable); below 3 months the relationship is extrapolated rather than
   directly measured; and it's one study, not a consensus figure - check the
   cross-validation table to see whether it actually improves accuracy on
   your own points.

   If you'd rather set a point's confidence yourself - your own engineering
   judgement, or an uncertainty you've computed independently - a **manual
   weight (0-1)** overrides both Duration and R² entirely for that point; it
   can be set per point manually, or via an optional column when importing a
   calibration CSV. This applies to all three methods, including Site Average
   CF. For Kriging specifically, this is implemented as a per-point addition
   to the kriging system's diagonal (each point's own uncertainty, distinct
   from the shared decay/range parameter) - a well-established geostatistical
   technique sometimes called kriging with measurement error, or filtered
   kriging. One side effect worth knowing about: with few calibration points,
   or points clustered close together, a heavily down-weighted point can push
   its own kriging weight negative - expected, standard kriging behaviour
   (it isn't a constrained weighted average), not a bug. A "Prevent negative
   kriging weights" checkbox (on by default when Kriging is selected) clips
   any negative weight to zero and renormalizes the rest; testing against
   randomized small (2-3 point) calibration sets with one point flagged for
   a short record found this happens in nearly all of them, with clipping
   cutting cross-validated error roughly 5-8x on average - but it isn't a
   universal fix for every individual configuration, so check the
   cross-validation table below with it toggled either way for your own
   points rather than assuming.

   With 2 or more calibration points, a leave-one-out cross-validation table
   shows the actual RMSE and MAE each method achieves on your own points
   (each point is predicted from the others in turn and compared to its
   measured value) - use this to see which method genuinely fits your site
   rather than assuming one is best.

   Shows the calibrated result on its own interactive map alongside the
   calibration points, plus each point's average influence across the layout.
7. **Download Plots and Table**: packages an Excel workbook (site boundary,
   layout, ERA5/CFSR/weighted wind speed by position, and calibration data
   if available) plus static map images into a ZIP.

## How to use - Wind Measurement Tracker

Each month brings a new raw file. Rather than re-uploading every prior
month's raw files and redoing every column mapping each time, download a
**transferable package** at the end of a session (Step 5) and load it back
in next time via the **"Continue tracking"** toggle at the top of the page,
before uploading anything else - it carries forward the combined
measurement history and the height-mapping/analysis settings, so only the
new month's file(s) need adding. The package is a plain ZIP (a CSV of the
combined measurement data plus a JSON config file - deliberately not a
binary format, so it needs no new dependency and stays inspectable). Nothing
about the modelled dataset (Vortex/ERA5/etc.) is stored in it, since that
doesn't change month to month - re-upload it fresh each session, same as
every other mode in this app.

"Start fresh" is a one-time thing, for setting up a new project - after
that, every regular monthly visit should go through "Continue tracking".
The Step 5 button reflects which one you're doing: it reads **"Create
Transferable Package"** the first time, and **"Update Transferable
Package"** once a package has actually been loaded this session - each
update already reflects the full history to date (old months plus whatever
was just added), not just the newest month, so there's no separate
history log to maintain - just keep loading the most recently downloaded
package back in next time.

1. **Upload New Measurement Data** (labelled "Upload Measurement Data" when
   starting fresh): as many files as you have - `.csv`, Campbell Scientific
   TOA5 `.dat`/`.sta` (a 4-line ASCII header, then comma-delimited data -
   the standard format for any contemporary Campbell Scientific
   datalogger), a second, distinct `.dat`/`.sta` layout seen in practice
   from floating LiDAR buoy systems (a short title line, then an unquoted
   header row whose first field is literally `timestamp`, usually a units
   row, then unquoted comma-delimited data), or NetCDF `.nc`
   (self-describing - the tool lists whatever variables a given file
   contains for you to map, rather than assuming a fixed structure). Files
   are auto-detected by format, concatenated, and sorted by time; any
   overlapping timestamps across files are de-duplicated (first occurrence
   kept). `.dat`/`.sta` files that match neither layout are skipped with a
   warning rather than silently mis-parsed.
   - CSV files share one column mapping (timestamp column, day-first
     setting, invalid-value codes) applied to every CSV uploaded.
   - The LiDAR-buoy-style `.dat`/`.sta` files share one dayfirst/invalid-
     codes setting, applied to every file in that group - dayfirst is
     pre-filled from the units row's date-format descriptor (e.g.
     `DD-MM-YYYY hh:mm`) when present, and the invalid-codes default
     includes `9998`, a real sentinel value seen in this kind of export
     (e.g. a LiDAR range gate with no valid return at a given timestamp).
   - NetCDF files share one variable mapping (which variable is time,
     which are the values to import) applied to every `.nc` uploaded. Only
     1-D (time-indexed) variables are supported - a single-height dataset.
   - TOA5 files need no mapping - the timestamp format is unambiguous, and
     missing values (recorded as the literal string `NAN` in that format)
     convert to real NaNs automatically.
2. **Map heights**: when continuing from a package, this is restored
   automatically and shown as a **read-only table** - expand "Height
   mapping" only if something actually needs changing (a new height, a
   renamed column). Starting fresh shows the full editable mapping
   directly: for each height, pick the wind speed column (required), and
   optionally wind direction and turbulence intensity - either a direct TI
   column, or computed from a wind-speed standard-deviation column
   alongside the mean (the standard TOA5 convention: a `_Std` column next
   to `_Avg`, meant for exactly this).

   Beyond whatever invalid-value codes were used during upload, any wind
   speed reading over 100 m/s (or negative) or wind direction outside
   0-360° is automatically treated as invalid too, for every mapped
   height - a sensor fault can produce a garbage value that isn't one of
   the usual sentinels, and even a couple of such rows can quietly wreck a
   whole correlation later in Long-Term Analysis. A message shows exactly
   what got caught, whenever this catches something. The modelled dataset
   used in Long-Term Analysis isn't checked this way, since it's assumed
   clean.

   Once heights are mapped, and only when continuing from a package, a
   **"What's new this update"** table compares mean wind speed and data
   availability per height across three columns - before this update, the
   newly added month(s) alone, and the combined result - so what changed is
   visible immediately rather than needing to be inferred from the charts.
3. Browse the result tabs: **Data Availability**, **Monthly Mean WS**,
   **TI vs Wind Speed**, and **Wind Rose by Month** - one small rose per
   calendar month, sharing both the speed-bin colours and the radial
   (percentage) scale across every month so they're honestly comparable to
   each other, rather than each auto-scaled to its own peak. Built this way
   specifically to spot directional shifts across the tracked period, which
   neither a single overall rose nor a monthly mean direction number can
   show - a circular mean collapses an entire month's distribution to one
   value and can be actively misleading for a spread-out or bimodal one, so
   there's no separate "Monthly Mean WD" metric.

   **TI vs Wind Speed** is a grid of small charts, one per calendar month,
   each plotting mean turbulence intensity against wind speed (in 1 m/s
   bins, with bins under 3 samples dropped as too sparse to trust) - the
   standard way turbulence intensity is characterised in the wind industry,
   since TI varies systematically with wind speed (typically higher and
   noisier at low speeds, decreasing and flattening at higher ones) rather
   than being one representative number. Readings below 1 m/s are excluded
   from the binning, since TI is not meaningful that close to calm. Both
   axes share the same scale across every month's subplot for a fair
   comparison. When continuing from a package, the Monthly Mean WS chart
   colours the newly added month(s) in amber, distinct from the rest of the
   history in blue - the TI-vs-WS grid doesn't carry this highlighting,
   since each subplot already represents a single month on its own rather
   than a shared bar chart with one bar per month.

   Both the wind rose grid and the TI-vs-WS grid are cached (keyed on the
   underlying data) rather than rebuilt on every interaction - both are
   genuinely expensive to render (several seconds for a multi-year
   dataset), and without caching, that cost was being paid on every rerun
   of the whole app, including ones triggered by something unrelated
   elsewhere on the page, like typing in a Long-Term Analysis field.
4. **Long-Term Analysis**: upload a modelled dataset (Vortex `.txt` or any
   hourly CSV reanalysis product - ERA5, CFSR, MERRA-2), same format
   handling as Long-Term Correction. A shear exponent is fit across your
   mapped heights, the measured series closest to the modelled dataset's
   own height is correlated against it (TLS regression, matching
   Long-Term Correction's method), and that regression is fit **once** -
   using every concurrent hour currently available - to produce a single
   long-term wind speed estimate at your chosen height of interest, shown
   as a fixed reference line. Separately, the **cumulative** mean of your
   actual measured data at that height - month 1's mean, then months 1-2's,
   then 1-3's, and so on - is plotted alongside it, so you can see whether
   your running on-site average is settling in near the long-term estimate
   or still moving as more months come in. The regression itself isn't
   refit at each point; only the amount of measured data feeding the
   running average changes - this tracks whether your *sample* looks
   representative yet, not whether the *regression* has stabilized.
5. **Download**: package every chart into a single ZIP, or download the
   **transferable package** to continue tracking next month without
   re-uploading or re-mapping everything from scratch (see the top of this
   section).
