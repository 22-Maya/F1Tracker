# how to install: .venv/bin/pip install flask fastf1 pandas matplotlib numpy && .venv/bin/python app.py
from flask import Flask, render_template, url_for, redirect, flash, Response, request, abort
import hashlib
import fastf1
import pandas as pd
from fastf1 import plotting
import numpy as np
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import matplotlib as mpl
import io
import base64
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- FastF1/Matplotlib Configuration ---
# Enable caching for FastF1
fastf1.Cache.enable_cache('cache') 
plotting.setup_mpl()
colormap = mpl.cm.plasma

# Define plot colors for track layout
BACKGROUND_COLOR = '#0f1724' # Dark background
CIRCLE_COLOR = '#FFFFFF'     # White circle around corner number
LINE_COLOR = '#9aa6b2'       # Muted line to corner number
TEXT_COLOR = '#000000'       # Black text for corner number (for contrast on white circle)
TITLE_COLOR = '#FFFFFF'      # White title text

app = Flask(__name__)

# Mapping of country names to ISO 3166-1 alpha-2 codes for flag emojis
COUNTRY_TO_FLAG = {
    "Azerbaijan": "🇦🇿", "Bahrain": "🇧🇭", "Saudi Arabia": "🇸🇦", "Australia": "🇦🇺", "Japan": "🇯🇵",
    "China": "🇨🇳", "Miami": "🇺🇸", "Monaco": "🇲🇨", "Canada": "🇨🇦",
    "Spain": "🇪🇸", "Austria": "🇦🇹", "United Kingdom": "🇬🇧", "Hungary": "🇭🇺",
    "Belgium": "🇧🇪", "Netherlands": "🇳🇱", "Italy": "🇮🇹", "Germany": "🇩🇪",
    "Singapore": "🇸🇬", "Mexico": "🇲🇽", "Brazil": "🇧🇷", "United Arab Emirates": "🇦🇪",
    "France": "🇫🇷", "Portugal": "🇵🇹", "Turkey": "🇹🇷", "USA": "🇺🇸",
    "United States": "🇺🇸", "UAE": "🇦🇪", "South Africa": "🇿🇦", "Qatar": "🇶🇦"
}

def get_flag(country):
    # Get flag emoji for a country, with fallback
    return COUNTRY_TO_FLAG.get(country, "🏁")

def load_calendar():
    # Loads the F1 event schedule for the current year.
    year = pd.Timestamp.today().year
    # FIXED: Removed 'force_download' argument
    return fastf1.get_event_schedule(year), year 

def get_next_event(schedule):
    # Finds the next upcoming race based on the schedule.
    schedule["date"] = pd.to_datetime(schedule["EventDate"], errors="coerce")
    today = pd.Timestamp.today().date()
    upcoming = schedule[schedule["date"].dt.date >= today]
    if not upcoming.empty:
        return upcoming.iloc[0]
    return schedule.iloc[-1]

# --- Track Visualization Functions ---

def rotate(xy, *, angle):
    """Rotates a 2D array of points (track) by a given angle in radians."""
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot_mat)

def draw_f1_circuit(year, gp_name, event_type='R', max_years_back=1, *, angle_rad=None, show_axes=False, figsize=(16, 9), require_official=True):
    """Generate a track layout visualization for a given F1 Grand Prix.
    
    Returns:
        tuple: (base64_image_data, title_string, year_used)
    """
    current_year = year
    session_event = None
    
    for y in range(current_year, current_year - max_years_back - 1, -1):
        try:
            logging.info(f"Attempting to load data for {y} {gp_name}...")
            
            session_event = fastf1.get_session(y, gp_name, event_type)
            session_event.load(laps=True, telemetry=True, weather=False, messages=False)
            
            # Check if the session actually loaded any laps
            if session_event.laps.empty:
                raise ValueError(f"Session data for {y} {gp_name} contains no laps.")
                
            lap = session_event.laps.pick_fastest()

            # Check if the fastest lap has positional data
            pos = lap.get_pos_data() 
            
            if pos.empty:
                raise ValueError(f"Fastest lap for {y} {gp_name} has no positional data.")

            # Data loaded successfully!
            year = y
            break 
            
        except fastf1._api.SessionNotAvailableError:
            if y == current_year and max_years_back > 0:
                logging.warning(f"Data for {y} {gp_name} not available. Falling back to previous year.")
                continue
            else:
                raise
                
        except ValueError as e:
            logging.warning(f"Data quality error for {y} {gp_name}: {e}")
            if y == current_year and max_years_back > 0:
                continue
            else:
                raise
        
        except Exception as e:
            logging.error(f"Non-API error loading data for {y} {gp_name}: {e}")
            raise e
    else:
        raise fastf1._api.SessionNotAvailableError(f"No data found for {gp_name} from {current_year} to {current_year - max_years_back}")
    
    # --- Generate the track plot ---
    try:
        lap = session_event.laps.pick_fastest()
        pos = lap.get_pos_data()
        
        # 1) Build a Nx2 array of points
        xy = np.column_stack((pos['X'].values, pos['Y'].values))

        # 2) (Optional) rotate around the track centroid to keep it centered
        center = xy.mean(axis=0)
        xy_centered = xy - center

        # 3) Determine rotation to apply. Prefer FastF1's circuit rotation (if available)
        #    The CircuitInfo.rotation is in degrees (positive = CCW) and matches our rotate() convention.
        session_circ = None
        try:
            session_circ = session_event.get_circuit_info()
        except Exception:
            session_circ = None

        rotation_deg = None
        if session_circ is not None:
            rotation_deg = getattr(session_circ, 'rotation', None)

        if angle_rad is not None:
            # explicit override from caller (angle in radians)
            xy_rot = rotate(xy_centered, angle=angle_rad)
        elif rotation_deg is not None:
            # use official circuit rotation (convert degrees -> radians)
            xy_rot = rotate(xy_centered, angle=np.deg2rad(rotation_deg))
            logging.info(f"Applied circuit rotation from FastF1: {rotation_deg} deg for {gp_name} ({year})")
        else:
            if require_official:
                # strict mode: do not apply any fallback rotation
                xy_rot = xy_centered.copy()
                logging.info(f"No official rotation available for {gp_name} ({year}); strict mode: no rotation applied")
            else:
                # fallback: legacy -90 degree rotation for horizontal layout
                xy_rot = rotate(xy_centered, angle=-np.pi/2)

        # 4) Restore centroid and push rotated coords back into pos for plotting
        xy_rot += center
        pos['X_rot'] = xy_rot[:, 0]
        pos['Y_rot'] = xy_rot[:, 1]

        # Create figure and axis (allow overriding figsize)
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        ax.set_facecolor(BACKGROUND_COLOR)
        
        # Plot the track
        ax.plot(pos['X_rot'], pos['Y_rot'], color=LINE_COLOR, linewidth=3, label='Track')
        
        # Mark corners (simplified: mark every nth point as a corner)
        corner_indices = np.arange(0, len(pos), max(1, len(pos) // 8))
        for idx, i in enumerate(corner_indices):
            x, y = pos['X_rot'].iloc[i], pos['Y_rot'].iloc[i]
            ax.add_patch(plt.Circle((x, y), 50, color=CIRCLE_COLOR, fill=False, linewidth=2))
            ax.text(x, y, str(idx + 1), color=TEXT_COLOR, fontsize=10, ha='center', va='center', weight='bold')
        
        ax.set_aspect('equal')

        # Show or hide axes based on flag
        if show_axes:
            ax.axis('on')
            ax.set_title(f'{gp_name} Track Layout ({year})', color=TITLE_COLOR, fontsize=14, weight='bold', pad=20)
            ax.tick_params(colors=TITLE_COLOR)
        else:
            ax.axis('off')
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=BACKGROUND_COLOR, edgecolor='none', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        image_data = base64.b64encode(buf.read()).decode('utf-8')
        
        title = f'{gp_name} ({year})'
        return image_data, title, year
        
    except Exception as e:
        logging.error(f"Error generating track plot for {gp_name}: {e}")
        raise

# --- Flask Routes ---

@app.route("/")
def index():
    schedule, year = load_calendar()
    return render_template("index.html", schedule=schedule.to_dict(orient="records"), year=year, get_flag=get_flag)

@app.route("/schedule")
def schedule_page():
    schedule, year = load_calendar()
    # Parse UTC date for the template
    schedule["date_utc"] = pd.to_datetime(schedule["Session1DateUtc"], utc=True)

    # --- Precompute FastF1-compatible safe name ---
    schedule["FastF1Name"] = (
        schedule["EventName"]
        .fillna("")                                        # avoid NaN
        .str.replace(r"\s*Grand Prix$", "", regex=True)    # remove trailing ' Grand Prix'
        .str.strip()                                       # trim whitespace
        .str.replace(r"\s+", "_", regex=True)              # spaces -> underscores
        .str.replace(r"[^A-Za-z0-9_]", "", regex=True)     # drop punctuation
    )

    schedule = schedule.assign(
        date_iso=schedule["date_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    return render_template(
        "schedule.html",
        year=year,
        schedule=schedule.to_dict(orient="records"),
        get_flag=get_flag,
    )

@app.route("/next")
def next_race():
    schedule, year = load_calendar()
    next_event = get_next_event(schedule)
    
    event_dict = next_event.to_dict()
    if "Session1DateUtc" in event_dict:
        event_dict["date_utc"] = pd.to_datetime(event_dict["Session1DateUtc"], utc=True).isoformat()
    
    # Add the FastF1-compatible name for the track link
    event_dict["FastF1Name"] = next_event["EventName"].replace(' Grand Prix', '').replace(' ', '_')

    # --- Server-side: generate the track image for the Next Race page ---
    try:
        image_data, title, source_year = draw_f1_circuit(year, event_dict["FastF1Name"], max_years_back=1)
        data_available = bool(image_data)

    except fastf1._api.SessionNotAvailableError:
        image_data = ""
        title = f"Track Data Unavailable for {event_dict['FastF1Name']} ({year} or {year-1})"
        data_available = False
        source_year = year

    except Exception as e:
        logging.error(f"Error generating track for Next Race: {e}")
        image_data = ""
        title = f"Error loading track data for {event_dict['FastF1Name']}."
        data_available = False
        source_year = year

    # Basic track info for the About section
    track_info = event_dict.get('Location', '') or event_dict.get('EventName', '')

    return render_template("next_race.html",
                           event=event_dict,
                           year=year,
                           get_flag=get_flag,
                           image_data=image_data,
                           title=title,
                           gp_name=event_dict["FastF1Name"],
                           data_available=data_available,
                           track_info=track_info)

@app.route("/race/<int:year>/<string:gp_name>")
def race_view(year, gp_name):
    # Sanitize and find the event in the schedule
    schedule, sched_year = load_calendar()
    def _san(s): return s.replace(' Grand Prix', '').replace(' ', '_')
    matches = schedule[schedule['EventName'].apply(lambda s: _san(s) == gp_name)]
    if matches.empty:
        # not found -> 404
        return abort(404, f"Race '{gp_name}' not found in {year}")

    event = matches.iloc[0].to_dict()
    
    if "Session1DateUtc" in event:
        event["date_utc"] = pd.to_datetime(event["Session1DateUtc"], utc=True).isoformat()
    event["FastF1Name"] = gp_name

    # Generate the same track variables as in next_race()
    try:
        image_data, title, source_year = draw_f1_circuit(year, gp_name, max_years_back=1)
        data_available = bool(image_data)
    except fastf1._api.SessionNotAvailableError:
        image_data = ""
        title = f"Track Data Unavailable for {gp_name} ({year} or {year-1})"
        data_available = False
        source_year = year
    except Exception as e:
        logging.error(f"Error generating track for race view: {e}")
        image_data = ""
        title = f"Error loading track data for {gp_name}."
        data_available = False
        source_year = year

    return render_template("next_race.html",
                           event=event,
                           year=year,
                           get_flag=get_flag,
                           image_data=image_data,
                           title=title,
                           gp_name=gp_name,
                           data_available=data_available,
                           track_info=event.get('Location', ''))

@app.route("/track/<int:year>/<string:gp_name>")
def track_layout(year, gp_name):
    gp_name_sanitized = gp_name.replace(' Grand Prix', '').replace(' ', '_')
    
    source_year = year 

    try:
        # Try to generate the image, with fallback to previous year
        image_data, title, source_year = draw_f1_circuit(year, gp_name_sanitized, max_years_back=1)
        data_available = bool(image_data)
        
    except fastf1._api.SessionNotAvailableError:
        image_data = ""
        title = f"Track Data Unavailable for {gp_name} ({year} or {year-1})"
        data_available = False
        source_year = year # Use the requested year for the error message

    except Exception as e:
        logging.error(f"Critical error in track_layout route for {year} {gp_name}: {e}")
        image_data = ""
        title = f"Error loading track data for {gp_name}."
        data_available = False
        source_year = year
    
    return render_template('track_layout.html', 
                           image_data=image_data, 
                           title=title,
                           gp_name=gp_name, 
                           year=year, 
                           source_year=source_year,
                           data_available=data_available)

@app.route("/track_image/<int:year>/<string:gp_name>")
def track_image(year, gp_name):
    """Return the PNG bytes for a track image. Supports optional query params:
       - angle: rotation in degrees (positive CCW)
       - show_axes: 1 to show axes, 0 (default) to hide
       - w, h: figsize width and height in inches (floats)
    """
    gp_name_sanitized = gp_name.replace(' ', '_')

    angle = request.args.get('angle', None)
    show_axes_flag = request.args.get('show_axes', '0')
    w = request.args.get('w', None)
    h = request.args.get('h', None)
    refresh_flag_raw = request.args.get('refresh', '0')

    try:
        angle_rad = None if angle is None else np.deg2rad(float(angle))
    except Exception:
        return abort(400, "Invalid angle parameter")

    try:
        show_axes = bool(int(show_axes_flag))
    except Exception:
        show_axes = False

    figsize = None
    try:
        if w is not None and h is not None:
            figsize = (float(w), float(h))
    except Exception:
        return abort(400, "Invalid w/h parameters")

    cache_dir = os.path.join('cache', 'generated')
    os.makedirs(cache_dir, exist_ok=True)

    cache_key = f"{year}:{gp_name_sanitized}:{angle}:{show_axes}:{figsize}"
    cache_hash = hashlib.sha256(cache_key.encode('utf-8')).hexdigest()
    cache_file = os.path.join(cache_dir, f"{cache_hash}.png")

    # If not forcing refresh and cache exists, return cached image
    try:
        refresh_flag = bool(int(refresh_flag_raw))
    except Exception:
        refresh_flag = False

    if os.path.exists(cache_file) and not refresh_flag:
        with open(cache_file, 'rb') as f:
            img_bytes = f.read()
        resp = Response(img_bytes, mimetype='image/png')
        resp.headers['Cache-Control'] = 'public, max-age=3600'
        return resp

    # If refresh requested and cache exists, remove it and regenerate
    if os.path.exists(cache_file) and refresh_flag:
        try:
            os.remove(cache_file)
        except Exception:
            logging.warning(f"Unable to remove cache file {cache_file}")

    # Generate fresh image
    try:
        # In strict-only mode we do not allow manual rotation via query params.
        if angle is not None:
            return abort(400, "Manual angle parameter is not allowed: server enforces strict rotation only.")

        kw = {}
        # Always require official rotation (strict-only)
        kw['require_official'] = True
        if figsize is not None:
            kw['figsize'] = figsize
        kw['show_axes'] = show_axes

        image_b64, title, used_year = draw_f1_circuit(year, gp_name_sanitized, max_years_back=1, **kw)
        img_bytes = base64.b64decode(image_b64)

        # Save to cache for future reuse (overwrite if refresh)
        try:
            with open(cache_file, 'wb') as f:
                f.write(img_bytes)
        except Exception:
            logging.warning(f"Unable to write cache file {cache_file}")

        resp = Response(img_bytes, mimetype='image/png')
        # If refresh, mark as no-cache for immediate client fetch
        if refresh_flag:
            resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        else:
            resp.headers['Cache-Control'] = 'public, max-age=3600'
        return resp

    except fastf1._api.SessionNotAvailableError:
        return abort(404, "Track data not available")
    except Exception as e:
        logging.error(f"Error generating track image: {e}")
        return abort(500, "Error generating image")

if __name__ == "__main__":
    if not os.path.exists('cache'):
        os.makedirs('cache')
    # Use port 5001 to avoid macOS ControlCenter/AirPlay on 5000
    app.run(debug=True, port=5001)