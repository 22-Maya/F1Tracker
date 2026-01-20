# What Everything Means - Quick Reference

## 1. API Functions

### `get_openf1_json(url)` 
**What:** Fetches JSON data from Ergast API
**Why:** Replaces the old Formula 1 API (no API key needed)
**Example:**
```python
data = get_openf1_json("https://ergast.com/api/f1/2026/driverStandings.json")
# Returns: {"MRData": {"StandingsTable": {...}}}
```

---

## 2. TEAM_DATA Dictionary

**What:** Maps F1 team names to their brand colors
**Why:** Used to color-code standings by team
**Structure:**
```python
TEAM_DATA = {
    "Mercedes": {"color": "#00D2BE", "logo": "mercedes.png"},
    #           └─ Team name used in API responses
    #                          └─ Hex color code (like CSS)
    #                                            └─ Logo filename (future feature)
}
```

**Usage in championship route:**
```python
"team_color": TEAM_DATA.get(d['Constructors'][0]['name'], {}).get('color', '#9aa6b2')
#             └─ Look up the team name
#                                                              └─ Fallback color if team not found
```

---

## 3. Championship Route (`@app.route("/championship")`)

**What:** Fetches and displays F1 standings
**How it works:**

### Step 1: Get current year
```python
year = pd.Timestamp.today().year  # 2026
```

### Step 2: Fetch drivers standings from API
```python
drv_url = f"https://ergast.com/api/f1/{year}/driverStandings.json"
#         └─ URL format: ergast.com/api/f1/{year}/{what}.json
drv_data = get_openf1_json(drv_url)
#         └─ Makes HTTP request and returns JSON
drivers_raw = drv_data['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']
#            └─ Navigate through nested JSON structure
```

### Step 3: Transform raw data into clean dictionaries
```python
drivers = [
    {
        "name": f"{d['Driver']['givenName']} {d['Driver']['familyName']}",
        #        └─ Combine first + last name
        "team": d['Constructors'][0]['name'],
        #       └─ Get team name (first constructor in list)
        "team_color": TEAM_DATA.get(d['Constructors'][0]['name'], {}).get('color', '#9aa6b2'),
        #            └─ Look up team color from TEAM_DATA
        "points": d['points']
        #        └─ Championship points
    }
    for d in drivers_raw  # For each driver in the raw list
]
```

### Step 4: Repeat for constructors
```python
constructors = [
    {
        "team": c['Constructor']['name'],
        "team_color": TEAM_DATA.get(c['Constructor']['name'], {}).get('color', '#9aa6b2'),
        "points": c['points']
    }
    for c in constructors_raw
]
```

### Step 5: Pass to template
```python
return render_template(
    "championship.html",
    drivers=drivers,  # List of driver dicts
    constructors=constructors,  # List of constructor dicts
    year=year  # Current year
)
```

---

## 4. Championship Template (championship.html)

### Jinja2 Loop
```html
{% for d in drivers %}
    <li style="border-left: 4px solid {{ d.team_color }};">
        <strong>{{ loop.index }}. {{ d.name }}</strong>
        <!--    └─ Auto-incrementing counter (1, 2, 3...)
                                └─ Team color variable from Python
                                              └─ Driver name -->
        <span>{{ d.team }} — {{ d.points }} pts</span>
    </li>
{% endfor %}
```

### Tab Switching JavaScript
```javascript
driversBtn.onclick = () => {
    drivers.style.display = "block";      // Show drivers div
    constructors.style.display = "none";  // Hide constructors div
}
```

---

## 5. Navigation Updates

**Before:**
```html
<a href="{{ url_for('index') }}">🏁 Home</a>
<a href="{{ url_for('schedule_page') }}">📅 Race Schedule</a>
<a href="{{ url_for('next_race') }}">🗺️ Next Track</a>
```

**After:**
```html
<a href="{{ url_for('index') }}">🏁 Home</a>
<a href="{{ url_for('championship') }}">🏆 Championship</a>
<!--  └─ NEW LINK ────────────────────┘  └─ Trophy emoji -->
<a href="{{ url_for('schedule_page') }}">📅 Race Schedule</a>
<a href="{{ url_for('next_race') }}">🗺️ Next Track</a>
```

---

## 6. Data Flow Diagram

```
┌─────────────────────────────────────┐
│  User clicks "Championship" (🏆)    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Flask route @app.route("/championship")
│  Fetches current year (2026)        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  HTTP GET ergast.com/api/f1/2026/.. │
│  (Drivers & Constructors standings) │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Process JSON response:             │
│  - Extract driver/team info         │
│  - Look up team colors              │
│  - Sort by points                   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Pass data to template:             │
│  render_template("championship.html",
│    drivers=[...],                   │
│    constructors=[...],              │
│    year=2026)                       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Jinja2 renders HTML:               │
│  - Loop through drivers             │
│  - Use team colors for styling      │
│  - Show points and positions        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Browser displays page with:        │
│  ✓ Driver standings (colored)       │
│  ✓ Constructor standings (colored)  │
│  ✓ Tab switching between both       │
└─────────────────────────────────────┘
```

---

## 7. Key Concepts

### Ergast API Structure
All Ergast endpoints return this structure:
```json
{
  "MRData": {
    "StandingsTable": {
      "StandingsLists": [
        {
          "DriverStandings": [...]  // or "ConstructorStandings": [...]
        }
      ]
    }
  }
}
```

### Jinja2 in Templates
- `{{ variable }}` - Print variable
- `{% for item in list %}...{% endfor %}` - Loop
- `{{ loop.index }}` - Current loop counter (1-based)

### Hex Colors
- `#00D2BE` - Cyan (Mercedes)
- `#DC0000` - Red (Ferrari)
- `#1E41FF` - Blue (Red Bull)

---

## 8. Testing

### Run the app:
```bash
.venv/bin/python app.py
```

### Access:
- Home: http://localhost:5001/
- Championship: http://localhost:5001/championship
- Schedule: http://localhost:5001/schedule
- Next Race: http://localhost:5001/next

### View data:
- Browser DevTools (F12)
- Python console for logging

