# F1Tracker Implementation Guide

## What Has Been Implemented

### 1. **OpenF1/Ergast API Integration**

**What it means:** Your app now uses the **Ergast API** (which is free and doesn't require API keys) instead of the Formula 1 API.

**Key files:**
- `app.py` lines 213-218: `get_openf1_json()` function

**How it works:**
```python
def get_openf1_json(url):
    response = requests.get(url)  # Make HTTP request
    response.raise_for_status()   # Raise error if request fails
    return response.json()        # Return JSON data
```

**API Endpoints used:**
- Drivers standings: `https://ergast.com/api/f1/{year}/driverStandings.json`
- Constructors standings: `https://ergast.com/api/f1/{year}/constructorStandings.json`

---

### 2. **Championship Page with Standings**

**What it means:** New page at `/championship` showing current drivers and constructors standings.

**Key files:**
- `app.py` lines 444-482: `@app.route("/championship")` function
- `templates/championship.html`: Template with tabs for Drivers/Constructors

**How it works:**

1. **Fetch data from Ergast API:**
   ```python
   drv_url = f"https://ergast.com/api/f1/{year}/driverStandings.json"
   drv_data = get_openf1_json(drv_url)
   ```

2. **Parse the response** into a list of dictionaries:
   ```python
   drivers = [
       {
           "name": f"{d['Driver']['givenName']} {d['Driver']['familyName']}",
           "team": d['Constructors'][0]['name'],
           "team_color": TEAM_DATA.get(d['Constructors'][0]['name'], {}).get('color', '#9aa6b2'),
           "points": d['points']
       }
       for d in drivers_raw
   ]
   ```

3. **Pass to template** and render with team colors

---

### 3. **Team Colors Dictionary**

**What it means:** Each F1 team has an official brand color used in the UI.

**Location:** `app.py` lines 35-47

```python
TEAM_DATA = {
    "Mercedes": {"color": "#00D2BE", "logo": "mercedes.png"},    # Cyan/Turquoise
    "Red Bull": {"color": "#1E41FF", "logo": "redbull.png"},      # Dark Blue
    "Ferrari": {"color": "#DC0000", "logo": "ferrari.png"},       # Red
    "McLaren": {"color": "#FF8700", "logo": "mclaren.png"},       # Orange
    # ... etc
}
```

**Why it matters:**
- The left border of each team's standings entry uses their brand color
- Makes the championship page visually distinctive
- Easy to recognize teams at a glance

---

### 4. **Template Updates**

**Championship.html improvements:**

1. **Removed broken image references** (driver headshots and logos)
   - Was trying to load images that don't exist yet
   - Fixed to use team colors instead

2. **Added tab switching** for Drivers vs Constructors
   ```html
   <button id="driversBtn" class="pill">Drivers</button>
   <button id="constructorsBtn" class="pill">Constructors</button>
   ```

3. **Styled standings with team colors:**
   ```html
   <li style="border-left: 4px solid {{ d.team_color }}; background: rgba(255,255,255,0.02);">
   ```

---

### 5. **Navigation Updates**

**Updated:** `templates/index.html`

Added Championship link to all navigation bars:
```html
<a href="{{ url_for('championship') }}"><span class="icon">🏆</span>Championship</a>
```

This appears in:
- Home page navigation
- Schedule page navigation
- Next race page navigation

---

## How to Use

### View Championship Standings
1. Click "Championship" (🏆) in the navigation
2. See current drivers standings with points and team colors
3. Click "Constructors" tab to see team standings

### Data Auto-Updates
- Data comes directly from Ergast API
- No caching needed (API is free and fast)
- Always shows current standings for the current year

---

## Features Ready to Add (Optional)

### 1. **Year Selector** (Not yet implemented)
Choose to view standings from previous years (2020-2025)

```python
year = request.args.get("year", pd.Timestamp.today().year, type=int)
```

### 2. **Driver Headshots & Team Logos** (Not yet implemented)
Requires adding images to:
- `static/images/drivers/` (e.g., `hamilton.png`)
- `static/images/teams/` (e.g., `mercedes.png`)

### 3. **Weekly Auto-Refresh** (Not yet implemented)
```javascript
setTimeout(() => {
    location.reload();
}, 604800000); // 7 days in milliseconds
```

### 4. **Stats Sidebar** (Not yet implemented)
Display quick stats:
- Total drivers in championship
- Total teams
- Next race info

---

## Troubleshooting

### Issue: "Import flask could not be resolved"
**Solution:** Linting error only. Packages are installed in `.venv/bin/`.

### Issue: Championship page shows no data
**Check:**
1. Server is running: `.venv/bin/python app.py`
2. Internet connection (API calls need network)
3. Python console for errors

### Issue: Colors look wrong
**Check:**
- Hex color codes in `TEAM_DATA` dictionary
- Browser cache (try Ctrl+Shift+R hard refresh)

---

## File Structure

```
F1Tracker/
├── app.py                           # Main Flask app with championship route
├── templates/
│   ├── index.html                   # Home page (updated with nav link)
│   ├── championship.html            # NEW: Championship standings page
│   ├── schedule.html
│   └── next_race.html
├── static/
│   ├── style.css
│   └── images/                      # Ready for: drivers/ and teams/
└── cache/
    └── generated/                   # Track image cache
```

---

## Next Steps

1. ✅ **Championship page works** - Test it out!
2. 📸 **Add driver/team images** (optional, improves UI)
3. 📅 **Add year selector** for historical standings
4. 🔄 **Add auto-refresh** for live updates
5. 📊 **Add stats sidebar** with championship info

