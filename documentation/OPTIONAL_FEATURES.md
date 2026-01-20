# Optional Features to Add Next

## Feature 1: Year Selector for Historical Standings

**What it does:** Let users view championship standings from any year (2020-current)

### Implementation Steps:

#### 1. Update the championship route to accept year parameter:

```python
@app.route("/championship")
def championship():
    year = request.args.get("year", pd.Timestamp.today().year, type=int)
    # NEW: ↑ Get year from URL query string (e.g., ?year=2023)
    
    # ... rest of the code stays the same
    
    current_year = pd.Timestamp.today().year
    available_years = list(range(2020, current_year + 1))
    # NEW: ↑ Create list of years [2020, 2021, ..., 2026]
    
    return render_template(
        "championship.html",
        drivers=drivers,
        constructors=constructors,
        year=year,
        available_years=available_years  # NEW: Pass to template
    )
```

#### 2. Add year dropdown to championship.html:

```html
<div style="margin-top:16px; display:flex; gap:12px; align-items:center;">
    <label for="yearSelect">Year:</label>
    <select id="yearSelect" style="padding: 8px; border-radius: 6px; background: var(--glass);">
        {% for y in available_years %}
            <option value="{{ y }}" {% if y == year %}selected{% endif %}>{{ y }}</option>
        {% endfor %}
    </select>
    
    <button id="driversBtn" class="pill">Drivers</button>
    <button id="constructorsBtn" class="pill">Constructors</button>
</div>

<script>
    document.getElementById("yearSelect").addEventListener("change", function() {
        const year = this.value;
        window.location.href = `/championship?year=${year}`;
        // Reloads page with new year
    });
</script>
```

---

## Feature 2: Driver Headshots & Team Logos

**What it does:** Display small images of drivers and team logos

### Implementation Steps:

#### 1. Create folder structure:
```
static/
├── images/
│   ├── drivers/
│   │   ├── hamilton.png (100x120px)
│   │   ├── verstappen.png
│   │   └── ... (one per driver)
│   └── teams/
│       ├── mercedes.png (50x50px)
│       ├── ferrari.png
│       └── ... (one per team)
```

#### 2. Get images from:
- **Official F1 website:** formula1.com/en/drivers
- **Team websites:** mercedes-amg.com, redbullracing.com, etc.
- **Wikipedia:** en.wikipedia.org/wiki/Formula_One (has images)

#### 3. Update championship.html template:

```html
{% for d in drivers %}
<li style="display: flex; align-items: center; gap: 12px; padding: 12px; border-left: 4px solid {{ d.team_color }};">
    <!-- Driver image -->
    <img src="{{ url_for('static', filename='images/drivers/' + d.name.split(' ')[-1].lower() + '.png') }}"
         alt="{{ d.name }}" width="40" height="50" style="border-radius: 4px;">
    
    <!-- Driver info -->
    <div style="flex: 1;">
        <strong>{{ loop.index }}. {{ d.name }}</strong><br>
        <span style="color: var(--muted); font-size: 0.9rem;">{{ d.team }}</span>
    </div>
    
    <!-- Points & team logo -->
    <div style="text-align: right;">
        <span style="font-weight: bold; font-size: 1.2rem;">{{ d.points }}</span>
        <img src="{{ url_for('static', filename='images/teams/' + d.team.lower().replace(' ', '_') + '.png') }}"
             alt="{{ d.team }}" width="30" height="30" style="margin-top: 4px;">
    </div>
</li>
{% endfor %}
```

---

## Feature 3: Weekly Auto-Refresh

**What it does:** Automatically refresh standings every 7 days

### Implementation Steps:

#### 1. Add to championship.html (inside `<script>` tag):

```javascript
// Auto-refresh every 7 days (604800000 ms)
const REFRESH_INTERVAL = 604800000; // 7 days in milliseconds

setTimeout(() => {
    console.log("Weekly refresh triggered");
    location.reload();
}, REFRESH_INTERVAL);

// Optional: Show countdown timer in console
const daysRemaining = Math.floor(REFRESH_INTERVAL / (1000 * 60 * 60 * 24));
console.log(`Next refresh in ${daysRemaining} days`);
```

#### 2. Alternative: Server-side weekly cache clear:

```python
import time
from datetime import datetime

# Add to app.py
CACHE_EXPIRY = 604800  # 7 days in seconds

def should_refresh_standings(cache_timestamp):
    """Check if standings cache is older than 7 days"""
    current_time = time.time()
    return (current_time - cache_timestamp) > CACHE_EXPIRY

# In championship route:
cache_file = 'cache/standings_timestamp.txt'
if os.path.exists(cache_file):
    with open(cache_file, 'r') as f:
        timestamp = float(f.read())
    
    if not should_refresh_standings(timestamp):
        # Use cached data (faster)
        pass
    else:
        # Refresh from API
        pass
```

---

## Feature 4: Stats Sidebar

**What it does:** Display quick stats about the championship

### Implementation Steps:

#### 1. Add to championship route:

```python
# After fetching drivers and constructors data:
next_race_info = {
    "name": "TBD",
    "date": "TBD"
}

try:
    schedule, _ = load_calendar()
    next_event = get_next_event(schedule)
    next_race_info = {
        "name": next_event["EventName"],
        "date": next_event["EventDate"]
    }
except:
    pass

stats = {
    "total_drivers": len(drivers),
    "total_teams": len(constructors),
    "next_race": next_race_info,
    "leader_name": drivers[0]["name"] if drivers else "N/A",
    "leader_points": drivers[0]["points"] if drivers else 0
}

return render_template(
    "championship.html",
    drivers=drivers,
    constructors=constructors,
    year=year,
    stats=stats  # NEW
)
```

#### 2. Add to championship.html (inside main):

```html
<aside class="stats-sidebar">
    <h3>Quick Stats</h3>
    
    <div class="stat-item">
        <span class="stat-label">Leading Driver:</span>
        <span class="stat-value">{{ stats.leader_name }}</span>
    </div>
    
    <div class="stat-item">
        <span class="stat-label">Points:</span>
        <span class="stat-value">{{ stats.leader_points }}</span>
    </div>
    
    <div class="stat-item">
        <span class="stat-label">Total Drivers:</span>
        <span class="stat-value">{{ stats.total_drivers }}</span>
    </div>
    
    <div class="stat-item">
        <span class="stat-label">Total Teams:</span>
        <span class="stat-value">{{ stats.total_teams }}</span>
    </div>
    
    <div class="stat-item">
        <span class="stat-label">Next Race:</span>
        <span class="stat-value">{{ stats.next_race.name }}</span>
    </div>
</aside>
```

#### 3. Add CSS to style.css:

```css
.stats-sidebar {
    position: fixed;
    top: 120px;
    right: 20px;
    width: 220px;
    background: var(--glass);
    backdrop-filter: blur(var(--glass-blur));
    padding: 16px;
    border-radius: var(--radius);
    border: 1px solid rgba(255,255,255,0.05);
    box-shadow: 0 8px 30px rgba(2,6,23,0.6);
}

.stats-sidebar h3 {
    margin-top: 0;
    color: #fff;
    font-size: 1.1rem;
}

.stat-item {
    display: flex;
    flex-direction: column;
    margin: 12px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

.stat-item:last-child {
    border-bottom: none;
}

.stat-label {
    color: var(--muted);
    font-size: 0.85rem;
    font-weight: 600;
}

.stat-value {
    color: #fff;
    font-weight: 700;
    margin-top: 4px;
    font-size: 1rem;
}

@media (max-width: 1200px) {
    .stats-sidebar {
        display: none;
    }
}
```

---

## Feature 5: Live Points Graph

**What it does:** Show how championship points changed over the season

### Implementation Steps:

#### 1. Add new route:

```python
@app.route("/championship/points-history")
def points_history():
    year = request.args.get("year", pd.Timestamp.today().year, type=int)
    
    # Fetch all races for the year
    races_url = f"https://ergast.com/api/f1/{year}.json"
    races_data = get_openf1_json(races_url)
    races = races_data['MRData']['RaceTable']['Races']
    
    # For each race, fetch standings at that point
    history = []
    for race in races:
        round_num = race['round']
        standings_url = f"https://ergast.com/api/f1/{year}/{round_num}/driverStandings.json"
        standings_data = get_openf1_json(standings_url)
        
        round_standings = {
            "race": race['raceName'],
            "round": round_num,
            "drivers": [...]
        }
        history.append(round_standings)
    
    return render_template("points_history.html", history=history, year=year)
```

#### 2. Use Chart.js library for visualization:

```html
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<canvas id="pointsChart"></canvas>

<script>
    const ctx = document.getElementById('pointsChart').getContext('2d');
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Race 1', 'Race 2', 'Race 3', ...],
            datasets: [
                {
                    label: 'Driver 1',
                    data: [25, 50, 75, ...],
                    borderColor: '#00D2BE',
                    tension: 0.1
                },
                // ... more drivers
            ]
        }
    });
</script>
```

---

## Implementation Priority

**Easy (Start here):**
1. Year selector
2. Stats sidebar

**Medium:**
3. Driver headshots & logos
4. Weekly auto-refresh

**Complex:**
5. Points history graph

---

## Testing Each Feature

```bash
# Start server
.venv/bin/python app.py

# Test championship
http://localhost:5001/championship

# Test with year selector (when implemented)
http://localhost:5001/championship?year=2023
```

