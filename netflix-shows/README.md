<think>
I need to create a README.md file for the Netflix Shows API. The user wants me to put the content directly in a file called README.md. I'll format my previous response as a markdown file for the README.md.
</think>

```markdown:README.md
# Netflix Shows API

A FastAPI-based REST API that provides access to Netflix shows and movies data. This API allows users to browse, search, and filter Netflix content programmatically.

## Table of Contents

- [Features](#features)
- [Setup](#setup)
- [API Endpoints](#api-endpoints)
- [API Examples](#api-examples)
- [Data Format](#data-format)
- [Contributing](#contributing)

## Features

- Browse all Netflix shows and movies with pagination
- Filter content by type, genre, and release year
- Search by title and description
- Advanced search by title, director, or cast
- Get detailed information about specific shows by ID
- RESTful API with JSON responses
- Interactive API documentation with Swagger UI

## Setup

### Prerequisites

- Python 3.7+
- pandas
- FastAPI
- uvicorn

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/netflix-shows.git
   cd netflix-shows
```

2. Install dependencies:

   ```bash
   pip install fastapi uvicorn pandas
   ```
3. Prepare the data:

   - Place your Netflix dataset (`netflix_titles.csv`) in the `data/` directory
   - Ensure the CSV has the expected columns: show_id, type, title, director, cast, country, date_added, release_year, rating, duration, listed_in, description
4. Run the API:

   ```bash
   cd api
   python main.py
   ```
5. Access the API documentation at http://localhost:8000/docs

## API Endpoints


| Endpoint           | Method | Description                                 |
| ------------------ | ------ | ------------------------------------------- |
| `/`                | GET    | Welcome message                             |
| `/shows`           | GET    | List all shows with optional filtering      |
| `/shows/{show_id}` | GET    | Get details for a specific show             |
| `/search`          | GET    | Search shows by title or description        |
| `/search/advanced` | GET    | Advanced search by title, director, or cast |

## API Examples

### Basic Usage

#### 1. List all shows (limited to 100)

```bash
curl http://localhost:8000/shows
```

Response:

```json
[
  {
    "show_id": "s1",
    "type": "Movie",
    "title": "Inception",
    "director": "Christopher Nolan",
    "cast": "Leonardo DiCaprio, Joseph Gordon-Levitt, Ellen Page",
    "country": "United States",
    "date_added": "January 1, 2021",
    "release_year": 2010,
    "rating": "PG-13",
    "duration": "148 min",
    "listed_in": "Action, Sci-Fi, Thriller",
    "description": "A thief who steals corporate secrets through the use of dream-sharing technology..."
  },
  // More shows...
]
```

#### 2. Get a specific show by ID

```bash
curl http://localhost:8000/shows/s1
```

Response:

```json
{
  "show_id": "s1",
  "type": "Movie",
  "title": "Inception",
  "director": "Christopher Nolan",
  "cast": "Leonardo DiCaprio, Joseph Gordon-Levitt, Ellen Page",
  "country": "United States",
  "date_added": "January 1, 2021",
  "release_year": 2010,
  "rating": "PG-13",
  "duration": "148 min",
  "listed_in": "Action, Sci-Fi, Thriller",
  "description": "A thief who steals corporate secrets through the use of dream-sharing technology..."
}
```

### Filtering

#### Filter by show type (Movies or TV Shows)

```bash
curl "http://localhost:8000/shows?show_type=Movie"
```

#### Filter by genre

```bash
curl "http://localhost:8000/shows?genre=Comedy"
```

#### Filter by release year

```bash
curl "http://localhost:8000/shows?year=2020"
```

#### Combined filters with pagination

```bash
curl "http://localhost:8000/shows?show_type=TV%20Show&genre=Drama&limit=10&offset=20"
```

### Searching

#### Simple search (title or description)

```bash
curl "http://localhost:8000/search?query=stranger"
```

#### Advanced search by title

```bash
curl "http://localhost:8000/search/advanced?title=breaking"
```

#### Advanced search by director

```bash
curl "http://localhost:8000/search/advanced?director=spielberg"
```

#### Advanced search by cast member

```bash
curl "http://localhost:8000/search/advanced?cast=pitt"
```

#### Combined advanced search

```bash
curl "http://localhost:8000/search/advanced?director=nolan&title=dark"
```

### Python Requests Example

```python
import requests

# Base URL
base_url = "http://localhost:8000"

# Get all shows
response = requests.get(f"{base_url}/shows", params={"limit": 5})
print(f"Number of shows: {len(response.json())}")
for show in response.json():
    print(f"- {show['title']} ({show['release_year']})")

# Search for shows
search_response = requests.get(f"{base_url}/search", params={"query": "stranger things"})
print(f"\nSearch results: {len(search_response.json())}")
for show in search_response.json():
    print(f"- {show['title']}")

# Advanced search
advanced_search = requests.get(
    f"{base_url}/search/advanced", 
    params={"director": "spielberg", "limit": 3}
)
print(f"\nDirector search results: {len(advanced_search.json())}")
for show in advanced_search.json():
    print(f"- {show['title']} (Directed by: {show['director']})")
```

### JavaScript Fetch Example

```javascript
// Get all TV shows
fetch('http://localhost:8000/shows?show_type=TV%20Show&limit=5')
  .then(response => response.json())
  .then(data => {
    console.log(`Found ${data.length} TV shows:`);
    data.forEach(show => console.log(`- ${show.title}`));
  });

// Search by director
fetch('http://localhost:8000/search/advanced?director=spielberg')
  .then(response => response.json())
  .then(data => {
    console.log(`\nFound ${data.length} shows directed by Spielberg:`);
    data.forEach(show => console.log(`- ${show.title} (${show.release_year})`));
  });
```

## Data Format

Each show in the Netflix dataset has the following properties:


| Property     | Type    | Description                               |
| ------------ | ------- | ----------------------------------------- |
| show_id      | string  | Unique identifier for the show            |
| type         | string  | Either "Movie" or "TV Show"               |
| title        | string  | The title of the show                     |
| director     | string  | The director(s) of the show               |
| cast         | string  | Comma-separated list of cast members      |
| country      | string  | Country or countries of origin            |
| date_added   | string  | Date when added to Netflix                |
| release_year | integer | Year of release                           |
| rating       | string  | Content rating (e.g., "PG-13", "TV-MA")   |
| duration     | string  | Duration (e.g., "148 min" or "3 Seasons") |
| listed_in    | string  | Comma-separated list of genres            |
| description  | string  | Brief description of the show             |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

```

```
