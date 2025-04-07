from fastapi import FastAPI, Query, HTTPException
from typing import List, Optional
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Netflix Shows API", 
              description="API to access and query Netflix shows and movies data")

# Load the Netflix data
try:
    # Assuming you have a CSV file with Netflix data
    # Adjust the path as needed
    df = pd.read_csv("../data/netflix_titles.csv")
except Exception as e:
    print(f"Error loading data: {e}")
    # Create empty DataFrame with expected columns if file is not found
    df = pd.DataFrame(columns=["show_id", "type", "title", "director", "cast", 
                              "country", "date_added", "release_year", 
                              "rating", "duration", "listed_in", "description"])

# Define Pydantic models for data validation
class NetflixShow(BaseModel):
    show_id: str
    type: str
    title: str
    director: Optional[str] = None
    cast: Optional[str] = None
    country: Optional[str] = None
    date_added: Optional[str] = None
    release_year: Optional[int] = None
    rating: Optional[str] = None
    duration: Optional[str] = None
    listed_in: Optional[str] = None  # genres
    description: Optional[str] = None

@app.get("/", tags=["Root"])
async def root():
    return {"message": "Welcome to the Netflix Shows API"}

@app.get("/shows", response_model=List[NetflixShow], tags=["Shows"])
async def get_shows(
    show_type: Optional[str] = Query(None, description="Filter by type (Movie or TV Show)"),
    genre: Optional[str] = Query(None, description="Filter by genre"),
    year: Optional[int] = Query(None, description="Filter by release year"),
    limit: int = Query(100, description="Limit the number of results"),
    offset: int = Query(0, description="Offset for pagination")
):
    """
    Get a list of Netflix shows with optional filtering.
    """
    filtered_df = df.copy()
    
    # Apply filters
    if show_type:
        filtered_df = filtered_df[filtered_df["type"].str.lower() == show_type.lower()]
    if genre:
        filtered_df = filtered_df[filtered_df["listed_in"].str.contains(genre, case=False, na=False)]
    if year:
        filtered_df = filtered_df[filtered_df["release_year"] == year]
    
    # Pagination
    total_results = len(filtered_df)
    filtered_df = filtered_df.iloc[offset:offset+limit]
    
    # Convert to list of dicts, replacing NaN values with None for JSON serialization
    shows = filtered_df.replace({float('nan'): None}).to_dict(orient="records")
    
    return shows

@app.get("/shows/{show_id}", response_model=NetflixShow, tags=["Shows"])
async def get_show(show_id: str):
    """
    Get details for a specific Netflix show by ID.
    """
    show = df[df["show_id"] == show_id]
    if show.empty:
        raise HTTPException(status_code=404, detail=f"Show with ID {show_id} not found")
    
    return show.replace({float('nan'): None}).to_dict(orient="records")[0]

@app.get("/search", response_model=List[NetflixShow], tags=["Search"])
async def search_shows(
    query: str = Query(..., description="Search query"),
    limit: int = Query(20, description="Limit the number of results")
):
    """
    Search for Netflix shows by title or description.
    """
    results = df[
        df["title"].str.contains(query, case=False, na=False) | 
        df["description"].str.contains(query, case=False, na=False)
    ]
    
    total_results = len(results)
    results = results.head(limit)
    
    return results.replace({float('nan'): None}).to_dict(orient="records")

@app.get("/search/advanced", response_model=List[NetflixShow], tags=["Search"])
async def advanced_search(
    title: Optional[str] = Query(None, description="Search in title"),
    director: Optional[str] = Query(None, description="Search by director"),
    cast: Optional[str] = Query(None, description="Search by cast member"),
    limit: int = Query(20, description="Limit the number of results")
):
    """
    Advanced search for Netflix shows with specific fields.
    Search by title, director, or cast member.
    """
    filtered_df = df.copy()
    
    if title:
        filtered_df = filtered_df[filtered_df["title"].str.contains(title, case=False, na=False)]
    if director:
        filtered_df = filtered_df[filtered_df["director"].str.contains(director, case=False, na=False)]
    if cast:
        filtered_df = filtered_df[filtered_df["cast"].str.contains(cast, case=False, na=False)]
    
    if not any([title, director, cast]):
        # Return empty result if no search criteria provided
        return []
    
    total_results = len(filtered_df)
    filtered_df = filtered_df.head(limit)
    
    return filtered_df.replace({float('nan'): None}).to_dict(orient="records")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
