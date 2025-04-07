from fastapi import FastAPI, Query, HTTPException, Path
from typing import List, Optional, Dict, Any
import pandas as pd
from pydantic import BaseModel
from datetime import datetime
import numpy as np

app = FastAPI(
    title="Supermarket Sales API",
    description="API to access and analyze supermarket sales dataset"
)

# Load the supermarket sales data
try:
    df = pd.read_csv("../data/supermarket_sales - Sheet1.csv")
    # Convert date to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    # Add an ID column based on Invoice ID for easier reference
    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)
except Exception as e:
    print(f"Error loading data: {e}")
    # Create empty DataFrame with expected columns if file is not found
    df = pd.DataFrame(columns=[
        "id", "Invoice ID", "Branch", "City", "Customer type", "Gender", "Product line", 
        "Unit price", "Quantity", "Tax 5%", "Total", "Date", "Time", "Payment", 
        "cogs", "gross margin percentage", "gross income", "Rating"
    ])
    df["id"] = []

# Define Pydantic models for data validation
class SalesRecord(BaseModel):
    id: int
    Invoice_ID: str
    Branch: str
    City: str
    Customer_type: str
    Gender: str
    Product_line: str
    Unit_price: float
    Quantity: int
    Tax: float
    Total: float
    Date: str
    Time: str
    Payment: str
    cogs: float
    gross_margin_percentage: float
    gross_income: float
    Rating: float

class SalesStatistics(BaseModel):
    count: int
    total_revenue: float
    avg_rating: float
    total_quantity: int
    top_product_lines: Dict[str, int]
    payment_distribution: Dict[str, int]
    gender_distribution: Dict[str, int]

@app.get("/", tags=["General"])
async def root():
    """
    Get general information about the supermarket sales dataset.
    """
    return {
        "message": "Supermarket Sales API",
        "total_records": len(df),
        "date_range": {
            "start": df["Date"].min().strftime("%Y-%m-%d") if not df.empty else None,
            "end": df["Date"].max().strftime("%Y-%m-%d") if not df.empty else None
        },
        "branches": sorted(df["Branch"].unique().tolist()) if not df.empty else [],
        "product_lines": sorted(df["Product line"].unique().tolist()) if not df.empty else [],
        "payment_methods": sorted(df["Payment"].unique().tolist()) if not df.empty else []
    }

@app.get("/sales", tags=["Sales Records"])
async def get_sales(
    branch: Optional[str] = Query(None, description="Filter by branch (A, B, C)"),
    city: Optional[str] = Query(None, description="Filter by city"),
    customer_type: Optional[str] = Query(None, description="Filter by customer type (Member, Normal)"),
    gender: Optional[str] = Query(None, description="Filter by gender (Male, Female)"),
    product_line: Optional[str] = Query(None, description="Filter by product line"),
    payment: Optional[str] = Query(None, description="Filter by payment method"),
    min_date: Optional[str] = Query(None, description="Minimum date (YYYY-MM-DD)"),
    max_date: Optional[str] = Query(None, description="Maximum date (YYYY-MM-DD)"),
    min_rating: Optional[float] = Query(None, description="Minimum rating"),
    max_rating: Optional[float] = Query(None, description="Maximum rating"),
    limit: int = Query(20, description="Limit the number of results"),
    offset: int = Query(0, description="Skip the first N results")
):
    """
    Get sales records with optional filtering.
    """
    filtered_df = df.copy()
    
    # Apply filters
    if branch:
        filtered_df = filtered_df[filtered_df["Branch"] == branch]
    if city:
        filtered_df = filtered_df[filtered_df["City"] == city]
    if customer_type:
        filtered_df = filtered_df[filtered_df["Customer type"] == customer_type]
    if gender:
        filtered_df = filtered_df[filtered_df["Gender"] == gender]
    if product_line:
        filtered_df = filtered_df[filtered_df["Product line"] == product_line]
    if payment:
        filtered_df = filtered_df[filtered_df["Payment"] == payment]
    if min_date:
        try:
            min_date_dt = pd.to_datetime(min_date)
            filtered_df = filtered_df[filtered_df["Date"] >= min_date_dt]
        except:
            raise HTTPException(status_code=400, detail="Invalid min_date format. Use YYYY-MM-DD.")
    if max_date:
        try:
            max_date_dt = pd.to_datetime(max_date)
            filtered_df = filtered_df[filtered_df["Date"] <= max_date_dt]
        except:
            raise HTTPException(status_code=400, detail="Invalid max_date format. Use YYYY-MM-DD.")
    if min_rating is not None:
        filtered_df = filtered_df[filtered_df["Rating"] >= min_rating]
    if max_rating is not None:
        filtered_df = filtered_df[filtered_df["Rating"] <= max_rating]
    
    # Get total count before pagination
    total_count = len(filtered_df)
    
    # Apply pagination
    filtered_df = filtered_df.iloc[offset:offset+limit]
    
    # Convert datetime to string for JSON serialization
    filtered_df["Date"] = filtered_df["Date"].dt.strftime("%Y-%m-%d")
    
    # Convert to list of dicts
    records = filtered_df.replace({float('nan'): None}).to_dict(orient="records")
    
    return {
        "total_count": total_count,
        "results": records
    }

@app.get("/sales/{sale_id}", tags=["Sales Records"])
async def get_sale_by_id(
    sale_id: int = Path(..., description="The ID of the sale record")
):
    """
    Get a specific sale record by ID.
    """
    if sale_id < 1 or sale_id > len(df):
        raise HTTPException(status_code=404, detail="Sale record not found")
    
    record = df.iloc[sale_id-1].copy()
    
    # Convert datetime to string for JSON serialization
    if isinstance(record["Date"], pd.Timestamp):
        record["Date"] = record["Date"].strftime("%Y-%m-%d")
    
    # Convert to dict
    record_dict = record.replace({float('nan'): None}).to_dict()
    
    return record_dict

@app.get("/statistics", tags=["Analysis"])
async def get_statistics(
    branch: Optional[str] = Query(None, description="Filter by branch"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """
    Get statistical summary of sales data.
    """
    filtered_df = df.copy()
    
    # Apply filters
    if branch:
        filtered_df = filtered_df[filtered_df["Branch"] == branch]
    if start_date:
        try:
            start_date_dt = pd.to_datetime(start_date)
            filtered_df = filtered_df[filtered_df["Date"] >= start_date_dt]
        except:
            raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD.")
    if end_date:
        try:
            end_date_dt = pd.to_datetime(end_date)
            filtered_df = filtered_df[filtered_df["Date"] <= end_date_dt]
        except:
            raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD.")
    
    # Calculate statistics
    stats = {
        "count": len(filtered_df),
        "total_revenue": round(filtered_df["Total"].sum(), 2),
        "avg_revenue_per_sale": round(filtered_df["Total"].mean(), 2),
        "avg_rating": round(filtered_df["Rating"].mean(), 2),
        "total_quantity": int(filtered_df["Quantity"].sum()),
        "total_gross_income": round(filtered_df["gross income"].sum(), 2),
        "avg_gross_income": round(filtered_df["gross income"].mean(), 2),
        "payment_distribution": filtered_df["Payment"].value_counts().to_dict(),
        "gender_distribution": filtered_df["Gender"].value_counts().to_dict(),
        "customer_type_distribution": filtered_df["Customer type"].value_counts().to_dict(),
        "product_line_counts": filtered_df["Product line"].value_counts().to_dict(),
        "branch_counts": filtered_df["Branch"].value_counts().to_dict(),
        "city_counts": filtered_df["City"].value_counts().to_dict()
    }
    
    return stats

@app.get("/analysis/product-lines", tags=["Analysis"])
async def analyze_product_lines():
    """
    Analyze sales, revenue, and ratings by product line.
    """
    analysis = []
    
    for product_line in df["Product line"].unique():
        product_df = df[df["Product line"] == product_line]
        
        analysis.append({
            "product_line": product_line,
            "total_sales": len(product_df),
            "total_revenue": round(product_df["Total"].sum(), 2),
            "avg_revenue_per_sale": round(product_df["Total"].mean(), 2),
            "total_quantity": int(product_df["Quantity"].sum()),
            "avg_quantity_per_sale": round(product_df["Quantity"].mean(), 2),
            "total_gross_income": round(product_df["gross income"].sum(), 2),
            "avg_rating": round(product_df["Rating"].mean(), 2),
            "gender_distribution": product_df["Gender"].value_counts().to_dict(),
            "customer_type_distribution": product_df["Customer type"].value_counts().to_dict(),
            "payment_distribution": product_df["Payment"].value_counts().to_dict()
        })
    
    # Sort by total revenue
    analysis.sort(key=lambda x: x["total_revenue"], reverse=True)
    
    return analysis

@app.get("/analysis/sales-by-date", tags=["Analysis"])
async def analyze_sales_by_date(
    groupby: str = Query("day", description="Group by 'day', 'month', or 'year'")
):
    """
    Analyze sales trends by date.
    """
    # Make a copy of the dataframe and ensure Date is datetime
    temp_df = df.copy()
    temp_df['Date'] = pd.to_datetime(temp_df['Date'])
    
    # Group by different time periods
    if groupby.lower() == "month":
        temp_df['period'] = temp_df['Date'].dt.to_period('M')
        date_format = "%Y-%m"
    elif groupby.lower() == "year":
        temp_df['period'] = temp_df['Date'].dt.to_period('Y')
        date_format = "%Y"
    else:  # default to day
        temp_df['period'] = temp_df['Date'].dt.to_period('D')
        date_format = "%Y-%m-%d"
    
    # Group by period and calculate aggregates
    grouped = temp_df.groupby('period').agg({
        'Total': 'sum',
        'Quantity': 'sum',
        'gross income': 'sum',
        'Rating': 'mean',
        'Invoice ID': 'count'  # count of sales
    }).reset_index()
    
    # Convert period to string for JSON serialization
    grouped['period'] = grouped['period'].astype(str)
    
    # Rename columns for clarity
    grouped = grouped.rename(columns={
        'period': 'date',
        'Invoice ID': 'sales_count',
        'gross income': 'gross_income'
    })
    
    # Round numeric columns
    for col in ['Total', 'gross_income', 'Rating']:
        grouped[col] = grouped[col].round(2)
    
    # Convert to records
    result = grouped.to_dict(orient='records')
    
    # Sort by date
    result.sort(key=lambda x: x['date'])
    
    return result

@app.get("/analysis/payment-methods", tags=["Analysis"])
async def analyze_payment_methods():
    """
    Analyze sales and customer behavior by payment method.
    """
    analysis = []
    
    for payment in df["Payment"].unique():
        payment_df = df[df["Payment"] == payment]
        
        analysis.append({
            "payment_method": payment,
            "total_sales": len(payment_df),
            "total_revenue": round(payment_df["Total"].sum(), 2),
            "avg_revenue_per_sale": round(payment_df["Total"].mean(), 2),
            "avg_rating": round(payment_df["Rating"].mean(), 2),
            "product_line_distribution": payment_df["Product line"].value_counts().to_dict(),
            "gender_distribution": payment_df["Gender"].value_counts().to_dict(),
            "customer_type_distribution": payment_df["Customer type"].value_counts().to_dict(),
            "branch_distribution": payment_df["Branch"].value_counts().to_dict()
        })
    
    # Sort by total sales
    analysis.sort(key=lambda x: x["total_sales"], reverse=True)
    
    return analysis

@app.get("/analysis/customer-segments", tags=["Analysis"])
async def analyze_customer_segments():
    """
    Analyze customer segments by gender and membership status.
    """
    segments = []
    
    # Group by gender and customer type
    for gender in df["Gender"].unique():
        for customer_type in df["Customer type"].unique():
            segment_df = df[(df["Gender"] == gender) & (df["Customer type"] == customer_type)]
            
            segments.append({
                "segment": f"{gender} {customer_type}",
                "gender": gender,
                "customer_type": customer_type,
                "count": len(segment_df),
                "total_revenue": round(segment_df["Total"].sum(), 2),
                "avg_revenue_per_customer": round(segment_df["Total"].mean(), 2),
                "avg_rating": round(segment_df["Rating"].mean(), 2),
                "product_preferences": segment_df["Product line"].value_counts().to_dict(),
                "payment_preferences": segment_df["Payment"].value_counts().to_dict(),
                "branch_distribution": segment_df["Branch"].value_counts().to_dict()
            })
    
    # Sort by count
    segments.sort(key=lambda x: x["count"], reverse=True)
    
    return segments

@app.get("/analysis/branch-performance", tags=["Analysis"])
async def analyze_branch_performance():
    """
    Compare performance metrics across different branches.
    """
    branches = []
    
    for branch in df["Branch"].unique():
        branch_df = df[df["Branch"] == branch]
        
        # Get city for this branch
        city = branch_df["City"].iloc[0] if len(branch_df) > 0 else "Unknown"
        
        branches.append({
            "branch": branch,
            "city": city,
            "total_sales": len(branch_df),
            "total_revenue": round(branch_df["Total"].sum(), 2),
            "avg_revenue_per_sale": round(branch_df["Total"].mean(), 2),
            "total_gross_income": round(branch_df["gross income"].sum(), 2),
            "avg_rating": round(branch_df["Rating"].mean(), 2),
            "product_mix": branch_df["Product line"].value_counts().to_dict(),
            "customer_mix": {
                "gender": branch_df["Gender"].value_counts().to_dict(),
                "customer_type": branch_df["Customer type"].value_counts().to_dict()
            },
            "payment_mix": branch_df["Payment"].value_counts().to_dict()
        })
    
    # Sort by total revenue
    branches.sort(key=lambda x: x["total_revenue"], reverse=True)
    
    return branches

@app.get("/analysis/ratings", tags=["Analysis"])
async def analyze_ratings(
    bin_size: float = Query(0.5, description="Size of rating bins")
):
    """
    Analyze customer ratings and correlations with other factors.
    """
    # Round ratings to the specified bin size
    df['RatingBin'] = (df['Rating'] / bin_size).round() * bin_size
    
    # Group by rating bin
    rating_analysis = df.groupby('RatingBin').agg({
        'Invoice ID': 'count',
        'Total': 'mean',
        'Quantity': 'mean',
        'gross income': 'mean'
    }).reset_index()
    
    # Rename columns
    rating_analysis = rating_analysis.rename(columns={
        'Invoice ID': 'count',
        'Total': 'avg_total',
        'Quantity': 'avg_quantity',
        'gross income': 'avg_gross_income'
    })
    
    # Round numeric columns
    for col in ['avg_total', 'avg_quantity', 'avg_gross_income']:
        rating_analysis[col] = rating_analysis[col].round(2)
    
    # Convert to records
    binned_ratings = rating_analysis.to_dict(orient='records')
    
    # Calculate correlations with rating
    correlations = {
        'total_vs_rating': df[['Total', 'Rating']].corr().iloc[0, 1],
        'quantity_vs_rating': df[['Quantity', 'Rating']].corr().iloc[0, 1],
        'gross_income_vs_rating': df[['gross income', 'Rating']].corr().iloc[0, 1]
    }
    
    # Analyze ratings by product line
    product_ratings = df.groupby('Product line')['Rating'].agg(['mean', 'count']).reset_index()
    product_ratings['mean'] = product_ratings['mean'].round(2)
    product_ratings = product_ratings.rename(columns={'mean': 'avg_rating'})
    product_ratings = product_ratings.sort_values('avg_rating', ascending=False)
    
    # Analyze ratings by other factors
    ratings_by_factor = {
        'by_gender': df.groupby('Gender')['Rating'].mean().round(2).to_dict(),
        'by_customer_type': df.groupby('Customer type')['Rating'].mean().round(2).to_dict(),
        'by_branch': df.groupby('Branch')['Rating'].mean().round(2).to_dict(),
        'by_payment': df.groupby('Payment')['Rating'].mean().round(2).to_dict()
    }
    
    return {
        'binned_ratings': binned_ratings,
        'correlations': correlations,
        'product_line_ratings': product_ratings.to_dict(orient='records'),
        'ratings_by_factor': ratings_by_factor
    }

@app.get("/search", tags=["Search"])
async def search_sales(
    query: Optional[str] = Query(None, description="Search term for invoice ID"),
    product_line: Optional[str] = Query(None, description="Filter by product line"),
    min_price: Optional[float] = Query(None, description="Minimum unit price"),
    max_price: Optional[float] = Query(None, description="Maximum unit price"),
    min_quantity: Optional[int] = Query(None, description="Minimum quantity"),
    max_quantity: Optional[int] = Query(None, description="Maximum quantity"),
    payment_method: Optional[str] = Query(None, description="Filter by payment method"),
    branch: Optional[str] = Query(None, description="Filter by branch"),
    gender: Optional[str] = Query(None, description="Filter by customer gender"),
    customer_type: Optional[str] = Query(None, description="Filter by customer type"),
    min_rating: Optional[float] = Query(None, description="Minimum rating"),
    max_rating: Optional[float] = Query(None, description="Maximum rating"),
    min_date: Optional[str] = Query(None, description="Minimum date (YYYY-MM-DD)"),
    max_date: Optional[str] = Query(None, description="Maximum date (YYYY-MM-DD)"),
    limit: int = Query(20, description="Limit the number of results")
):
    """
    Advanced search for sales records based on multiple criteria.
    """
    filtered_df = df.copy()
    
    # Apply all the filters that are provided
    if query:
        filtered_df = filtered_df[filtered_df["Invoice ID"].str.contains(query, case=False)]
    if product_line:
        filtered_df = filtered_df[filtered_df["Product line"] == product_line]
    if min_price is not None:
        filtered_df = filtered_df[filtered_df["Unit price"] >= min_price]
    if max_price is not None:
        filtered_df = filtered_df[filtered_df["Unit price"] <= max_price]
    if min_quantity is not None:
        filtered_df = filtered_df[filtered_df["Quantity"] >= min_quantity]
    if max_quantity is not None:
        filtered_df = filtered_df[filtered_df["Quantity"] <= max_quantity]
    if payment_method:
        filtered_df = filtered_df[filtered_df["Payment"] == payment_method]
    if branch:
        filtered_df = filtered_df[filtered_df["Branch"] == branch]
    if gender:
        filtered_df = filtered_df[filtered_df["Gender"] == gender]
    if customer_type:
        filtered_df = filtered_df[filtered_df["Customer type"] == customer_type]
    if min_rating is not None:
        filtered_df = filtered_df[filtered_df["Rating"] >= min_rating]
    if max_rating is not None:
        filtered_df = filtered_df[filtered_df["Rating"] <= max_rating]
    if min_date:
        try:
            min_date_dt = pd.to_datetime(min_date)
            filtered_df = filtered_df[filtered_df["Date"] >= min_date_dt]
        except:
            raise HTTPException(status_code=400, detail="Invalid min_date format. Use YYYY-MM-DD.")
    if max_date:
        try:
            max_date_dt = pd.to_datetime(max_date)
            filtered_df = filtered_df[filtered_df["Date"] <= max_date_dt]
        except:
            raise HTTPException(status_code=400, detail="Invalid max_date format. Use YYYY-MM-DD.")
    
    # Get total count before limiting
    total_count = len(filtered_df)
    
    # Apply limit
    filtered_df = filtered_df.head(limit)
    
    # Convert datetime to string for JSON serialization
    filtered_df["Date"] = filtered_df["Date"].dt.strftime("%Y-%m-%d")
    
    # Convert to list of dicts
    records = filtered_df.replace({float('nan'): None}).to_dict(orient="records")
    
    return {
        "total_count": total_count,
        "results": records
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
