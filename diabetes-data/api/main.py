from fastapi import FastAPI, Query, HTTPException, Path
from typing import List, Optional, Dict, Any
import pandas as pd
from pydantic import BaseModel
from statistics import mean, median, stdev
import numpy as np

app = FastAPI(
    title="Diabetes API",
    description="API to access and analyze diabetes dataset"
)

# Load the diabetes data
try:
    df = pd.read_csv("../data/diabetes.csv")
    # Add an ID column if not present (for referencing individual records)
    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)
except Exception as e:
    print(f"Error loading data: {e}")
    # Create empty DataFrame with expected columns if file is not found
    df = pd.DataFrame(columns=[
        "id", "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ])
    df["id"] = []

# Define Pydantic models for data validation
class DiabetesRecord(BaseModel):
    id: int
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int
    Outcome: int

class DiabetesStatistics(BaseModel):
    count: int
    mean: Dict[str, float]
    median: Dict[str, float]
    std: Dict[str, float]
    min: Dict[str, float]
    max: Dict[str, float]

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to the Diabetes Dataset API",
        "dataset_info": {
            "rows": len(df),
            "columns": list(df.columns),
            "positive_cases": int(df["Outcome"].sum()),
            "negative_cases": int(len(df) - df["Outcome"].sum())
        }
    }

@app.get("/records", response_model=List[DiabetesRecord], tags=["Records"])
async def get_records(
    min_age: Optional[int] = Query(None, description="Minimum age"),
    max_age: Optional[int] = Query(None, description="Maximum age"),
    min_bmi: Optional[float] = Query(None, description="Minimum BMI"),
    max_bmi: Optional[float] = Query(None, description="Maximum BMI"),
    outcome: Optional[int] = Query(None, description="Diabetes outcome (0 or 1)"),
    min_glucose: Optional[int] = Query(None, description="Minimum glucose level"),
    limit: int = Query(100, description="Limit the number of results"),
    offset: int = Query(0, description="Offset for pagination")
):
    """
    Get diabetes records with optional filtering.
    """
    filtered_df = df.copy()
    
    # Apply filters
    if min_age is not None:
        filtered_df = filtered_df[filtered_df["Age"] >= min_age]
    if max_age is not None:
        filtered_df = filtered_df[filtered_df["Age"] <= max_age]
    if min_bmi is not None:
        filtered_df = filtered_df[filtered_df["BMI"] >= min_bmi]
    if max_bmi is not None:
        filtered_df = filtered_df[filtered_df["BMI"] <= max_bmi]
    if outcome is not None:
        filtered_df = filtered_df[filtered_df["Outcome"] == outcome]
    if min_glucose is not None:
        filtered_df = filtered_df[filtered_df["Glucose"] >= min_glucose]
    
    # Pagination
    total_results = len(filtered_df)
    filtered_df = filtered_df.iloc[offset:offset+limit]
    
    # Convert to list of dicts
    records = filtered_df.replace({float('nan'): None}).to_dict(orient="records")
    
    return records

@app.get("/records/{record_id}", response_model=DiabetesRecord, tags=["Records"])
async def get_record(record_id: int = Path(..., description="The ID of the record to retrieve")):
    """
    Get a specific diabetes record by ID.
    """
    record = df[df["id"] == record_id]
    if record.empty:
        raise HTTPException(status_code=404, detail=f"Record with ID {record_id} not found")
    
    return record.replace({float('nan'): None}).to_dict(orient="records")[0]

@app.get("/statistics", response_model=DiabetesStatistics, tags=["Analysis"])
async def get_statistics(
    outcome: Optional[int] = Query(None, description="Filter statistics by outcome (0 or 1)")
):
    """
    Get statistical summary of the diabetes dataset.
    Optional filtering by outcome (diabetic vs non-diabetic).
    """
    stats_df = df.copy()
    
    if outcome is not None:
        stats_df = stats_df[stats_df["Outcome"] == outcome]
    
    if stats_df.empty:
        raise HTTPException(status_code=404, detail="No records found with the specified criteria")
    
    # Calculate statistics
    numeric_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                     "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    
    result = {
        "count": len(stats_df),
        "mean": {},
        "median": {},
        "std": {},
        "min": {},
        "max": {}
    }
    
    for col in numeric_cols:
        result["mean"][col] = float(stats_df[col].mean())
        result["median"][col] = float(stats_df[col].median())
        result["std"][col] = float(stats_df[col].std())
        result["min"][col] = float(stats_df[col].min())
        result["max"][col] = float(stats_df[col].max())
    
    return result

@app.get("/analysis/age-groups", tags=["Analysis"])
async def analyze_by_age_groups(
    bin_size: int = Query(10, description="Size of age bins (e.g., 10 for decades)")
):
    """
    Analyze diabetes outcomes by age groups.
    Returns the count and percentage of diabetes cases in each age group.
    """
    # Create age bins
    df["AgeGroup"] = pd.cut(df["Age"], 
                           bins=range(0, df["Age"].max() + bin_size, bin_size),
                           right=False,
                           labels=[f"{i}-{i+bin_size-1}" for i in range(0, df["Age"].max(), bin_size)])
    
    # Group by age and outcome
    result = []
    for name, group in df.groupby("AgeGroup"):
        total = len(group)
        diabetic = group["Outcome"].sum()
        result.append({
            "age_group": str(name),
            "total_count": int(total),
            "diabetic_count": int(diabetic),
            "non_diabetic_count": int(total - diabetic),
            "diabetic_percentage": round(diabetic / total * 100, 2) if total > 0 else 0
        })
    
    return result

@app.get("/analysis/bmi-groups", tags=["Analysis"])
async def analyze_by_bmi_groups():
    """
    Analyze diabetes outcomes by BMI categories.
    Returns the count and percentage of diabetes cases in each BMI category.
    """
    # Define BMI categories
    def bmi_category(bmi):
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    
    df["BMICategory"] = df["BMI"].apply(bmi_category)
    
    # Group by BMI category and outcome
    result = []
    for name, group in df.groupby("BMICategory"):
        total = len(group)
        diabetic = group["Outcome"].sum()
        result.append({
            "bmi_category": name,
            "total_count": int(total),
            "diabetic_count": int(diabetic),
            "non_diabetic_count": int(total - diabetic),
            "diabetic_percentage": round(diabetic / total * 100, 2) if total > 0 else 0
        })
    
    return result

@app.get("/analysis/glucose-vs-outcome", tags=["Analysis"])
async def analyze_glucose_vs_outcome(bin_size: int = Query(10, description="Size of glucose level bins")):
    """
    Analyze the relationship between glucose levels and diabetes outcome.
    """
    # Create glucose bins
    df["GlucoseGroup"] = pd.cut(df["Glucose"], 
                                bins=range(0, df["Glucose"].max() + bin_size, bin_size),
                                right=False,
                                labels=[f"{i}-{i+bin_size-1}" for i in range(0, df["Glucose"].max(), bin_size)])
    
    # Group by glucose level and outcome
    result = []
    for name, group in df.groupby("GlucoseGroup"):
        total = len(group)
        diabetic = group["Outcome"].sum()
        result.append({
            "glucose_range": str(name),
            "total_count": int(total),
            "diabetic_count": int(diabetic),
            "non_diabetic_count": int(total - diabetic),
            "diabetic_percentage": round(diabetic / total * 100, 2) if total > 0 else 0
        })
    
    return result

@app.get("/search", tags=["Search"])
async def search(
    min_pregnancies: Optional[int] = Query(None, description="Minimum pregnancies"),
    max_pregnancies: Optional[int] = Query(None, description="Maximum pregnancies"),
    min_glucose: Optional[int] = Query(None, description="Minimum glucose"),
    max_glucose: Optional[int] = Query(None, description="Maximum glucose"),
    min_blood_pressure: Optional[int] = Query(None, description="Minimum blood pressure"),
    max_blood_pressure: Optional[int] = Query(None, description="Maximum blood pressure"),
    min_insulin: Optional[int] = Query(None, description="Minimum insulin"),
    max_insulin: Optional[int] = Query(None, description="Maximum insulin"),
    limit: int = Query(20, description="Limit the number of results")
):
    """
    Advanced search for diabetes records based on multiple criteria.
    """
    filtered_df = df.copy()
    
    # Apply all the filters that are provided
    if min_pregnancies is not None:
        filtered_df = filtered_df[filtered_df["Pregnancies"] >= min_pregnancies]
    if max_pregnancies is not None:
        filtered_df = filtered_df[filtered_df["Pregnancies"] <= max_pregnancies]
    if min_glucose is not None:
        filtered_df = filtered_df[filtered_df["Glucose"] >= min_glucose]
    if max_glucose is not None:
        filtered_df = filtered_df[filtered_df["Glucose"] <= max_glucose]
    if min_blood_pressure is not None:
        filtered_df = filtered_df[filtered_df["BloodPressure"] >= min_blood_pressure]
    if max_blood_pressure is not None:
        filtered_df = filtered_df[filtered_df["BloodPressure"] <= max_blood_pressure]
    if min_insulin is not None:
        filtered_df = filtered_df[filtered_df["Insulin"] >= min_insulin]
    if max_insulin is not None:
        filtered_df = filtered_df[filtered_df["Insulin"] <= max_insulin]
    
    # Apply limit
    filtered_df = filtered_df.head(limit)
    
    # Convert to list of dicts
    records = filtered_df.replace({float('nan'): None}).to_dict(orient="records")
    
    return records

@app.get("/analysis/correlation", tags=["Analysis"])
async def correlation_analysis():
    """
    Analyze correlations between different features and diabetes outcome.
    Returns the Pearson correlation coefficients.
    """
    # Calculate correlations
    correlation_matrix = df.corr()
    
    # Sort correlations with Outcome in descending order (strongest to weakest)
    outcome_correlations = correlation_matrix["Outcome"].drop("Outcome").sort_values(ascending=False)
    
    result = {
        "outcome_correlations": {
            feature: round(correlation, 4) 
            for feature, correlation in outcome_correlations.items()
        },
        "full_correlation_matrix": {
            feature1: {
                feature2: round(correlation, 4)
                for feature2, correlation in correlation_matrix[feature1].items()
            }
            for feature1 in correlation_matrix.columns
        }
    }
    
    return result

@app.get("/analysis/pregnancies-vs-outcome", tags=["Analysis"])
async def analyze_pregnancies_vs_outcome():
    """
    Analyze the relationship between number of pregnancies and diabetes outcome.
    """
    # Group by number of pregnancies
    grouped = df.groupby("Pregnancies").agg({
        "Outcome": ["count", "sum"]
    })
    
    grouped.columns = ["total_count", "diabetic_count"]
    grouped = grouped.reset_index()
    
    # Calculate percentages
    result = []
    for _, row in grouped.iterrows():
        result.append({
            "pregnancies": int(row["Pregnancies"]),
            "total_count": int(row["total_count"]),
            "diabetic_count": int(row["diabetic_count"]),
            "non_diabetic_count": int(row["total_count"] - row["diabetic_count"]),
            "diabetic_percentage": round(row["diabetic_count"] / row["total_count"] * 100, 2)
        })
    
    return result

@app.get("/analysis/insulin-vs-glucose", tags=["Analysis"])
async def analyze_insulin_vs_glucose(bin_size: int = Query(20, description="Size of glucose level bins")):
    """
    Analyze the relationship between insulin levels and glucose levels.
    """
    # Filter out records with zero insulin (might be missing data)
    filtered_df = df[df["Insulin"] > 0]
    
    # Create glucose bins
    filtered_df["GlucoseGroup"] = pd.cut(
        filtered_df["Glucose"], 
        bins=range(0, filtered_df["Glucose"].max() + bin_size, bin_size),
        right=False,
        labels=[f"{i}-{i+bin_size-1}" for i in range(0, filtered_df["Glucose"].max(), bin_size)]
    )
    
    # Group by glucose level and calculate insulin statistics
    result = []
    for name, group in filtered_df.groupby("GlucoseGroup"):
        result.append({
            "glucose_range": str(name),
            "count": len(group),
            "avg_insulin": round(group["Insulin"].mean(), 2),
            "median_insulin": round(group["Insulin"].median(), 2),
            "min_insulin": int(group["Insulin"].min()),
            "max_insulin": int(group["Insulin"].max()),
            "diabetic_percentage": round(group["Outcome"].mean() * 100, 2)
        })
    
    return result

@app.get("/analysis/risk-factors", tags=["Analysis"])
async def analyze_risk_factors():
    """
    Identify key risk factors for diabetes based on the dataset.
    Returns a simple risk score for each feature.
    """
    # Calculate risk score (simplified version - just based on correlation)
    correlation_with_outcome = df.corr()["Outcome"].drop("Outcome")
    
    # Convert to absolute values and sort
    risk_factors = correlation_with_outcome.abs().sort_values(ascending=False)
    
    result = {
        "risk_factors": {
            feature: {
                "correlation": round(correlation_with_outcome[feature], 4),
                "abs_correlation": round(risk_factors[feature], 4),
                "risk_score": round(risk_factors[feature] * 100, 2),  # Simple percentage-based score
                "direction": "positive" if correlation_with_outcome[feature] > 0 else "negative"
            }
            for feature in risk_factors.index
        }
    }
    
    return result

@app.get("/analysis/feature-distribution", tags=["Analysis"])
async def analyze_feature_distribution(
    feature: str = Query(..., description="Feature to analyze (column name)"),
    bins: int = Query(10, description="Number of bins for distribution")
):
    """
    Analyze the distribution of a specific feature, separated by diabetes outcome.
    """
    if feature not in df.columns:
        raise HTTPException(status_code=400, detail=f"Feature '{feature}' not found in dataset")
    
    # Calculate overall statistics
    overall_stats = {
        "mean": float(df[feature].mean()),
        "median": float(df[feature].median()),
        "std": float(df[feature].std()),
        "min": float(df[feature].min()),
        "max": float(df[feature].max())
    }
    
    # Calculate statistics by outcome
    stats_by_outcome = {}
    for outcome, group in df.groupby("Outcome"):
        stats_by_outcome[int(outcome)] = {
            "mean": float(group[feature].mean()),
            "median": float(group[feature].median()),
            "std": float(group[feature].std()),
            "min": float(group[feature].min()),
            "max": float(group[feature].max())
        }
    
    # Create distribution histogram
    hist_data = []
    
    # Get histogram data for all records
    all_hist, all_bins = np.histogram(df[feature].dropna(), bins=bins)
    
    for i in range(len(all_hist)):
        bin_start = float(all_bins[i])
        bin_end = float(all_bins[i+1])
        
        # Count records by outcome in this bin
        bin_df = df[(df[feature] >= bin_start) & (df[feature] < bin_end)]
        diabetic_count = int(bin_df["Outcome"].sum())
        total_count = len(bin_df)
        
        hist_data.append({
            "bin_range": f"{bin_start:.1f}-{bin_end:.1f}",
            "count": int(all_hist[i]),
            "diabetic_count": diabetic_count,
            "non_diabetic_count": total_count - diabetic_count,
            "diabetic_percentage": round(diabetic_count / total_count * 100, 2) if total_count > 0 else 0
        })
    
    return {
        "feature": feature,
        "overall_statistics": overall_stats,
        "statistics_by_outcome": stats_by_outcome,
        "distribution": hist_data
    }

@app.get("/analysis/age-bmi-matrix", tags=["Analysis"])
async def analyze_age_bmi_matrix(
    age_bins: int = Query(5, description="Number of age bins"),
    bmi_bins: int = Query(4, description="Number of BMI bins")
):
    """
    Create a matrix showing diabetes prevalence by age and BMI categories.
    """
    # Create age and BMI bins
    df["AgeGroup"] = pd.cut(
        df["Age"], 
        bins=age_bins,
        labels=[f"Age Group {i+1}" for i in range(age_bins)]
    )
    
    df["BMIGroup"] = pd.cut(
        df["BMI"], 
        bins=bmi_bins,
        labels=[f"BMI Group {i+1}" for i in range(bmi_bins)]
    )
    
    # Create a matrix of age vs BMI
    matrix = {}
    
    # Get the bin ranges
    age_min, age_max = df["Age"].min(), df["Age"].max()
    bmi_min, bmi_max = df["BMI"].min(), df["BMI"].max()
    
    age_ranges = [(age_min + i*(age_max-age_min)/age_bins, age_min + (i+1)*(age_max-age_min)/age_bins) 
                  for i in range(age_bins)]
    bmi_ranges = [(bmi_min + i*(bmi_max-bmi_min)/bmi_bins, bmi_min + (i+1)*(bmi_max-bmi_min)/bmi_bins) 
                  for i in range(bmi_bins)]
    
    # Add readable ranges to the output
    age_labels = [f"Age {a[0]:.1f}-{a[1]:.1f}" for a in age_ranges]
    bmi_labels = [f"BMI {b[0]:.1f}-{b[1]:.1f}" for b in bmi_ranges]
    
    for age_group in df["AgeGroup"].dropna().unique():
        matrix[str(age_group)] = {}
        for bmi_group in df["BMIGroup"].dropna().unique():
            group_df = df[(df["AgeGroup"] == age_group) & (df["BMIGroup"] == bmi_group)]
            total = len(group_df)
            diabetic = group_df["Outcome"].sum()
            
            matrix[str(age_group)][str(bmi_group)] = {
                "total_count": int(total),
                "diabetic_count": int(diabetic),
                "diabetic_percentage": round(diabetic / total * 100, 2) if total > 0 else 0
            }
    
    return {
        "age_ranges": {f"Age Group {i+1}": age_labels[i] for i in range(len(age_labels))},
        "bmi_ranges": {f"BMI Group {i+1}": bmi_labels[i] for i in range(len(bmi_labels))},
        "matrix": matrix
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
