# Diabetes Dataset API

A FastAPI-based REST API that provides access and analysis tools for the Pima Indians Diabetes Dataset. This API allows users to query, filter, and analyze diabetes data to discover patterns and relationships between various factors and diabetes outcomes.

## Table of Contents

- [Features](#features)
- [Setup](#setup)
- [API Endpoints](#api-endpoints)
- [API Examples](#api-examples)
- [Data Format](#data-format)
- [Contributing](#contributing)

## Features

- Browse and filter diabetes records with pagination
- Get detailed statistics on the dataset
- Analyze relationships between various factors and diabetes outcomes
- View distributions of features separated by diabetes outcome
- Explore correlations between different metrics
- Identify key risk factors for diabetes
- Visualize relationships using binned data and matrices
- RESTful API with JSON responses
- Interactive API documentation with Swagger UI

## Setup

### Prerequisites

- Python 3.7+
- pandas
- FastAPI
- uvicorn
- numpy

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/diabetes-data.git
   cd diabetes-data
   ```
2. Install dependencies:

   ```bash
   pip install fastapi uvicorn pandas numpy
   ```
3. Ensure the data file is available:

   - Place the diabetes dataset (`diabetes.csv`) in the `data/` directory
   - The CSV should have the expected columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
4. Run the API:

   ```bash
   cd api
   python main.py
   ```
5. Access the API documentation at http://localhost:8000/docs

## API Endpoints


| Endpoint                           | Method | Description                                      |
| ---------------------------------- | ------ | ------------------------------------------------ |
| `/`                                | GET    | Welcome message and dataset overview             |
| `/records`                         | GET    | List diabetes records with optional filtering    |
| `/records/{record_id}`             | GET    | Get details for a specific record                |
| `/statistics`                      | GET    | Get statistical summary of the dataset           |
| `/analysis/age-groups`             | GET    | Analyze diabetes outcomes by age groups          |
| `/analysis/bmi-groups`             | GET    | Analyze diabetes outcomes by BMI categories      |
| `/analysis/glucose-vs-outcome`     | GET    | Analyze glucose levels vs diabetes outcome       |
| `/analysis/correlation`            | GET    | Analyze correlations between features            |
| `/analysis/pregnancies-vs-outcome` | GET    | Analyze pregnancies vs diabetes outcome          |
| `/analysis/insulin-vs-glucose`     | GET    | Analyze relationship between insulin and glucose |
| `/analysis/risk-factors`           | GET    | Identify key risk factors for diabetes           |
| `/analysis/feature-distribution`   | GET    | Analyze distribution of any feature              |
| `/analysis/age-bmi-matrix`         | GET    | Create age-BMI matrix of diabetes prevalence     |
| `/search`                          | GET    | Advanced search with multiple criteria           |

## API Examples

### Basic Usage

#### 1. Get dataset overview

```bash
curl http://localhost:8000/
```

Response:

```json
{
  "message": "Welcome to the Diabetes Dataset API",
  "dataset_info": {
    "rows": 768,
    "columns": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome", "id"],
    "positive_cases": 268,
    "negative_cases": 500
  }
}
```

#### 2. List diabetes records (with pagination)

```bash
curl http://localhost:8000/records?limit=5
```

Response:

```json
[
  {
    "id": 1,
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50,
    "Outcome": 1
  },
  ...
]
```

#### 3. Get a specific record

```bash
curl http://localhost:8000/records/1
```

### Filtering Records

#### Filter by age range

```bash
curl "http://localhost:8000/records?min_age=50&max_age=60"
```

#### Filter by BMI and outcome

```bash
curl "http://localhost:8000/records?min_bmi=30&outcome=1"
```

### Statistical Analysis

#### Get overall statistics

```bash
curl http://localhost:8000/statistics
```

Response:

```json
{
  "count": 768,
  "mean": {
    "Pregnancies": 3.8,
    "Glucose": 120.9,
    ...
  },
  "median": {
    "Pregnancies": 3.0,
    "Glucose": 117.0,
    ...
  },
  ...
}
```

### Feature Analysis

#### Analyze age groups

```bash
curl http://localhost:8000/analysis/age-groups
```

Response:

```json
[
  {
    "age_group": "21-30",
    "total_count": 392,
    "diabetic_count": 115,
    "non_diabetic_count": 277,
    "diabetic_percentage": 29.34
  },
  ...
]
```

#### Analyze glucose levels vs diabetes

```bash
curl "http://localhost:8000/analysis/glucose-vs-outcome?bin_size=20"
```

#### Get correlations between features

```bash
curl http://localhost:8000/analysis/correlation
```

Response:

```json
{
  "outcome_correlations": {
    "Glucose": 0.4666,
    "BMI": 0.2927,
    ...
  },
  "full_correlation_matrix": {
    ...
  }
}
```

#### Identify risk factors

```bash
curl http://localhost:8000/analysis/risk-factors
```

#### Analyze distribution of a feature

```bash
curl "http://localhost:8000/analysis/feature-distribution?feature=Glucose&bins=10"
```

#### Create age-BMI matrix

```bash
curl "http://localhost:8000/analysis/age-bmi-matrix?age_bins=5&bmi_bins=4"
```

### Advanced Search

```bash
curl "http://localhost:8000/search?min_glucose=140&min_age=30&max_age=50"
```

### Python Requests Example

```python
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Base URL
base_url = "http://localhost:8000"

# Get statistics
stats_response = requests.get(f"{base_url}/statistics")
stats = stats_response.json()
print(f"Dataset count: {stats['count']} records")
print(f"Average glucose level: {stats['mean']['Glucose']}")
print(f"Average BMI: {stats['mean']['BMI']}")

# Get glucose vs outcome analysis
glucose_analysis = requests.get(f"{base_url}/analysis/glucose-vs-outcome", params={"bin_size": 20})
glucose_data = glucose_analysis.json()

# Convert to DataFrame for easier plotting
df = pd.DataFrame(glucose_data)
df = df.sort_values('glucose_range')

# Plot diabetic percentage vs glucose
plt.figure(figsize=(10, 6))
plt.bar(df['glucose_range'], df['diabetic_percentage'])
plt.xlabel('Glucose Level Range')
plt.ylabel('Diabetic Percentage (%)')
plt.title('Diabetes Percentage by Glucose Level')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('glucose_vs_diabetes.png')
print("Saved glucose analysis chart to glucose_vs_diabetes.png")

# Get risk factors
risk_factors = requests.get(f"{base_url}/analysis/risk-factors")
risks = risk_factors.json()['risk_factors']
print("\nTop 3 risk factors for diabetes:")
for i, (feature, data) in enumerate(sorted(risks.items(), key=lambda x: x[1]['risk_score'], reverse=True)[:3]):
    print(f"{i+1}. {feature}: {data['risk_score']} risk score (correlation: {data['correlation']})")
```

## Data Format

Each record in the Pima Indians Diabetes Dataset has the following properties:


| Feature                  | Type  | Description                                                         |
| ------------------------ | ----- | ------------------------------------------------------------------- |
| Pregnancies              | int   | Number of times pregnant                                            |
| Glucose                  | int   | Plasma glucose concentration (2 hours after glucose tolerance test) |
| BloodPressure            | int   | Diastolic blood pressure (mm Hg)                                    |
| SkinThickness            | int   | Triceps skinfold thickness (mm)                                     |
| Insulin                  | int   | 2-Hour serum insulin (mu U/ml)                                      |
| BMI                      | float | Body mass index (weight in kg/(height in m)²)                      |
| DiabetesPedigreeFunction | float | Diabetes pedigree function (genetic influence)                      |
| Age                      | int   | Age in years                                                        |
| Outcome                  | int   | Class variable (0: no diabetes, 1: diabetes)                        |

Note: In this dataset, a 1 in the "Outcome" column indicates the presence of diabetes, while a 0 indicates its absence.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Pima Indians Diabetes Dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases
- The dataset is widely used for machine learning and data analysis education
