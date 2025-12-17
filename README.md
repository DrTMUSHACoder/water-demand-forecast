# Water Demand Forecasting (SIH 2024)

This project forecasts water demand and assesses reservoir storage capacities using AI/ML.

## Features
- **District Data**: View water audit data for various districts.
- **Reservoir Data**: Monitor reservoir levels and storage.
- **Forecasting**: Predict future water demand using an interactive graph.
- **Water Theme UI**: A modern, glassmorphism-inspired user interface.

## Setup Instructions

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Initialize Database**:
    This uses a local SQLite database. Run the setup script to populate it with dummy data:
    ```bash
    python setup_sqlite.py
    ```

3.  **Run the Application**:
    ```bash
    python app1.py
    ```
    Access the app at `http://127.0.0.1:5000/Home`.

## Technologies
- **Backend**: Flask (Python)
- **Database**: SQLite
- **Frontend**: HTML5, CSS3, Chart.js, Bootstrap 5
