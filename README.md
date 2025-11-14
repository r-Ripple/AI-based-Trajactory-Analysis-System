# AI-based-Trajactory-Analysis-System


This repository provides a pipeline for analyzing trajectory data from the GeoLife dataset. The analysis flow processes the data in sequential steps, with each component performing a specific task, from data preprocessing to visualization.

## Overview of the Pipeline

### 1. **Pipeline (`geolife_pipeline.py`)**
   - **Input**: Takes GeoLife trajectory data in JSON format.
   - **Process**: Processes the raw data into structured metrics related to time, space, and spatio-temporal behavior patterns.
   - **Output**: Generates a JSON containing the processed results, ready for further analysis.

#### Key Steps:
   - **Time and Space Metrics**: Computes time-based statistics such as the number of trips, total travel time, and spatial metrics like radius of gyration, convex hull area, and standard deviational ellipses.
   - **Spatio-temporal Metrics**: Includes analysis of stay points, trip segmentation, and the study of transitions between activities (e.g., from one POI to another).
   - **User Profiling**: Generates summaries of user behavior based on spatio-temporal data, including insights about the user’s mobility patterns.

#### Example Usage:
   ```bash
   python geolife_pipeline.py --input geolife_data.json --output processed_data.json
   ```

### 2. **Reverse Geocoding (`geocoding.py`)**
   - **Input**: Processes the trajectory data and outputs geocoded location information.
   - **Process**: Uses the Amap API to convert latitude and longitude coordinates to human-readable addresses.
   - **Output**: A JSON file containing geocoded data, including formatted addresses, place names, and additional POI information.

#### Example Usage:
   ```bash
   python geocoding.py --input processed_data.json --output geocoded_data.json
   ```

### 3. **WordCloud Generation (`wordcloud_English.py`)**
   - **Input**: The geocoded data, specifically the names of places (e.g., POIs).
   - **Process**: Generates a word cloud visualization from the extracted place names, optionally translating non-English names.
   - **Output**: Saves a word cloud image and JSON summary containing place frequency data and the translation mapping.

#### Example Usage:
   ```bash
   python wordcloud_English.py --input geocoded_data.json --output wordcloud.png --output_json wordcloud_data.json
   ```

### 4. **Visualization (`folium_map_dynamic.py`)**
   - **Input**: Processes the location data and trajectory points.
   - **Process**: Visualizes the trajectories, stay points, and statistical patterns (e.g., heatmaps, convex hulls) on an interactive map using Folium.
   - **Output**: Generates an interactive HTML map visualizing the geographic analysis.

#### Example Usage:
   ```bash
   python folium_map_dynamic.py --input geocoded_data.json --output map.html
   ```

### 5. **Batch Analyzer (`batch_analyzer.py`)**
   - **Input**: The batch analyzer allows for the processing of multiple user trajectory files in parallel.
   - **Process**: Handles the batch processing of GeoLife data, including data segmentation, trajectory analysis, and feature extraction across many users.
   - **Output**: Generates detailed analysis reports for multiple users, including summaries and deeper insights based on user movement patterns.

#### Example Usage:
   ```bash
   python batch_analyzer.py --input trajectory_data_folder --output analysis_results.json
   ```

### 6. **Trajectory AI Analyzer (`trajectory_ai_analyzer.py`)**
   - **Input**: This script performs advanced analysis using AI models for trajectory data, analyzing user behavior, spatial patterns, and temporal transitions.
   - **Process**: Uses AI (e.g., OpenAI models) to analyze behavior patterns, time-series transitions, and spatial dynamics from trajectory data.
   - **Output**: Generates a comprehensive analysis report, including key insights and actionable findings based on the user’s movement patterns.

#### Example Usage:
   ```bash
   python trajectory_ai_analyzer.py --input trajectory_data.json --output ai_analysis_results.json
   ```

## How to Run the Pipeline

### 1. **Run the Pipeline (`geolife_pipeline.py`)**
   This script will process the raw GeoLife data and produce a JSON output with various metrics.

   ```bash
   python geolife_pipeline.py --input geolife_data.json --output processed_data.json
   ```

### 2. **Reverse Geocoding**
   After generating the processed data, you can run reverse geocoding to convert raw coordinates into human-readable locations.

   ```bash
   python geocoding.py --input processed_data.json --output geocoded_data.json
   ```

### 3. **Generate Word Cloud**
   To create a word cloud from the geocoded places, run the following:

   ```bash
   python wordcloud_English.py --input geocoded_data.json --output wordcloud.png --output_json wordcloud_data.json
   ```

### 4. **Generate Map Visualization**
   Finally, visualize the trajectories and locations on a map:

   ```bash
   python folium_map_dynamic.py --input geocoded_data.json --output map.html
   ```

### 5. **Run Batch Analysis**
   If you have multiple trajectory files to process, use the batch analyzer to process and generate reports.

   ```bash
   python batch_analyzer.py --input trajectory_data_folder --output batch_analysis_results.json
   ```

### 6. **Run AI-Based Analysis**
   For AI-based trajectory analysis, use the `trajectory_ai_analyzer.py` to generate deep insights.

   ```bash
   python trajectory_ai_analyzer.py --input trajectory_data.json --output ai_analysis_results.json
   ```

## Requirements
- Python 3.6+
- Required libraries (can be installed using `pip`):
  ```bash
  pip install pandas numpy geopandas shapely pyproj scikit-learn folium wordcloud deep-translator matplotlib openai
  ```

## Additional Notes
- The **reverse geocoding** step uses the Amap API. You need to get an API key from Amap and set it in the script.
- The **WordCloud Generation** can optionally translate place names using the Google Translator API (via `deep-translator`).
- **Visualization** uses **Folium** to create interactive maps. The generated map is saved as an HTML file.

## Output
1. **JSON Files**:
   - `processed_data.json`: The processed GeoLife data with various metrics.
   - `geocoded_data.json`: The data with geographical information.
   - `wordcloud_data.json`: Data used to generate the word cloud.

2. **Images**:
   - `wordcloud.png`: Word cloud image showing the frequency of place names.

3. **HTML Files**:
   - `map.html`: Interactive map visualizing the geographical analysis.
