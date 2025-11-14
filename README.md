# AI-based-Trajactory-Analysis-System


## Overview

This Python tool processes trajectory data from the GeoLife dataset. It consists of a series of steps that include:

1. **Data Processing**: The `geolife_pipeline.py` script processes raw trajectory data and generates a JSON file that contains the users' trajectory points, stays, trips, and various spatial-temporal metrics.

2. **Reverse Geocoding**: The `geocoding.py` script takes the generated JSON file and performs reverse geocoding to convert coordinates into human-readable place names (using the Amap API).

3. **Word Cloud Generation**: The `wordcloud_English.py` script analyzes the place names and generates a word cloud representing the most frequent locations. It also provides a top-8 list of place types.

4. **Visualization**: The `folium_map_dynamic.py` script generates an interactive web map using `folium`, displaying users' trajectories, stays, and visualizations like KDE hotspots and SDE (Standard Deviational Ellipses).

## Functionality

### 1. `geolife_pipeline.py`
- **Purpose**: Preprocess the GeoLife dataset to extract spatial and temporal metrics, producing a JSON file suitable for further analysis.
- **Input**: GeoLife raw trajectory files (PLT format).
- **Output**: JSON file containing:
  - User's spatial and temporal metrics
  - Stay points and trip segments
  - Standard Deviational Ellipse (SDE) and radius of gyration
  - Time entropy, movement patterns, and more

### 2. `geocoding.py`
- **Purpose**: Converts raw GPS coordinates into human-readable place names using the Amap Reverse Geocoding API.
- **Input**: JSON file from `geolife_pipeline.py` (generated output).
- **Output**: Updated JSON file with place names and additional geolocation information (e.g., neighborhood, POIs).

### 3. `wordcloud_English.py`
- **Purpose**: Generates a word cloud based on the names of places where users stayed.
- **Input**: Geocoded JSON file (from `geocoding.py`).
- **Output**: 
  - A word cloud image showing the most frequently visited places.
  - A JSON file containing the frequency of place names and types.
  - A summary of the top 8 place types.

### 4. `folium_map_dynamic.py`
- **Purpose**: Visualizes the user trajectory and stay points on an interactive map.
- **Input**: JSON file with geocoded location data.
- **Output**: HTML file containing an interactive map with:
  - Trajectories and stay points
  - Convex hull and Standard Deviational Ellipses (SDE)
  - KDE hotspots and heatmaps
  - Raw trajectory visualization (if applicable)

## Input Parameters

### 1. `geolife_pipeline.py`
- **Input**:
  - Raw trajectory files (`*.plt` format) from the GeoLife dataset.
- **Parameters**:
  - `root_dir`: Path to the directory containing the GeoLife dataset (`Data/000/Trajectory/`).

### 2. `geocoding.py`
- **Input**:
  - JSON file from `geolife_pipeline.py` (generated output).
  - API Key for Amap Reverse Geocoding API.
- **Parameters**:
  - `INPUT_FILE`: Path to the input JSON file.
  - `OUTPUT_FILE`: Path to the output JSON file.
  - `AMAP_KEY`: Your Amap API key.

### 3. `wordcloud_English.py`
- **Input**:
  - Geocoded JSON file.
- **Parameters**:
  - `FILE_PATH`: Path to the geocoded input JSON file.
  - `OUTPUT_WORDCLOUD_JSON`: Path to save the JSON output.
  - `OUTPUT_WORDCLOUD_NAME`: Path to save the word cloud image.
  - `OUTPUT_TYPE_JSON`: Path to save the place type statistics.
  - `FONT_PATH`: Path to the font file for rendering Chinese characters (if needed).
  - `ENABLE_TRANSLATION`: Set to `True` to enable translation of place names from Chinese to English.

### 4. `folium_map_dynamic.py`
- **Input**:
  - JSON file with geocoded location data (from `geocoding.py`).
- **Parameters**:
  - `input_path`: Path to the input JSON file.
  - `html_out`: Path to save the generated HTML map.
  - `kde_bandwidth_m`: Bandwidth parameter for Kernel Density Estimation (default: `1500` meters).
  - `plt_folder`: Optional folder containing raw PLT files to visualize raw trajectories.

## Outputs

### 1. `geolife_pipeline.py`
- **Output**: 
  - JSON file containing preprocessed spatial-temporal metrics, stay points, trips, and more.

### 2. `geocoding.py`
- **Output**:
  - JSON file with reverse geocoding results (place names, POIs, address components, etc.).

### 3. `wordcloud_English.py`
- **Output**:
  - Word cloud image showing the most frequently visited places.
  - JSON file with the frequency of place names and types.
  - JSON file with a summary of the top-8 place types.

### 4. `folium_map_dynamic.py`
- **Output**:
  - HTML file with an interactive map visualizing trajectories, stay points, and other metrics.

## Installation

To run this tool, ensure you have the following dependencies installed:

```bash
pip install pandas numpy shapely pyproj scikit-learn folium deep-translator matplotlib jieba
```

## Usage

1. **Run `geolife_pipeline.py`**:
   ```bash
   python geolife_pipeline.py --root /path/to/GeoLife/Data/000
   ```

2. **Run `geocoding.py`**:
   ```bash
   python geocoding.py --input /path/to/geolife_output.json --output /path/to/geocoded_output.json --api_key YOUR_AMAP_API_KEY
   ```

3. **Run `wordcloud_English.py`**:
   ```bash
   python wordcloud_English.py --input /path/to/geocoded_output.json --output_wordcloud_name /path/to/wordcloud.png --output_wordcloud_json /path/to/wordcloud_data.json --output_type_json /path/to/type_data.json --font_path /path/to/font.ttf
   ```

4. **Run `folium_map_dynamic.py`**:
   ```bash
   python folium_map_dynamic.py --input /path/to/geocoded_output.json --html_out /path/to/interactive_map.html
   ```


