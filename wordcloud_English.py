import json
from collections import Counter
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import jieba
import os
from deep_translator import GoogleTranslator
import time

# --- 1. parameters ---
FILE_PATH = r"E:\NUS_Applied_GIS\3.Spatial_Programming\Final project\geocoded_output.json"
OUTPUT_WORDCLOUD_JSON = r"C:\Users\Yvanp\Desktop\stay_points_name_wordcloud2.json"
OUTPUT_WORDCLOUD_NAME = r"C:\Users\Yvanp\Desktop\stay_points_name_wordcloud2.png"
OUTPUT_TYPE_JSON = r"C:\Users\Yvanp\Desktop\stay_points_type_top8.json"
FONT_PATH = r"C:/Windows/Fonts/simhei.ttf"
ENABLE_TRANSLATION = True  

# --- 2. trans func ---
def translate_text(text, max_retries=3):
    
    if not text:
        return text
    
    for attempt in range(max_retries):
        try:
            translator = GoogleTranslator(source='zh-CN', target='en')
            result = translator.translate(text)
            time.sleep(0.2)  # sleep here
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"翻译失败 '{text}'，重试中... ({attempt + 1}/{max_retries})")
                time.sleep(2)
            else:
                print(f"翻译失败 '{text}'，使用原文: {e}")
                return text
    return text

def batch_translate(texts):
   
    translated = {}
    unique_texts = list(set(texts))
    
    print(f"正在翻译 {len(unique_texts)} 个唯一文本...")
    for i, text in enumerate(unique_texts):
        if i % 10 == 0:
            print(f"翻译进度: {i}/{len(unique_texts)}")
        translated[text] = translate_text(text)
        time.sleep(0.3)  # sleep
    
    print(f"翻译完成！")
    return translated

# --- 3. extract place ---
def extract_place_info(location):
    name = None
    place_type = None
    
    try:
        pois = location['geocoding_result']['regeocode'].get('pois')
        
        if pois and len(pois) > 0:
            first_poi = pois[0]
            
            if 'name' in first_poi and first_poi['name'].strip():
                name = first_poi['name'].strip()
                
            if 'type' in first_poi and first_poi['type'].strip():
                place_type = first_poi['type'].split(';')[0].strip()

        if not name:
            neighborhood_name = location['geocoding_result']['regeocode']['addressComponent']['neighborhood'].get('name')
            if neighborhood_name:
                name = neighborhood_name.strip()
                
    except (KeyError, TypeError):
        pass
    
    return name, place_type

# --- 4. main ---
def analyze_and_generate_plots(file_path, output_wc_name, output_wc_json, output_type_json, font_path, enable_translation=True):
    
    if not os.path.exists(font_path):
        print(f"no font file '{font_path}'")
        return

    print(f"--- 1. reading: {file_path} ---")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"error when reading: {e}")
        return

    results = data.get('results', [])
    
    place_names = []
    place_types = []
    
    print("--- 2. filtering stay points ---")
    
    stay_locations = [
        loc for loc in results
        if loc.get('source_type') in ['stays', 'top5_stays']
    ]

    for loc in stay_locations:
        name, p_type = extract_place_info(loc)
        if name:
            place_names.append(name)
        if p_type:
            place_types.append(p_type)

    if not place_names:
        print("warning: no stay points detected")
        return

    # --- translation ---
    name_translation = {}
    type_translation = {}
    
    if enable_translation:
        print("\n--- start to trans ---")
        
        print("\n--- trans type ---")
        type_translation = batch_translate(place_types)
        
        print("\n--- trans name ---")
        name_translation = batch_translate(place_names)

    # --- 3. wordcloud ---
    print(f"\n--- 3. ( {len(place_names)} valid name) ---")
    
    if enable_translation:
        translated_names = [name_translation.get(name, name) for name in place_names]
        name_counts = Counter(translated_names)
    else:
        name_counts = Counter(place_names)
    
    word_freq = dict(name_counts)
    

    wordcloud_font = None if enable_translation else font_path
    
    wordcloud = WordCloud(
        font_path=wordcloud_font,
        background_color="white",
        max_words=200,
        width=1200,
        height=700,
        margin=2,
        prefer_horizontal=0.9
    )

    wordcloud.generate_from_frequencies(word_freq)
    wordcloud.to_file(output_wc_name)
    print(f"✅ wordcloud save to: {output_wc_name}")

    # save as json
    wordcloud_data = {
        "total_places": len(place_names),
        "unique_places": len(name_counts),
        "translation_enabled": enable_translation,
        "word_frequencies": [
            {
                "rank": idx + 1,
                "name": name,
                "count": count
            }
            for idx, (name, count) in enumerate(name_counts.most_common())
        ]
    }
    
    with open(output_wc_json, 'w', encoding='utf-8') as f:
        json.dump(wordcloud_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ save json to: {output_wc_json}")

    # --- 4. POI type JSON ---
    print(f"\n--- 4. 正在统计类型并生成JSON (共 {len(place_types)} 个有效类型) ---")
    
    if enable_translation:
        translated_types = [type_translation.get(t, t) for t in place_types]
        type_counts = Counter(translated_types).most_common(8)
    else:
        type_counts = Counter(place_types).most_common(8)
    
    output_data = {
        "total_stay_points": len(place_types),
        "translation_enabled": enable_translation,
        "top8_types": [
            {
                "rank": idx + 1,
                "type": type_name,
                "count": count,
                "percentage": round((count / len(place_types)) * 100, 2)
            }
            for idx, (type_name, count) in enumerate(type_counts)
        ]
    }
    

    if enable_translation:
        output_data["translation_mapping"] = {
            "types": {original: type_translation.get(original, original) 
                     for original in set(place_types)}
        }
    
    with open(output_type_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 场所类型统计JSON已保存为: {output_type_json}")


# main
analyze_and_generate_plots(FILE_PATH, OUTPUT_WORDCLOUD_NAME, OUTPUT_WORDCLOUD_JSON, OUTPUT_TYPE_JSON, FONT_PATH, ENABLE_TRANSLATION)