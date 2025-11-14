#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高德API反地理编码程序
处理GeoLife数据集中的所有位置信息
"""

import json
import requests
import time
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class LocationInfo:
    """位置信息数据类"""
    lat: float
    lon: float
    source_type: str  # 'top5_stays', 'stays', 'trips_start', 'trips_end'
    index: int
    additional_info: Dict = None


class AmapReverseGeocoder:
    """高德反地理编码器"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://restapi.amap.com/v3/geocode/regeo"
        self.session = requests.Session()
        self.request_count = 0

    def reverse_geocode_single(self, lat: float, lon: float,
                               extensions: str = "all") -> Optional[Dict]:
        """
        单个位置反地理编码

        Args:
            lat: 纬度
            lon: 经度
            extensions: 返回信息详细程度 ('base' 或 'all')

        Returns:
            反地理编码结果字典，失败返回None
        """
        try:
            params = {
                'key': self.api_key,
                'location': f"{lon},{lat}",  # 高德API要求经度在前，纬度在后
                'extensions': extensions,
                'roadlevel': 1,
                'output': 'json'
            }

            response = self.session.get(self.base_url, params=params, timeout=10)
            self.request_count += 1

            if response.status_code == 200:
                result = response.json()
                if result.get('status') == '1':
                    return result
                else:
                    logger.error(f"API返回错误: {result.get('info', '未知错误')}")
                    return None
            else:
                logger.error(f"HTTP请求失败: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"反地理编码请求异常: {e}")
            return None

    def reverse_geocode_batch(self, locations: List[LocationInfo],
                              max_workers: int = 5,
                              delay: float = 0.1) -> List[Dict]:
        """
        批量反地理编码

        Args:
            locations: 位置信息列表
            max_workers: 并发工作线程数
            delay: 请求间隔（秒）

        Returns:
            结果列表
        """
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_location = {}
            for i, location in enumerate(locations):
                time.sleep(delay)  # 控制请求频率
                future = executor.submit(
                    self.reverse_geocode_single,
                    location.lat,
                    location.lon
                )
                future_to_location[future] = (i, location)

            # 收集结果
            for future in as_completed(future_to_location):
                index, location = future_to_location[future]
                try:
                    api_result = future.result()

                    result = {
                        'index': index,
                        'source_type': location.source_type,
                        'original_location': {
                            'lat': location.lat,
                            'lon': location.lon
                        },
                        'geocoding_result': api_result,
                        'additional_info': location.additional_info
                    }

                    if api_result and 'regeocode' in api_result:
                        regeocode = api_result['regeocode']
                        result['formatted_address'] = regeocode.get('formatted_address', '')

                        # 提取结构化地址信息
                        if 'addressComponent' in regeocode:
                            addr_comp = regeocode['addressComponent']
                            result['address_component'] = {
                                'country': addr_comp.get('country', ''),
                                'province': addr_comp.get('province', ''),
                                'city': addr_comp.get('city', ''),
                                'district': addr_comp.get('district', ''),
                                'township': addr_comp.get('township', ''),
                                'neighborhood': addr_comp.get('neighborhood', {}).get('name', ''),
                                'building': addr_comp.get('building', {}).get('name', ''),
                                'adcode': addr_comp.get('adcode', ''),
                                'citycode': addr_comp.get('citycode', '')
                            }

                        # 提取POI信息
                        if 'pois' in regeocode:
                            result['nearby_pois'] = [
                                {
                                    'name': poi.get('name', ''),
                                    'type': poi.get('type', ''),
                                    'distance': poi.get('distance', ''),
                                    'direction': poi.get('direction', '')
                                }
                                for poi in regeocode['pois'][:5]  # 只保留前5个POI
                            ]

                    results.append(result)
                    logger.info(f"完成第 {index + 1}/{len(locations)} 个位置的反地理编码")

                except Exception as e:
                    logger.error(f"处理位置 {index} 时出错: {e}")
                    results.append({
                        'index': index,
                        'source_type': location.source_type,
                        'original_location': {
                            'lat': location.lat,
                            'lon': location.lon
                        },
                        'error': str(e),
                        'additional_info': location.additional_info
                    })

        # 按index排序
        results.sort(key=lambda x: x['index'])
        return results


def extract_locations_from_geolife(data: Dict) -> List[LocationInfo]:
    """
    从GeoLife数据中提取所有位置信息

    Args:
        data: GeoLife JSON数据

    Returns:
        位置信息列表
    """
    locations = []

    for user_data in data.get('users', []):
        user_id = user_data.get('user_id', 'unknown')

        # 1. 提取top5_stays位置
        if 'spatiotemporal_metrics' in user_data and 'top5_stays' in user_data['spatiotemporal_metrics']:
            for i, stay in enumerate(user_data['spatiotemporal_metrics']['top5_stays']):
                locations.append(LocationInfo(
                    lat=stay['lat'],
                    lon=stay['lon'],
                    source_type='top5_stays',
                    index=len(locations),
                    additional_info={
                        'user_id': user_id,
                        'stay_index': i,
                        'duration_s': stay.get('duration_s'),
                        't_start': stay.get('t_start'),
                        't_end': stay.get('t_end')
                    }
                ))

        # 2. 提取stays位置
        if 'stays' in user_data:
            for i, stay in enumerate(user_data['stays']):
                locations.append(LocationInfo(
                    lat=stay['lat'],
                    lon=stay['lon'],
                    source_type='stays',
                    index=len(locations),
                    additional_info={
                        'user_id': user_id,
                        'stay_index': i,
                        'duration_s': stay.get('duration_s'),
                        't_start': stay.get('t_start'),
                        't_end': stay.get('t_end')
                    }
                ))

        # 3. 提取trips的起始点和终点
        if 'trips' in user_data:
            for i, trip in enumerate(user_data['trips']):
                # 起点
                if 'start_lat' in trip and 'start_lon' in trip:
                    locations.append(LocationInfo(
                        lat=trip['start_lat'],
                        lon=trip['start_lon'],
                        source_type='trips_start',
                        index=len(locations),
                        additional_info={
                            'user_id': user_id,
                            'trip_index': i,
                            'duration_s': trip.get('duration_s'),
                            'distance_m': trip.get('distance_m'),
                            't_start': trip.get('t_start'),
                            't_end': trip.get('t_end')
                        }
                    ))

                # 终点
                if 'end_lat' in trip and 'end_lon' in trip:
                    locations.append(LocationInfo(
                        lat=trip['end_lat'],
                        lon=trip['end_lon'],
                        source_type='trips_end',
                        index=len(locations),
                        additional_info={
                            'user_id': user_id,
                            'trip_index': i,
                            'duration_s': trip.get('duration_s'),
                            'distance_m': trip.get('distance_m'),
                            't_start': trip.get('t_start'),
                            't_end': trip.get('t_end')
                        }
                    ))

    return locations


def main():
    """主函数"""

    # 配置
    AMAP_KEY = "d57cca0891b730d4764252723531ce23"  # 你的API Key
    INPUT_FILE = "output.json"
    OUTPUT_FILE = "geocoded_output.json"

    logger.info("开始处理GeoLife数据反地理编码任务...")

    try:
        # 1. 读取JSON数据
        logger.info(f"读取数据文件: {INPUT_FILE}")
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 2. 提取位置信息
        logger.info("提取位置信息...")
        locations = extract_locations_from_geolife(data)
        logger.info(f"共提取到 {len(locations)} 个位置点")

        # 统计各类型位置数量
        type_counts = {}
        for loc in locations:
            type_counts[loc.source_type] = type_counts.get(loc.source_type, 0) + 1

        logger.info("位置类型统计:")
        for loc_type, count in type_counts.items():
            logger.info(f"  {loc_type}: {count} 个")

        if not locations:
            logger.warning("未找到任何位置数据！")
            return

        # 3. 反地理编码
        logger.info("开始反地理编码...")
        geocoder = AmapReverseGeocoder(AMAP_KEY)

        # 可以选择处理全部位置或仅处理前N个（用于测试）
        # locations = locations[:10]  # 仅处理前10个位置用于测试

        results = geocoder.reverse_geocode_batch(
            locations,
            max_workers=3,  # 控制并发数，避免超过API限制
            delay=0.5 # 请求间隔200ms
        )

        logger.info(f"反地理编码完成，共处理 {len(results)} 个位置")
        logger.info(f"总共发送了 {geocoder.request_count} 次API请求")

        # 4. 保存结果
        output_data = {
            'metadata': {
                'original_file': INPUT_FILE,
                'total_locations': len(locations),
                'processed_locations': len(results),
                'api_requests_sent': geocoder.request_count,
                'location_type_counts': type_counts,
                'processing_time': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'results': results
        }

        logger.info(f"保存结果到: {OUTPUT_FILE}")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        # 5. 生成摘要报告
        successful_results = [r for r in results if 'formatted_address' in r]
        failed_results = [r for r in results if 'error' in r]

        logger.info("=" * 50)
        logger.info("处理完成！摘要报告:")
        logger.info(f"总位置数: {len(locations)}")
        logger.info(f"成功处理: {len(successful_results)}")
        logger.info(f"处理失败: {len(failed_results)}")
        logger.info(f"成功率: {len(successful_results) / len(locations) * 100:.1f}%")
        logger.info(f"API调用次数: {geocoder.request_count}")
        logger.info(f"结果已保存到: {OUTPUT_FILE}")

        # 显示一些示例结果
        if successful_results:
            logger.info("\n示例结果:")
            for i, result in enumerate(successful_results[:3]):
                logger.info(f"  {i + 1}. {result['source_type']}: {result.get('formatted_address', '未知地址')}")

    except FileNotFoundError:
        logger.error(f"文件未找到: {INPUT_FILE}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析错误: {e}")
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        raise


if __name__ == "__main__":
    main()