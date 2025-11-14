#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量轨迹数据分析器 - OpenAI SDK版本
支持批量处理多个用户的轨迹数据
"""
import time
import os
import sys
import argparse
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# 导入OpenAI版本的分析器
from trajectory_ai_analyzer import TrajectoryAIAnalyzer, AnalysisConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BatchAnalysisManager:
    """批量分析管理器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化批量分析管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config.get('output', {}).get('base_dir', 'analysis_output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 验证必要的配置项
            if 'api' not in config:
                raise ValueError("配置文件缺少 'api' 部分")

            # 兼容性处理：bot_name -> model_name
            if 'bot_name' in config['api'] and 'model_name' not in config['api']:
                logger.info("检测到 bot_name，自动转换为 model_name")
                config['api']['model_name'] = config['api'].pop('bot_name')

            # 如果没有model_name，使用默认值
            if 'model_name' not in config['api']:
                logger.warning("配置文件缺少 model_name，使用默认值 GPT-4o")
                config['api']['model_name'] = 'GPT-4o'

            # 设置默认输出配置
            if 'output' not in config:
                config['output'] = {}

            output_defaults = {
                'base_dir': 'analysis_output',
                'output_level': 'summary',
                'max_preview_length': 500,
                'save_detailed_separately': True,
                'generate_markdown_report': True,
                'include_preview': False  # 默认关闭预览
            }

            for key, value in output_defaults.items():
                if key not in config['output']:
                    config['output'][key] = value

            return config
        except FileNotFoundError:
            logger.error(f"配置文件不存在: {config_path}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            sys.exit(1)

    def run_batch_analysis(self,
                           trajectory_file: str,
                           geocoded_file: Optional[str] = None,
                           output_name: Optional[str] = None) -> str:
        """运行批量分析"""
        logger.info("=" * 50)
        logger.info("批量轨迹数据AI分析")
        logger.info("=" * 50)

        # 获取API配置
        api_config = self.config.get('api', {})
        api_key = api_config.get('api_key') or os.getenv('POE_API_KEY')

        if not api_key:
            logger.error("未找到API密钥！")
            logger.error("请在config.yaml中设置或设置环境变量 POE_API_KEY")
            sys.exit(1)

        # 获取输出配置
        output_config = self.config.get('output', {})

        # 创建分析配置
        analysis_config = AnalysisConfig(
            api_key=api_key,
            model_name=api_config.get('model_name', 'GPT-4o'),
            max_tokens=api_config.get('max_tokens', 4000),
            temperature=api_config.get('temperature', 0.7),
            analysis_types=self.config.get('analysis', {}).get('enabled_types', None),
            output_level=output_config.get('output_level', 'summary'),
            max_preview_length=output_config.get('max_preview_length', 500),
            save_detailed_separately=output_config.get('save_detailed_separately', True),
            generate_markdown_report=output_config.get('generate_markdown_report', True)
        )

        logger.info(f"模型: {analysis_config.model_name} | 输出级别: {analysis_config.output_level}")

        # 创建分析器
        analyzer = TrajectoryAIAnalyzer(analysis_config)

        # 确定输出文件名
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"analysis_{timestamp}.json"

        output_path = self.output_dir / output_name

        # 运行分析
        try:
            results = analyzer.analyze_trajectory_data(
                trajectory_json=trajectory_file,
                geocoded_json=geocoded_file,
                output_path=str(output_path)
            )

            # 生成简洁摘要报告
            summary_path = self.output_dir / f"summary_{output_name.replace('.json', '.txt')}"
            self._generate_summary_report(results, summary_path)

            logger.info("=" * 50)
            logger.info("✅ 分析完成")
            logger.info(f"摘要: {output_path}")
            if analysis_config.save_detailed_separately:
                logger.info(f"详细: {str(output_path).replace('.json', '_detailed.json')}")
            if analysis_config.generate_markdown_report:
                logger.info(f"报告: {str(output_path).replace('.json', '.md')}")
            logger.info("=" * 50)

            return str(output_path)

        except Exception as e:
            logger.error(f"批量分析失败: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def _generate_summary_report(self, results: Dict, output_path: Path):
        """生成简洁摘要报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("轨迹AI分析摘要\n")
            f.write(f"时间: {results['analysis_timestamp'][:19]}\n")
            f.write(f"模型: {results['config']['model_name']}\n")
            f.write("-" * 40 + "\n")

            for user_id, user_results in results['results'].items():
                f.write(f"\n用户 {user_id}:\n")

                success_count = sum(1 for r in user_results.values() if 'error' not in r)
                total_count = len(user_results)
                f.write(f"  完成: {success_count}/{total_count}\n")

                # 只显示摘要信息
                for analysis_type, result in user_results.items():
                    if 'error' in result:
                        f.write(f"  ✗ {analysis_type[:20]}: 失败\n")
                    else:
                        summary = result.get('summary', '')[:80]
                        if summary:
                            f.write(f"  ✓ {analysis_type[:20]}: {summary}\n")
                        else:
                            f.write(f"  ✓ {analysis_type[:20]}: 完成\n")


def create_default_config():
    """创建默认配置文件"""
    default_config = {
        'api': {
            'model_name': 'GPT-4o',
            'max_tokens': 4000,
            'temperature': 0.7,
            'api_key': None
        },
        'analysis': {
            'enabled_types': [
    'temporal_comparative',           # 必选1：时间对比分析
    'spatial_differential',          # 必选2：空间差分分析
    'spatiotemporal_transitions',    # 必选3：时空转场与链条
    'cross_feature_insights',        # 必选4：跨维关联分析
    'anomaly_explanatory'            # 可选：解释性异常检测
    'meta_synthesis'  # 最终：综合洞察汇总
]
        },
        'output': {
            'base_dir': 'analysis_output1',
            'output_level': 'summary',  # summary/standard/detailed
            'max_preview_length': 500,
            'save_detailed_separately': True,
            'generate_markdown_report': True,
            'include_preview': False
        }
    }

    with open('config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, allow_unicode=True, default_flow_style=False)

    print("✅ 已创建配置文件: config.yaml")
    print("\n配置说明:")
    print("output_level选项:")
    print("  - summary: 仅保存摘要和要点")
    print("  - standard: 包含预览(前500字符)")
    print("  - detailed: 包含完整响应")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量轨迹数据分析管理器')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # init命令 - 初始化配置
    init_parser = subparsers.add_parser('init', help='初始化配置文件')
    init_parser.add_argument(
        '-o', '--output',
        default='config.yaml',
        help='配置文件输出路径（默认: config.yaml）'
    )

    # analyze命令 - 运行分析
    analyze_parser = subparsers.add_parser('analyze', help='运行批量分析')
    analyze_parser.add_argument(
        '-t', '--trajectory',
        required=True,
        help='轨迹数据JSON文件路径'
    )
    analyze_parser.add_argument(
        '-g', '--geocoded',
        help='地理编码JSON文件路径（可选）'
    )
    analyze_parser.add_argument(
        '-c', '--config',
        default='config.yaml',
        help='配置文件路径（默认: config.yaml）'
    )
    analyze_parser.add_argument(
        '-o', '--output',
        help='输出文件名（可选，默认使用时间戳）'
    )
    # 新增命令行参数
    analyze_parser.add_argument(
        '--level',
        choices=['summary', 'standard', 'detailed'],
        help='覆盖配置文件的输出级别设置'
    )
    analyze_parser.add_argument(
        '--model',
        help='覆盖配置文件的模型设置'
    )
    analyze_parser.add_argument(
        '--types',
        nargs='+',
        help='指定要运行的分析类型（空格分隔）'
    )

    # list命令 - 列出分析结果
    list_parser = subparsers.add_parser('list', help='列出所有分析结果')
    list_parser.add_argument(
        '-d', '--dir',
        default='analysis_output',
        help='分析输出目录（默认: analysis_output）'
    )
    list_parser.add_argument(
        '-n', '--limit',
        type=int,
        default=10,
        help='显示最近的N个结果（默认: 10）'
    )

    # view命令 - 查看分析结果
    view_parser = subparsers.add_parser('view', help='查看特定分析结果')
    view_parser.add_argument(
        'file',
        help='要查看的结果文件路径'
    )
    view_parser.add_argument(
        '-f', '--format',
        choices=['json', 'summary', 'markdown'],
        default='summary',
        help='显示格式（默认: summary）'
    )

    # compare命令 - 对比分析结果
    compare_parser = subparsers.add_parser('compare', help='对比两个分析结果')
    compare_parser.add_argument(
        'file1',
        help='第一个结果文件'
    )
    compare_parser.add_argument(
        'file2',
        help='第二个结果文件'
    )

    # clean命令 - 清理旧结果
    clean_parser = subparsers.add_parser('clean', help='清理旧的分析结果')
    clean_parser.add_argument(
        '-d', '--dir',
        default='analysis_output',
        help='分析输出目录（默认: analysis_output）'
    )
    clean_parser.add_argument(
        '-k', '--keep',
        type=int,
        default=5,
        help='保留最近的N个结果（默认: 5）'
    )
    clean_parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='跳过确认直接删除'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # 执行命令
    if args.command == 'init':
        # 创建默认配置文件
        default_config = {
            'api': {
                'base_url': 'https://api.poe.com/v1',
                'model_name': 'GPT-4o',
                'max_tokens': 4000,
                'temperature': 0.7,
                'api_key': None
            },
            'analysis': {
                'enabled_types': [
                    'behavior_pattern',
                    'mobility_summary',
                    'spatial_analysis',
                    'temporal_analysis',
                    'lifestyle_inference',
                    'recommendations'
                ],
                'batch_size': 5,
                'retry_count': 3,
                'retry_delay': 2
            },
            'output': {
                'base_dir': 'analysis_output',
                'output_level': 'summary',
                'max_preview_length': 500,
                'save_detailed_separately': True,
                'generate_markdown_report': True,
                'generate_summary_txt': True,
                'save_raw_responses': False
            },
            'data_processing': {
                'sample_size': None,
                'time_zone': 'UTC',
                'coordinate_precision': 6
            }
        }

        if os.path.exists(args.output):
            response = input(f"配置文件 {args.output} 已存在，是否覆盖？(y/n): ")
            if response.lower() != 'y':
                print("已取消")
                sys.exit(0)

        with open(args.output, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)

        print(f"✅ 配置文件已创建: {args.output}")
        print("\n请编辑配置文件，设置您的API密钥和其他参数")
        print("您也可以通过环境变量设置API密钥: export POE_API_KEY=your_key_here")

    elif args.command == 'analyze':
        # 检查轨迹文件是否存在
        if not os.path.exists(args.trajectory):
            logger.error(f"轨迹文件不存在: {args.trajectory}")
            sys.exit(1)

        # 检查地理编码文件（如果提供）
        if args.geocoded and not os.path.exists(args.geocoded):
            logger.warning(f"地理编码文件不存在: {args.geocoded}")
            args.geocoded = None

        # 检查配置文件
        if not os.path.exists(args.config):
            logger.error(f"配置文件不存在: {args.config}")
            logger.info("使用 'python batch_analyzer.py init' 创建默认配置文件")
            sys.exit(1)

        # 创建管理器
        manager = BatchAnalysisManager(args.config)

        # 覆盖配置（如果提供了命令行参数）
        if args.level:
            manager.config['output']['output_level'] = args.level
            logger.info(f"使用命令行指定的输出级别: {args.level}")

        if args.model:
            manager.config['api']['model_name'] = args.model
            logger.info(f"使用命令行指定的模型: {args.model}")

        if args.types:
            manager.config['analysis']['enabled_types'] = args.types
            logger.info(f"使用命令行指定的分析类型: {', '.join(args.types)}")

        # 运行分析
        results_file = manager.run_batch_analysis(
            trajectory_file=args.trajectory,
            geocoded_file=args.geocoded,
            output_name=args.output
        )

        print(f"\n✅ 分析完成！结果文件: {results_file}")

    elif args.command == 'list':
        # 列出分析结果
        if not os.path.exists(args.dir):
            print(f"目录不存在: {args.dir}")
            sys.exit(1)

        # 获取所有JSON文件
        json_files = []
        for file in os.listdir(args.dir):
            if file.endswith('.json') and not file.endswith('_detailed.json'):
                file_path = os.path.join(args.dir, file)
                stat = os.stat(file_path)
                json_files.append({
                    'file': file,
                    'path': file_path,
                    'size': stat.st_size,
                    'modified': stat.st_mtime
                })

        # 按修改时间排序
        json_files.sort(key=lambda x: x['modified'], reverse=True)

        if not json_files:
            print(f"没有找到分析结果文件在: {args.dir}")
            sys.exit(0)

        # 显示结果
        print(f"\n📊 最近的分析结果 (共 {len(json_files)} 个):\n")
        print(f"{'序号':<4} {'文件名':<40} {'大小':<10} {'修改时间':<20}")
        print("-" * 80)

        for i, file_info in enumerate(json_files[:args.limit], 1):
            size_str = f"{file_info['size'] / 1024:.1f}KB"
            time_str = datetime.fromtimestamp(file_info['modified']).strftime('%Y-%m-%d %H:%M:%S')
            print(f"{i:<4} {file_info['file']:<40} {size_str:<10} {time_str:<20}")

    elif args.command == 'view':
        # 查看分析结果
        if not os.path.exists(args.file):
            print(f"文件不存在: {args.file}")
            sys.exit(1)

        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if args.format == 'json':
                print(json.dumps(data, indent=2, ensure_ascii=False))

            elif args.format == 'summary':
                print("\n" + "=" * 60)
                print(f"📊 分析结果摘要")
                print("=" * 60)

                # 显示元数据
                if 'metadata' in data:
                    meta = data['metadata']
                    print(f"\n📅 分析时间: {meta.get('analysis_timestamp', 'N/A')}")
                    print(f"🤖 使用模型: {meta.get('model_used', 'N/A')}")
                    print(f"📝 输出级别: {meta.get('output_level', 'N/A')}")

                    if 'data_info' in meta:
                        info = meta['data_info']
                        print(f"\n📍 数据信息:")
                        print(f"  - 轨迹点数: {info.get('trajectory_points', 'N/A')}")
                        print(f"  - 时间范围: {info.get('date_range', 'N/A')}")
                        if info.get('has_geocoding'):
                            print(f"  - 包含地理编码: ✓")

                # 显示分析结果摘要
                if 'analysis_results' in data:
                    print(f"\n📊 分析类型:")
                    for analysis_type, content in data['analysis_results'].items():
                        print(f"\n  [{analysis_type}]")
                        if isinstance(content, dict):
                            if 'summary' in content:
                                print(f"    {content['summary'][:200]}...")
                            elif 'preview' in content:
                                print(f"    {content['preview'][:200]}...")
                        else:
                            print(f"    {str(content)[:200]}...")

                # 显示统计信息
                if 'statistics' in data:
                    stats = data['statistics']
                    print(f"\n📈 统计信息:")
                    print(f"  - 总分析数: {stats.get('total_analyses', 'N/A')}")
                    print(f"  - 成功数: {stats.get('successful_analyses', 'N/A')}")
                    print(f"  - 失败数: {stats.get('failed_analyses', 'N/A')}")
                    print(f"  - 总耗时: {stats.get('total_time', 'N/A')}秒")

            elif args.format == 'markdown':
                md_file = args.file.replace('.json', '.md')
                if os.path.exists(md_file):
                    with open(md_file, 'r', encoding='utf-8') as f:
                        print(f.read())
                else:
                    print(f"Markdown文件不存在: {md_file}")

        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
        except Exception as e:
            print(f"读取文件错误: {e}")

    elif args.command == 'compare':
        # 对比两个分析结果
        if not os.path.exists(args.file1):
            print(f"文件不存在: {args.file1}")
            sys.exit(1)
        if not os.path.exists(args.file2):
            print(f"文件不存在: {args.file2}")
            sys.exit(1)

        try:
            with open(args.file1, 'r', encoding='utf-8') as f:
                data1 = json.load(f)
            with open(args.file2, 'r', encoding='utf-8') as f:
                data2 = json.load(f)

            print("\n" + "=" * 60)
            print("📊 分析结果对比")
            print("=" * 60)

            # 对比元数据
            print("\n📋 基本信息对比:")
            print(f"{'项目':<20} {'文件1':<25} {'文件2':<25}")
            print("-" * 70)

            # 提取元数据
            meta1 = data1.get('metadata', {})
            meta2 = data2.get('metadata', {})

            items = [
                ('分析时间', meta1.get('analysis_timestamp', 'N/A'), meta2.get('analysis_timestamp', 'N/A')),
                ('使用模型', meta1.get('model_used', 'N/A'), meta2.get('model_used', 'N/A')),
                ('输出级别', meta1.get('output_level', 'N/A'), meta2.get('output_level', 'N/A')),
            ]

            for item, val1, val2 in items:
                print(f"{item:<20} {str(val1):<25} {str(val2):<25}")

            # 对比数据信息
            if 'data_info' in meta1 or 'data_info' in meta2:
                print("\n📍 数据信息对比:")
                info1 = meta1.get('data_info', {})
                info2 = meta2.get('data_info', {})

                data_items = [
                    ('轨迹点数', info1.get('trajectory_points', 'N/A'), info2.get('trajectory_points', 'N/A')),
                    ('时间范围', info1.get('date_range', 'N/A'), info2.get('date_range', 'N/A')),
                    ('包含地理编码', '✓' if info1.get('has_geocoding') else '✗',
                     '✓' if info2.get('has_geocoding') else '✗'),
                ]

                for item, val1, val2 in data_items:
                    diff_mark = " ⚠️" if val1 != val2 else ""
                    print(f"{item:<20} {str(val1):<25} {str(val2):<25}{diff_mark}")

            # 对比分析类型
            types1 = set(data1.get('analysis_results', {}).keys())
            types2 = set(data2.get('analysis_results', {}).keys())

            print("\n📊 分析类型对比:")
            common_types = types1 & types2
            only_in_1 = types1 - types2
            only_in_2 = types2 - types1

            if common_types:
                print(f"  共同分析: {', '.join(common_types)}")
            if only_in_1:
                print(f"  仅在文件1: {', '.join(only_in_1)}")
            if only_in_2:
                print(f"  仅在文件2: {', '.join(only_in_2)}")

            # 对比统计信息
            if 'statistics' in data1 or 'statistics' in data2:
                print("\n📈 统计对比:")
                stats1 = data1.get('statistics', {})
                stats2 = data2.get('statistics', {})

                stat_items = [
                    ('总分析数', stats1.get('total_analyses', 0), stats2.get('total_analyses', 0)),
                    ('成功数', stats1.get('successful_analyses', 0), stats2.get('successful_analyses', 0)),
                    ('失败数', stats1.get('failed_analyses', 0), stats2.get('failed_analyses', 0)),
                    ('总耗时(秒)', f"{stats1.get('total_time', 0):.2f}", f"{stats2.get('total_time', 0):.2f}"),
                ]

                for item, val1, val2 in stat_items:
                    print(f"{item:<20} {str(val1):<25} {str(val2):<25}")

        except Exception as e:
            print(f"对比分析错误: {e}")

    elif args.command == 'clean':
        # 清理旧结果
        if not os.path.exists(args.dir):
            print(f"目录不存在: {args.dir}")
            sys.exit(1)

        # 获取所有相关文件
        all_files = []
        for file in os.listdir(args.dir):
            file_path = os.path.join(args.dir, file)
            if os.path.isfile(file_path):
                # 检查是否为分析结果文件
                if file.endswith(('.json', '.md', '.txt')):
                    stat = os.stat(file_path)
                    all_files.append({
                        'path': file_path,
                        'name': file,
                        'modified': stat.st_mtime
                    })

        # 按修改时间排序
        all_files.sort(key=lambda x: x['modified'], reverse=True)

        # 识别要删除的文件组
        files_to_delete = []
        result_groups = {}

        # 按基础名称分组
        for file_info in all_files:
            base_name = file_info['name'].split('.')[0].replace('_detailed', '').replace('_summary', '')
            if base_name not in result_groups:
                result_groups[base_name] = []
            result_groups[base_name].append(file_info)

        # 保留最新的N组
        sorted_groups = sorted(result_groups.items(),
                               key=lambda x: max(f['modified'] for f in x[1]),
                               reverse=True)

        for i, (base_name, files) in enumerate(sorted_groups):
            if i >= args.keep:
                files_to_delete.extend(files)

        if not files_to_delete:
            print(f"没有需要清理的文件（保留最近 {args.keep} 个结果）")
            sys.exit(0)

        # 显示要删除的文件
        print(f"\n将删除以下 {len(files_to_delete)} 个文件:")
        for file_info in files_to_delete:
            print(f"  - {file_info['name']}")

        # 确认删除
        if not args.yes:
            response = input(f"\n确认删除这些文件？(y/n): ")
            if response.lower() != 'y':
                print("已取消")
                sys.exit(0)

        # 执行删除
        deleted_count = 0
        for file_info in files_to_delete:
            try:
                os.remove(file_info['path'])
                deleted_count += 1
            except Exception as e:
                logger.error(f"删除文件失败 {file_info['name']}: {e}")

        print(f"✅ 已删除 {deleted_count} 个文件")


if __name__ == "__main__":
    main()
