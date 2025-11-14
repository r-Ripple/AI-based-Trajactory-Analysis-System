#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨迹数据AI分析器 - OpenAI SDK版本
使用 openai.OpenAI() + base_url="https://api.poe.com/v1" 调用Poe API
"""
import time
import json
import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np

try:
    import openai
except ImportError:
    print("请安装 openai: pip install openai")
    exit(1)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """分析配置"""
    api_key: str
    model_name: str = "GPT-4o"
    max_tokens: int = 4000
    temperature: float = 0.7
    analysis_types: List[str] = None
    # 新增输出控制配置
    output_level: str = "summary"  # "summary", "standard", "detailed"
    max_preview_length: int = 500
    save_detailed_separately: bool = True
    generate_markdown_report: bool = True

    def __post_init__(self):
        if self.analysis_types is None:
            self.analysis_types = [
                "temporal_comparative",  # 必选1：时间对比分析
                "spatial_differential",  # 必选2：空间差分分析
                "spatiotemporal_transitions",  # 必选3：时空转场与链条
                "cross_feature_insights",  # 必选4：跨维关联分析
                "anomaly_explanatory",  # 可选：解释性异常检测
                "meta_synthesis"
            ]

class TrajectoryAIAnalyzer:
    """轨迹数据AI分析器 - 使用OpenAI SDK通过Poe"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        
        # 创建OpenAI客户端，指向Poe的base_url
        self.client = openai.OpenAI(
            api_key=config.api_key,
            base_url="https://api.poe.com/v1"
        )
        
        # 加载prompt模板
        self.prompt_templates = self._load_prompt_templates()

    def _run_meta_synthesis(self, user_id: str, previous_results: Dict[str, Any],
                            user_data: Dict, geocoded_data: Optional[List[Dict]]) -> Dict[str, Any]:
        """
        执行综合洞察汇总分析
        """
        logger.info(f"开始综合洞察汇总...")

        # 添加结果检查
        if not previous_results or all("error" in r for r in previous_results.values()):
            logger.error("没有可用的前置分析结果")
            return {
                "error": "无法进行综合分析：缺少前置分析结果",
                "analysis_type": "meta_synthesis",
                "timestamp": datetime.now().isoformat()
            }

        # 准备所有分析结果的摘要
        all_results_summary = self._prepare_all_results_summary(previous_results)

        # 检查摘要内容
        if not all_results_summary or len(all_results_summary) < 100:
            logger.warning(f"前置分析结果摘要过短: {len(all_results_summary)} 字符")

        # 构建prompt
        prompt_data = {
            "all_analysis_results": all_results_summary
        }

        prompt = self._build_prompt("meta_synthesis", prompt_data)

        # 调用AI API
        start_time = time.time()
        response = self._call_ai_api11(prompt)
        end_time = time.time()

        # 解析响应
        parsed_result = self._parse_ai_response(response, "meta_synthesis")
        parsed_result["processing_time"] = end_time - start_time
        parsed_result["based_on_analyses"] = list(previous_results.keys())

        return parsed_result

    def _prepare_all_results_summary(self, previous_results: Dict[str, Any]) -> str:
        """
        准备所有已完成分析的结果摘要
        """
        summary_parts = []

        for analysis_type, result in previous_results.items():
            if "error" in result:
                continue

            summary_parts.append(f"## {analysis_type.upper()}\n")

            # 修正：从正确的位置提取content
            # 优先从 structured_result.details 获取，其次从 raw_response，最后从 content
            content = ""
            if "structured_result" in result:
                content = result["structured_result"].get("details", "")
            if not content and "raw_response" in result:
                content = result.get("raw_response", "")
            if not content:
                content = result.get("content", "")

            # 如果还是没有内容，尝试构建一个基础摘要
            if not content:
                structured = result.get("structured_result", {})
                content_parts = []
                if structured.get("summary"):
                    content_parts.append(f"摘要: {structured['summary']}")
                if structured.get("highlights"):
                    content_parts.append("要点: " + "; ".join(structured['highlights']))
                if structured.get("metrics"):
                    content_parts.append("指标: " + str(structured['metrics']))
                content = "\n".join(content_parts) if content_parts else "无详细内容"

            # 根据输出级别决定包含多少内容
            if self.config.output_level == "summary":
                summary_parts.append(content[:800] + "...\n" if len(content) > 800 else content + "\n")
            elif self.config.output_level == "standard":
                summary_parts.append(content[:1500] + "...\n" if len(content) > 1500 else content + "\n")
            else:
                summary_parts.append(content + "\n")

            summary_parts.append("\n" + "=" * 80 + "\n\n")

        # 添加调试信息
        if not summary_parts:
            logger.warning("警告：没有找到任何可用的分析结果内容")
            return "未找到前置分析结果的详细内容"

        result_text = "".join(summary_parts)
        logger.info(f"准备的综合分析输入长度: {len(result_text)} 字符")

        return result_text
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """加载分析提示模板"""
        return {
            "system": """你是一位专业的人类移动行为和轨迹数据分析专家。专注发现"人眼难察觉但数据揭示"的高阶模式。

核心要求：
1. 从"分布描述"升级为"对比洞察"：必须产出带差值的发现
2. 避免泛化地名描述，聚焦空间-功能耦合和热点偏移
3. 用活动链（POI+时段+停留时长）与转场规则表达时空关系
4. 产出可检验的关联陈述，而非复述性描述
5. 异常必须具备解释价值，影响整体结论

输出格式：
- summary: 最重要发现（50字内）
- insights: 3-5条非直观洞察（含pattern/evidence/interpretation/novelty_score）
- metrics: 关键指标对比
- details: 深度分析

严格避免：泛述性结论、显而易见的模式、无对比的分布描述
""",




            "spatial_differential": """## 任务：空间模式差分分析（升级版）

### 输入数据
{spatial_data}
{geocoded_context}

**核心任务：热点偏移+方向性+空间差分，避免泛化地名**

必须输出：
1. **热点重心偏移分析**：
   - 周期性重心偏移：工作日vs周末的重心坐标差（米）
   - 偏移幅度：重心移动距离的统计分布
   - 偏移方向：主导偏移向量（角度+距离）

2. **走廊方向性分析**：
   - 方向玫瑰：8方向出行比例差异
   - 往返不对称性：去程vs回程的路径差异度
   - 方向偏好强度：角度集中度指标

3. **探索度与重访率差分**：
   - 新地点vs重访地点的时空分布差异
   - 探索半径扩张速率：按时间序列的半径增长率
   - 重访强度梯度：不同距离圈的重访频次递减率

4. **空间-功能耦合分析**：
   - POI功能区的停留时长分布差异
   - 功能转换的空间距离模式
   - 功能链条：A功能→B功能的空间跳跃距离分布

输出要求：用凸包/核密度差分量化，提供坐标差值和面积变化
""",

            "spatiotemporal_transitions": """## 任务：时空转场与活动链分析（重构版）

### 输入数据
{comprehensive_data}
{stays_summary}
{trips_summary}

**核心任务：重构行为模式分析，聚焦活动链与转场规则**

必须输出：
1. **活动链结构识别**：
   - 典型活动链：POI类型+时段+停留时长的组合模式
   - 链条完整度：起-承-转-合的完整性评分
   - 链条变异：标准链vs实际链的偏差分析

2. **转场规则量化**：
   - 条件概率矩阵：若在A(POI+时段)，则下一步到B的概率
   - 时间差依赖：转场时间间隔对目标选择的影响规律
   - 距离衰减函数：转场距离对选择概率的衰减模型

3. **日内策略模式**：
   - 早-中-晚三时段的策略差异
   - 时段内的微调策略：同类活动的时空微调模式
   - 策略应急性：计划外转场的应对模式

4. **因果偏移效应**：
   - 前效应：前一活动对当前选择的影响强度和衰减
   - 后效应：对未来活动的预期如何影响当前决策
   - 时滞分析：效应传递的时间延迟分布

输出要求：提供具体的转场概率数值、时间差统计、因果强度评分
""",

            "cross_feature_insights": """## 任务：跨维关联洞察（重构版）

            ### 输入数据
            {comprehensive_data}
            {pattern_data}

            **核心任务：从泛述转为可检验的关联陈述，体现AI洞察价值**

            必须输出：
            1. **半径-时间熵-夜间占比三元关联**：
               - 关联强度：活动半径↑ ⇔ 时间熵↑ ⇔ 夜间占比↑的相关系数
               - 阈值识别：关联发生突变的临界值点
               - 异常个体：偏离三元关联的异常模式及其解释

            2. **路径重复率与时间策略关联**：
               - 重复率-出发时差关系：路径熟悉度如何影响出发时间精准度
               - 探索成本-时间冗余关系：新路径探索的时间成本量化
               - 效率-灵活性权衡：重复vs探索的时空效率对比

            3. **停留时长-POI密度-回访间隔关联**：
               - 三变量关联模型：停留时长如何受POI密度和回访间隔影响
               - 饱和效应识别：POI密度对停留时长影响的饱和点
               - 记忆衰减效应：回访间隔对停留时长的影响规律

            4. **昼夜切换-距离选择-功能偏好关联**：
               - 时段-距离选择模式：昼夜时段如何影响出行距离选择
               - 功能-时段耦合强度：不同POI功能的时段偏好强度
               - 切换成本：昼夜功能切换的空间成本分析

            输出要求：每项关联必须提供相关系数、统计显著性、具体阈值数据
            ""","temporal_comparative": """## 任务：时间模式对比分析（升级版）

### 输入数据
{temporal_data}

**核心任务：从分布升级为对比，产出带差值的洞察**

必须输出：
1. **工作日vs周末对比**：
   - 出行次数差值：具体数字+百分比变化
   - 活跃时段偏移：峰值时间差异（小时）
   - 节律稳定性差异：熵值变化量

2. **正常日vs异常日识别**：
   - 异常日定义标准（基于出行量/时长/模式突变）
   - 异常日行为特征量化差异
   - 异常日对整体模式的影响程度

3. **Top/Bottom出行日分析**：
   - 最高/最低出行日的行为差异
   - 触发因子推断（天气/节假日/特殊事件）
   - 因果偏移效应：前/后日行为补偿机制

4. **节律转场逻辑**：
   - 出行启动的时间触发规律
   - 停留结束的时间决策模式
   - 昼夜切换的行为策略变化

输出要求：每项必须包含具体数值差异，避免"更活跃"等模糊描述
""",
            "anomaly_explanatory": """## 任务：解释性异常检测（精简版）

        ### 输入数据
        {pattern_data}

        **核心任务：仅识别影响整体结论解释的关键异常，避免罗列所有异常**

        限制输出1-2条解释性异常，要求：
        1. **异常识别标准**：
           - 偏离正常模式的统计阈值（3σ或分位数标准）
           - 异常对整体模式解释的影响权重评估

        2. **解释性价值评估**：
           - 剔除异常前后的核心指标差异（具体数值变化）
           - 异常是否改变对用户行为的整体判断
           - 异常背后的可能解释机制

        输出要求：
        - 只报告会改变整体结论的异常
        - 提供剔除前后的关键指标对比
        - 说明异常的解释价值和处理建议
        - 避免技术性异常罗列，聚焦行为解释意义
        """
            ,

            "meta_synthesis": """## 任务：综合洞察汇总与元分析

你已经完成了用户的多维轨迹分析，现在需要整合全部结果，提炼更高层次的行为规律与本质特征。但是需要特别注意的是，你输出的内容是给我的老板汇报的，他对于专业知识一窍不通，需要你把行为与特征用通俗易懂的语言进行描述



### 已完成的分析结果
{all_analysis_results}



## 核心任务：提炼跨维度的深层模式与行为本质,用更加通俗易懂的语言将下面的要求实现。



### 1. 跨维度模式整合 (Cross-Dimensional Pattern Integration)
**目标**：在不同分析维度之间找出呼应、矛盾或互补的规律，让结论之间形成逻辑闭环。

分析方向：
- **时间–空间耦合**：时间规律和空间分布如何相互解释？  
  - 例：工作日峰值时间是否与空间活动重心变化对应？  
  - 周末出行时间变化是否伴随探索半径扩大？
- **转场–关联耦合**：时空转换规律是否验证了跨维度的假设？  
  - 例：停留时长与POI密度的关系是否能在真实迁移动线中体现？
- **矛盾与张力识别**：不同维度间有无表面冲突或反常组合？  
  - 例：空间集中却时间分散，说明同地但作息多样等。




### 2. 行为本质归因 (Behavioral Essence Attribution)

**目标**：深入数据背后，解释“这个人为什么这样移动”。  

需完成以下分析：

- **主导约束识别**  
  判断时间约束、空间约束、功能约束中哪个最能解释其行为规律。  
  - 给出评分（1–10分），并简述理由（基于哪些数据）。
  



### 3. 核心行为画像 (Core Behavioral Profile)

**目标**：在 100 字以内，用自然平实的语言刻画该用户的核心移动特征。  
要求：
- 基于上文综合结论  
- 不使用模板化句式  
- 语言流畅




### 输出风格与语言要求

- 逻辑清晰但语言通顺、自然、人性化，尽量不包含过于学术的语言 
- 避免僵硬句式（如“体现出……特征”），使用更流畅表达  
- 最后附上一个 **“轻量版用户画像摘要”**，写成 2–3 句小结


"""
        }


    def _call_ai_api(self, prompt: str) -> str:
        """
        使用OpenAI SDK调用Poe API
        """
        try:
            # 根据输出级别调整prompt
            if self.config.output_level == "summary":
                prompt = f"{prompt}\n\n请提供简洁的分析，重点突出关键发现。每个部分控制在200字以内。"
            elif self.config.output_level == "standard":
                prompt = f"{prompt}\n\n请提供标准分析，平衡详细度和可读性。"

            # 使用OpenAI SDK的chat completions API
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": self.prompt_templates["system"]},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens if self.config.output_level == "detailed" else min(2000,
                                                                                                     self.config.max_tokens),
                temperature=self.config.temperature
            )

            # 提取响应文本
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"AI API调用失败: {str(e)}")
            return f"AI分析失败: {str(e)}"
    def _call_ai_api11(self, prompt: str) -> str:
        """
        使用OpenAI SDK调用Poe API
        """
        try:
            # 根据输出级别调整prompt
            if self.config.output_level == "summary":
                prompt = f"{prompt}\n\n请提供简洁的分析，重点突出关键发现。每个部分控制在200字以内。"
            elif self.config.output_level == "standard":
                prompt = f"{prompt}\n\n请提供标准分析，平衡详细度和可读性。"

            # 使用OpenAI SDK的chat completions API
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": self.prompt_templates["system"]},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.5
            )

            # 提取响应文本
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"AI API调用失败: {str(e)}")
            return f"AI分析失败: {str(e)}"
    def _structure_ai_response(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """将AI响应结构化"""
        # 尝试从响应中提取结构化内容
        structured = {
            "title": f"{analysis_type.replace('_', ' ').title()}",
            "summary": "",
            "highlights": [],
            "metrics": {},
            "details": response
        }

        # 简单的文本解析，提取摘要和要点
        lines = response.split('\n')
        summary_found = False
        highlights_section = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 识别摘要部分
            if '摘要' in line or 'Summary' in line.lower() or '总结' in line:
                summary_found = True
                continue

            if summary_found and not structured["summary"] and line:
                structured["summary"] = line[:200]  # 限制摘要长度
                summary_found = False

            # 识别要点部分
            if '要点' in line or 'highlights' in line.lower() or '关键发现' in line:
                highlights_section = True
                continue

            if highlights_section and line.startswith(('•', '-', '*', '1', '2', '3')):
                highlight = line.lstrip('•-*123456789. ')[:50]
                if highlight and len(structured["highlights"]) < 5:
                    structured["highlights"].append(highlight)

        # 如果没有找到摘要，生成一个
        if not structured["summary"]:
            structured["summary"] = response[:150].split('.')[0] + '.'

        # 如果没有要点，从详情中提取
        if not structured["highlights"] and len(response) > 100:
            sentences = response.split('。')[:3]
            structured["highlights"] = [s.strip()[:50] for s in sentences if s.strip()]

        return structured

    def analyze_trajectory_data(self,
                                trajectory_json: str,
                                geocoded_json: str = None,
                                output_path: str = "analysis_results.json") -> Dict[str, Any]:
        """
        分析轨迹数据

        Args:
            trajectory_json: 轨迹JSON文件路径
            geocoded_json: 地理编码JSON文件路径（可选）
            output_path: 输出文件路径

        Returns:
            分析结果字典
        """
        # 加载数据
        logger.info("加载轨迹数据...")
        with open(trajectory_json, 'r', encoding='utf-8') as f:
            traj_data = json.load(f)

        geocoded_data = None
        if geocoded_json and os.path.exists(geocoded_json):
            logger.info("加载地理编码数据...")
            with open(geocoded_json, 'r', encoding='utf-8') as f:
                geocoded_data = json.load(f)

        results = {}
        detailed_results = {}  # 存储详细结果

        # 对每个用户进行分析
        for user_data in traj_data.get('users', []):
            user_id = user_data['user_id']
            logger.info(f"分析用户 {user_id}...")

            # 为用户匹配地理编码数据
            user_geocoded = self._extract_user_geocoded_data(user_id, geocoded_data)

            # 执行各类分析
            user_results = {}
            user_detailed = {}

            # 分离meta_synthesis和其他分析类型
            regular_analyses = [a for a in self.config.analysis_types if a != "meta_synthesis"]
            has_meta_synthesis = "meta_synthesis" in self.config.analysis_types

            # 执行常规分析
            for analysis_type in regular_analyses:
                logger.info(f"  执行 {analysis_type} 分析...")
                try:
                    result = self._run_single_analysis(
                        analysis_type, user_data, user_geocoded
                    )

                    # 分离摘要和详细结果
                    summary_result = {
                        "analysis_type": result["analysis_type"],
                        "timestamp": result["timestamp"],
                        "summary": result["structured_result"]["summary"],
                        "highlights": result["structured_result"]["highlights"],
                        "metrics": result["structured_result"]["metrics"]
                    }
                    user_results[analysis_type] = summary_result

                    # 保存详细结果
                    if self.config.save_detailed_separately:
                        user_detailed[analysis_type] = result

                    # 添加延迟避免API请求过快
                    time.sleep(0.5)

                except Exception as e:
                    logger.error(f"分析 {analysis_type} 失败: {str(e)}")
                    user_results[analysis_type] = {"error": str(e)}

            # 如果启用了meta_synthesis，最后执行综合分析
            if has_meta_synthesis:
                logger.info(f"  执行 meta_synthesis 综合分析...")
                try:
                    # 准备所有已完成分析的结果
                    meta_result = self._run_meta_synthesis(
                        user_id, user_detailed, user_data, user_geocoded
                    )

                    # 分离摘要和详细结果
                    summary_result = {
                        "analysis_type": "meta_synthesis",
                        "timestamp": meta_result.get("timestamp"),
                        "based_on_analyses": meta_result.get("based_on_analyses", []),
                        "processing_time": meta_result.get("processing_time", 0),
                        "content": meta_result.get("content", "")[:500] + "..." if len(
                            meta_result.get("content", "")) > 500 else meta_result.get("content", "")
                    }
                    user_results["meta_synthesis"] = summary_result

                    # 保存详细结果
                    if self.config.save_detailed_separately:
                        user_detailed["meta_synthesis"] = meta_result

                except Exception as e:
                    logger.error(f"  meta_synthesis 分析失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    user_results["meta_synthesis"] = {"error": str(e)}

            results[user_id] = user_results
            if self.config.save_detailed_separately:
                detailed_results[user_id] = user_detailed

        # 保存摘要结果
        output_data = {
            "analysis_timestamp": datetime.now().isoformat(),
            "config": {
                "model_name": self.config.model_name,
                "output_level": self.config.output_level,
                "analysis_types": self.config.analysis_types
            },
            "results": results
        }
        detailed_path = None
        md_path = None
        logger.info(f"保存摘要结果到: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        # 保存详细结果到单独文件
        if self.config.save_detailed_separately and detailed_results:
            detailed_path = output_path.replace('.json', '_detailed.json')
            logger.info(f"保存详细结果到: {detailed_path}")
            with open(detailed_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "analysis_timestamp": datetime.now().isoformat(),
                    "config": {
                        "model_name": self.config.model_name,
                        "output_level": self.config.output_level
                    },
                    "detailed_results": detailed_results
                }, f, ensure_ascii=False, indent=2)

        # 生成Markdown报告
        if self.config.generate_markdown_report:
            md_path = output_path.replace('.json', '.md')
            logger.info(f"生成Markdown报告: {md_path}")
            self._generate_markdown_report(output_data, md_path)
        logger.info("=" * 50)
        logger.info(f"✅ 分析完成！")
        logger.info(f"📊 摘要结果: {output_path}")
        if self.config.save_detailed_separately and detailed_results:
            logger.info(f"📋 详细结果: {detailed_path}")
        if self.config.generate_markdown_report:
            logger.info(f"📄 Markdown报告: {md_path}")
        logger.info("=" * 50)

        return output_data

    def _generate_markdown_report(self, data: Dict, output_path: str):
        """生成Markdown格式报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 轨迹数据AI分析报告\n\n")
            f.write(f"**生成时间**: {data['analysis_timestamp']}\n")
            f.write(f"**使用模型**: {data['config']['model_name']}\n\n")

            for user_id, user_results in data['results'].items():
                f.write(f"## 用户 {user_id}\n\n")

                for analysis_type, result in user_results.items():
                    if 'error' in result:
                        f.write(f"### ❌ {analysis_type.replace('_', ' ').title()}\n")
                        f.write(f"错误: {result['error']}\n\n")
                        continue

                    f.write(f"### ✅ {analysis_type.replace('_', ' ').title()}\n\n")

                    # 特殊处理 meta_synthesis
                    if analysis_type == "meta_synthesis":
                        if 'content' in result and result['content']:
                            # 如果内容被截断，尝试获取完整内容
                            content = result['content']
                            if content.endswith('...'):
                                f.write(f"**综合分析摘要**:\n{content}\n\n")
                                f.write("*注：完整分析请查看详细结果文件*\n\n")
                            else:
                                f.write(f"{content}\n\n")

                        if 'based_on_analyses' in result:
                            f.write(f"**基于分析类型**: {', '.join(result['based_on_analyses'])}\n\n")
                    else:
                        # 原有逻辑处理其他分析类型
                        if 'summary' in result and result['summary']:
                            f.write(f"**摘要**: {result['summary']}\n\n")

                        if 'highlights' in result and result['highlights']:
                            f.write("**关键发现**:\n")
                            for highlight in result['highlights']:
                                f.write(f"- {highlight}\n")
                            f.write("\n")

                        if 'metrics' in result and result['metrics']:
                            f.write("**关键指标**:\n")
                            for key, value in result['metrics'].items():
                                f.write(f"- {key}: {value}\n")
                            f.write("\n")

                    f.write("---\n\n")
    def _extract_user_geocoded_data(self, user_id: str, geocoded_data: Optional[Dict]) -> Optional[List[Dict]]:
        """提取特定用户的地理编码数据"""
        if not geocoded_data:
            return None
        
        user_results = []
        for result in geocoded_data.get('results', []):
            if result.get('additional_info', {}).get('user_id') == user_id:
                user_results.append(result)
        
        return user_results if user_results else None

    def _run_single_analysis(self,
                             analysis_type: str,
                             user_data: Dict,
                             geocoded_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """运行单项分析"""

        # 准备分析数据
        analysis_data = self._prepare_analysis_data(analysis_type, user_data, geocoded_data)

        # 构建prompt
        prompt = self._build_prompt(analysis_type, analysis_data)

        # 调用AI API
        response = self._call_ai_api(prompt)

        # 结构化响应
        structured_result = self._structure_ai_response(response, analysis_type)

        # 根据输出级别处理结果
        result = {
            "analysis_type": analysis_type,
            "timestamp": datetime.now().isoformat(),
            "structured_result": structured_result,
            "data_summary": self._get_data_summary(user_data)
        }

        # 根据配置决定是否包含完整响应
        if self.config.output_level == "detailed":
            result["raw_response"] = response
        elif self.config.output_level == "standard":
            result["raw_response"] = response[:self.config.max_preview_length] + "..." if len(
                response) > self.config.max_preview_length else response
        # summary级别不包含raw_response

        return result
    def _prepare_analysis_data(self, 
                               analysis_type: str, 
                               user_data: Dict, 
                               geocoded_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """准备特定分析类型的数据"""
        
        base_data = {
            "n_points": user_data.get('n_points', 0),
            "time_metrics": json.dumps(user_data.get('time_metrics', {}), indent=2),
            "space_metrics": json.dumps(user_data.get('space_metrics', {}), indent=2),
            "spatiotemporal_metrics": json.dumps(user_data.get('spatiotemporal_metrics', {}), indent=2)
        }
        
        # 处理停留点信息
        stays = user_data.get('stays', [])
        stays_summary = self._summarize_stays(stays[:10])
        base_data["stays_summary"] = stays_summary
        
        # 处理出行信息
        trips = user_data.get('trips', [])
        trips_summary = self._summarize_trips(trips[:10])
        base_data["trips_summary"] = trips_summary
        
        # 时间跨度
        if stays:
            time_span = f"从 {stays[0].get('t_start', 'N/A')} 到 {stays[-1].get('t_end', 'N/A')}"
        else:
            time_span = "无法确定"
        base_data["time_span"] = time_span
        
        # 统计信息
        base_data.update({
            "n_stays": len(stays),
            "n_trips": len(trips)
        })
        
        # 地理编码样本
        if geocoded_data:
            geocoded_sample = json.dumps(geocoded_data[:3], indent=2, ensure_ascii=False)
            base_data["geocoded_sample"] = geocoded_sample
            base_data["geocoded_context"] = self._extract_geocoded_context(geocoded_data)
        else:
            base_data["geocoded_sample"] = "无地理编码数据"
            base_data["geocoded_context"] = "无地理编码数据"
        
        # 根据分析类型提供特定数据
        if analysis_type == "temporal_comparative":
            base_data["temporal_data"] = self._prepare_comparative_temporal_data(user_data)
        elif analysis_type == "spatial_differential":
            base_data["spatial_data"] = self._prepare_differential_spatial_data(user_data)
            base_data["geocoded_context"] = self._extract_geocoded_context(
                geocoded_data) if geocoded_data else "无地理编码上下文"
        elif analysis_type == "spatiotemporal_transitions":
            base_data.update({
                "comprehensive_data": self._prepare_comprehensive_data(user_data, geocoded_data),
                "stays_summary": self._summarize_stays(user_data.get('stays', [])),
                "trips_summary": self._summarize_trips(user_data.get('trips', []))
            })
        elif analysis_type == "cross_feature_insights":
            base_data.update({
                "comprehensive_data": self._prepare_comprehensive_data(user_data, geocoded_data),
                "pattern_data": self._prepare_pattern_data(user_data)
            })
        elif analysis_type == "anomaly_explanatory":
            base_data["pattern_data"] = self._prepare_pattern_data(user_data)
        elif analysis_type == "meta_synthesis":
            # meta_synthesis的数据准备在_run_meta_synthesis中单独处理
            pass
        
        return base_data
    
    def _summarize_stays(self, stays: List[Dict]) -> str:
        """总结停留点信息"""
        if not stays:
            return "无停留点数据"
        
        summary = []
        for i, stay in enumerate(stays):
            duration_h = stay.get('duration_s', 0) / 3600
            summary.append(
                f"停留点{i+1}: 位置({stay.get('lat', 'N/A'):.4f}, {stay.get('lon', 'N/A'):.4f}), "
                f"时长{duration_h:.1f}小时, 时间{stay.get('t_start', 'N/A')}"
            )
        
        return "\n".join(summary)
    
    def _summarize_trips(self, trips: List[Dict]) -> str:
        """总结出行信息"""
        if not trips:
            return "无出行数据"
        
        summary = []
        for i, trip in enumerate(trips):
            duration_h = trip.get('duration_s', 0) / 3600
            distance_km = trip.get('distance_m', 0) / 1000
            summary.append(
                f"出行{i+1}: 距离{distance_km:.2f}km, 时长{duration_h:.1f}小时, "
                f"起点({trip.get('start_lat', 'N/A'):.4f}, {trip.get('start_lon', 'N/A'):.4f}), "
                f"时间{trip.get('t_start', 'N/A')}"
            )
        
        return "\n".join(summary)
    
    def _prepare_spatial_data(self, user_data: Dict) -> str:
        """准备空间分析数据"""
        spatial = user_data.get('space_metrics', {})
        return json.dumps({
            "convex_hull_area_m2": spatial.get('hull_area_m2'),
            "radius_of_gyration_m": spatial.get('radius_of_gyration_m'),
            "ellipse_parameters": {
                "sx_m": spatial.get('ellipse_sx_m'),
                "sy_m": spatial.get('ellipse_sy_m'),
                "theta_deg": spatial.get('ellipse_theta_deg')
            },
            "kde_hotspots": spatial.get('kde_hotspots_utm32650', [])
        }, indent=2)

    def _prepare_comparative_temporal_data(self, user_data: Dict) -> str:
        """准备对比时间分析数据"""
        temporal = user_data.get('time_metrics', {})
        trips = user_data.get('trips', [])

        # 解析时间戳，区分工作日和周末
        weekday_trips = []
        weekend_trips = []

        for trip in trips:
            try:
                # 解析ISO 8601格式时间戳 (如: "2008-10-23T10:53:04.000001+08:00")
                trip_time = pd.to_datetime(trip.get('t_start', ''))
                weekday = trip_time.weekday()

                if weekday < 5:  # 0-4为周一到周五
                    weekday_trips.append(trip)
                else:  # 5-6为周六周日
                    weekend_trips.append(trip)
            except Exception as e:
                logger.debug(f"时间解析失败: {trip.get('t_start', 'N/A')}, 错误: {e}")
                continue

        # 同时处理stays数据用于更全面的对比分析
        stays = user_data.get('stays', [])
        weekday_stays = []
        weekend_stays = []

        for stay in stays:
            try:
                stay_time = pd.to_datetime(stay.get('t_start', ''))
                if stay_time.weekday() < 5:
                    weekday_stays.append(stay)
                else:
                    weekend_stays.append(stay)
            except:
                continue

        # 计算对比指标 - 扩展版
        weekday_trip_count = len(weekday_trips)
        weekend_trip_count = len(weekend_trips)
        weekday_stay_count = len(weekday_stays)
        weekend_stay_count = len(weekend_stays)

        # 计算差值和变化率
        trip_count_diff = weekend_trip_count - weekday_trip_count / 5 * 2 if weekday_trip_count > 0 else weekend_trip_count
        trip_count_change_pct = (trip_count_diff / max(weekday_trip_count / 5 * 2,
                                                       1)) * 100 if weekday_trip_count > 0 else 0

        # 计算平均出行时长对比
        weekday_trip_durations = [trip.get('duration_s', 0) for trip in weekday_trips]
        weekend_trip_durations = [trip.get('duration_s', 0) for trip in weekend_trips]

        weekday_avg_duration = np.mean(weekday_trip_durations) if weekday_trip_durations else 0
        weekend_avg_duration = np.mean(weekend_trip_durations) if weekend_trip_durations else 0
        duration_diff = weekend_avg_duration - weekday_avg_duration

        # 计算活跃时段峰值差异
        weekday_hours = [pd.to_datetime(trip.get('t_start', '')).hour for trip in weekday_trips if trip.get('t_start')]
        weekend_hours = [pd.to_datetime(trip.get('t_start', '')).hour for trip in weekend_trips if trip.get('t_start')]

        weekday_peak_hour = max(set(weekday_hours), key=weekday_hours.count) if weekday_hours else None
        weekend_peak_hour = max(set(weekend_hours), key=weekend_hours.count) if weekend_hours else None
        peak_hour_shift = (weekend_peak_hour - weekday_peak_hour) if (weekday_peak_hour and weekend_peak_hour) else 0

        return json.dumps({
            "temporal_comparison": {
                "weekday_vs_weekend": {
                    "trip_count": {
                        "weekday_total": weekday_trip_count,
                        "weekend_total": weekend_trip_count,
                        "weekday_daily_avg": weekday_trip_count / 5 if weekday_trip_count > 0 else 0,
                        "weekend_daily_avg": weekend_trip_count / 2 if weekend_trip_count > 0 else 0,
                        "difference": trip_count_diff,
                        "change_percentage": round(trip_count_change_pct, 2)
                    },
                    "trip_duration": {
                        "weekday_avg_seconds": round(weekday_avg_duration, 2),
                        "weekend_avg_seconds": round(weekend_avg_duration, 2),
                        "difference_seconds": round(duration_diff, 2),
                        "change_percentage": round((duration_diff / max(weekday_avg_duration, 1)) * 100,
                                                   2) if weekday_avg_duration > 0 else 0
                    },
                    "activity_peak": {
                        "weekday_peak_hour": weekday_peak_hour,
                        "weekend_peak_hour": weekend_peak_hour,
                        "peak_shift_hours": peak_hour_shift
                    },
                    "stay_pattern": {
                        "weekday_stay_count": weekday_stay_count,
                        "weekend_stay_count": weekend_stay_count
                    }
                }
            },
            "base_temporal_data": {
                "total_trip_count": temporal.get('n_trips'),
                "hourly_distribution": temporal.get('trip_start_hist_24h'),
                "time_entropy_normalized": temporal.get('time_entropy_hourly_norm'),
                "day_night_ratio": temporal.get('day_night_ratio')
            }
        }, indent=2, default=float)

    def _prepare_differential_spatial_data(self, user_data: Dict) -> str:
        """准备差分空间分析数据"""
        spatial = user_data.get('space_metrics', {})
        stays = user_data.get('stays', [])

        # 计算热点重心（简化版，实际应基于密度分析）
        if stays:
            lats = [stay.get('lat', 0) for stay in stays if stay.get('lat')]
            lons = [stay.get('lon', 0) for stay in stays if stay.get('lon')]

            if lats and lons:
                center_lat = np.mean(lats)
                center_lon = np.mean(lons)
                max_distance = max([
                    np.sqrt((lat - center_lat) ** 2 + (lon - center_lon) ** 2)
                    for lat, lon in zip(lats, lons)
                ]) if len(lats) > 1 else 0
            else:
                center_lat = center_lon = max_distance = 0
        else:
            center_lat = center_lon = max_distance = 0

        return json.dumps({
            "spatial_differential": {
                "hotspot_center": {
                    "lat": center_lat,
                    "lon": center_lon
                },
                "max_deviation_distance": max_distance,
                "spatial_concentration_index": 1 / (1 + max_distance) if max_distance > 0 else 1
            },
            "base_spatial_data": {
                "convex_hull_area_m2": spatial.get('hull_area_m2'),
                "radius_of_gyration_m": spatial.get('radius_of_gyration_m'),
                "ellipse_parameters": {
                    "sx_m": spatial.get('ellipse_sx_m'),
                    "sy_m": spatial.get('ellipse_sy_m'),
                    "theta_deg": spatial.get('ellipse_theta_deg')
                }
            }
        }, indent=2, default=float)
    
    def _prepare_comprehensive_data(self, user_data: Dict, geocoded_data: Optional[List[Dict]]) -> str:
        """准备综合分析数据"""
        data = {
            "trajectory_stats": user_data.get('time_metrics', {}),
            "spatial_patterns": user_data.get('space_metrics', {}),
            "stay_patterns": user_data.get('spatiotemporal_metrics', {}),
            "top_stays": user_data.get('spatiotemporal_metrics', {}).get('top5_stays', [])
        }
        
        if geocoded_data:
            poi_types = []
            for result in geocoded_data:
                pois = result.get('nearby_pois', [])
                poi_types.extend([poi.get('type', 'unknown') for poi in pois])
            
            poi_counts = {}
            for poi_type in poi_types:
                poi_counts[poi_type] = poi_counts.get(poi_type, 0) + 1
            
            data["poi_analysis"] = {
                "total_unique_pois": len(set(poi_types)),
                "poi_type_distribution": dict(sorted(poi_counts.items(), key=lambda x: x[1], reverse=True)[:10])
            }
        
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    def _prepare_pattern_data(self, user_data: Dict) -> str:
        """准备模式检测数据"""
        trips = user_data.get('trips', [])
        stays = user_data.get('stays', [])
        
        if trips:
            trip_durations = [trip.get('duration_s', 0) for trip in trips]
            trip_distances = [trip.get('distance_m', 0) for trip in trips]
            
            trip_stats = {
                "duration_mean": np.mean(trip_durations),
                "duration_std": np.std(trip_durations),
                "distance_mean": np.mean(trip_distances),
                "distance_std": np.std(trip_distances),
                "trip_count": len(trips)
            }
        else:
            trip_stats = {}
        
        if stays:
            stay_durations = [stay.get('duration_s', 0) for stay in stays]
            stay_stats = {
                "duration_mean": np.mean(stay_durations),
                "duration_std": np.std(stay_durations),
                "stay_count": len(stays)
            }
        else:
            stay_stats = {}
        
        return json.dumps({
            "trip_statistics": trip_stats,
            "stay_statistics": stay_stats,
            "sample_trips": trips[:5],
            "sample_stays": stays[:5]
        }, indent=2, default=float)
    
    def _prepare_summary_data(self, user_data: Dict) -> str:
        """准备总结数据"""
        return json.dumps({
            "key_metrics": {
                "total_points": user_data.get('n_points'),
                "total_trips": user_data.get('time_metrics', {}).get('n_trips'),
                "total_stays": len(user_data.get('stays', [])),
                "activity_area_m2": user_data.get('space_metrics', {}).get('hull_area_m2'),
                "movement_radius_m": user_data.get('space_metrics', {}).get('radius_of_gyration_m')
            },
            "patterns_summary": {
                "time_regularity": user_data.get('time_metrics', {}).get('time_entropy_hourly_norm'),
                "spatial_concentration": "high" if user_data.get('space_metrics', {}).get('radius_of_gyration_m', 0) < 5000 else "low",
                "activity_balance": user_data.get('time_metrics', {}).get('day_night_ratio')
            }
        }, indent=2, default=float)
    
    def _extract_geocoded_context(self, geocoded_data: List[Dict]) -> str:
        """提取地理编码上下文信息"""
        if not geocoded_data:
            return "无地理编码上下文"
        
        contexts = []
        for data in geocoded_data[:5]:
            addr = data.get('formatted_address', 'N/A')
            pois = data.get('nearby_pois', [])
            poi_names = [poi.get('name', 'unknown') for poi in pois[:3]]
            
            contexts.append(f"地址: {addr}, 附近POI: {', '.join(poi_names)}")
        
        return "\n".join(contexts)
    
    def _build_prompt(self, analysis_type: str, data: Dict[str, Any]) -> str:
        """构建分析提示"""
        task_prompt = self.prompt_templates.get(analysis_type, "")
        if not task_prompt:
            return ""
        
        # 格式化任务提示
        try:
            formatted_prompt = task_prompt.format(**data)
            return formatted_prompt
        except KeyError as e:
            logger.warning(f"Prompt格式化缺少键: {e}")
            return task_prompt
    
    def _parse_ai_response(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """解析AI响应"""
        return {
            "analysis_type": analysis_type,
            "content": response,
            "length": len(response),
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_data_summary(self, user_data: Dict) -> Dict[str, Any]:
        """获取数据摘要"""
        return {
            "user_id": user_data.get('user_id'),
            "n_points": user_data.get('n_points'),
            "n_stays": len(user_data.get('stays', [])),
            "n_trips": len(user_data.get('trips', [])),
            "has_labels": user_data.get('label_summary') is not None
        }

# 主函数示例
def main():
    """主函数示例"""
    import argparse

    parser = argparse.ArgumentParser(description='轨迹数据AI分析器')
    parser.add_argument('-t', '--trajectory', required=True, help='轨迹JSON文件路径')
    parser.add_argument('-g', '--geocoded', help='地理编码JSON文件路径')
    parser.add_argument('-o', '--output', default='analysis_results.json', help='输出文件路径')
    parser.add_argument('--level', choices=['summary', 'standard', 'detailed'],
                        default='summary', help='输出详细级别')
    parser.add_argument('--model', default='GPT-4o', help='使用的模型名称')
    parser.add_argument('--no-markdown', action='store_true', help='不生成Markdown报告')
    parser.add_argument('--no-detailed', action='store_true', help='不保存详细结果文件')

    args = parser.parse_args()

    # 配置
    config = AnalysisConfig(
        api_key=os.getenv("POE_API_KEY"),
        model_name=args.model,
        output_level=args.level,
        generate_markdown_report=not args.no_markdown,
        save_detailed_separately=not args.no_detailed and args.level != 'detailed',
        analysis_types=[
            "behavior_pattern",
            "mobility_summary",
            "spatial_analysis",
            "temporal_analysis",
            "lifestyle_inference",
            "recommendations"
        ]
    )

    if not config.api_key:
        print("请设置环境变量 POE_API_KEY")
        return

    # 创建分析器
    analyzer = TrajectoryAIAnalyzer(config)

    # 分析数据
    try:
        results = analyzer.analyze_trajectory_data(
            trajectory_json=args.trajectory,
            geocoded_json=args.geocoded,
            output_path=args.output
        )

        print(f"✅ 分析完成！")
        print(f"📊 输出级别: {args.level}")
        print(f"📁 结果文件: {args.output}")

        if config.save_detailed_separately and args.level != 'detailed':
            print(f"📁 详细结果: {args.output.replace('.json', '_detailed.json')}")

        if config.generate_markdown_report:
            print(f"📄 Markdown报告: {args.output.replace('.json', '.md')}")

    except Exception as e:
        logger.error(f"分析过程出错: {str(e)}")

if __name__ == "__main__":
    main()