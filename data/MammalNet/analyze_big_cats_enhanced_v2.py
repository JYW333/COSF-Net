import json
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import statistics
from datetime import datetime
import pandas as pd


def calculate_duration_distribution(durations):
    """
    计算时长分布
    返回分布字典
    """
    distribution = {
        '<3s': 0,
        '3-5s': 0,
        '5-10s': 0,
        '10-20s': 0,
        '20-60s': 0,
        '>60s': 0
    }

    for duration in durations:
        if duration < 3:
            distribution['<3s'] += 1
        elif duration < 5:
            distribution['3-5s'] += 1
        elif duration < 10:
            distribution['5-10s'] += 1
        elif duration < 20:
            distribution['10-20s'] += 1
        elif duration < 60:
            distribution['20-60s'] += 1
        else:
            distribution['>60s'] += 1

    return distribution


def format_duration(seconds):
    """格式化时长显示"""
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}小时"


def format_distribution_string(distribution, total):
    """格式化分布字符串"""
    if total == 0:
        return "无数据"

    parts = []
    for key in ['<3s', '3-5s', '5-10s', '10-20s', '20-60s', '>60s']:
        count = distribution.get(key, 0)
        if count > 0:
            pct = (count / total) * 100
            parts.append(f"{key}:{count}个({pct:.1f}%)")

    return ", ".join(parts)


def analyze_big_cats_with_time(json_file_path: str, include_medium_cats: bool = False):
    """
    分析MammalNet数据集中大型猫科动物的视频信息（增强版：包含详细时间分析）

    Args:
        json_file_path: JSON文件路径
        include_medium_cats: 是否包含中型猫科动物
    """

    # 定义大型猫科动物的属
    large_cat_genera = {
        'panthera': '豹属',
        'puma': '美洲狮属',
        'acinonyx': '猎豹属',
        'lynx': '猞猁属',
        'neofelis': '云豹属'
    }

    # 中型猫科动物
    medium_cat_genera = {
        'leopardus': '虎猫属',
        'leptailurus': '薮猫属',
        'caracal': '狞猫属',
        'catopuma': '金猫属',
        'pardofelis': '纹猫属',
        'prionailurus': '豹猫属',
        'otocolobus': '兔狲属'
    }

    # Panthera属的物种映射
    panthera_species_map = {
        'lion': '狮',
        'tiger': '虎',
        'leopard': '豹',
        'jaguar': '美洲豹',
        'snow leopard': '雪豹',
        'amur leopard': '远东豹',
        'panther': '黑豹',
    }

    # 根据参数决定要分析的猫科动物
    if include_medium_cats:
        target_genera = {**large_cat_genera, **medium_cat_genera}
        cat_type = "大中型猫科动物"
    else:
        target_genera = large_cat_genera
        cat_type = "大型猫科动物"

    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 统计变量初始化
    stats = {
        'cat_type': cat_type,
        'total_videos': 0,
        'total_duration': 0,  # 总视频时长
        'total_segment_duration': 0,  # 总片段时长
        'by_genus': defaultdict(lambda: {
            'count': 0,
            'total_duration': 0,
            'segment_duration': 0,
            'duration_list': [],
            'segment_list': [],
            'duration_distribution': {},
            'segment_distribution': {}
        }),
        'by_species': defaultdict(lambda: {
            'count': 0,
            'total_duration': 0,
            'segment_duration': 0,
            'duration_list': [],
            'segment_list': [],
            'duration_distribution': {},
            'segment_distribution': {}
        }),
        'by_behavior': defaultdict(lambda: {
            'count': 0,
            'total_duration': 0,
            'segment_duration': 0,
            'duration_list': [],
            'segment_list': [],
            'duration_distribution': {},
            'segment_distribution': {}
        }),
        'species_behavior': defaultdict(lambda: defaultdict(lambda: {
            'count': 0,
            'total_duration': 0,
            'segment_duration': 0,
            'duration_list': [],
            'segment_list': [],
            'avg_segment': 0
        })),
        'behavior_species': defaultdict(lambda: defaultdict(lambda: {
            'count': 0,
            'segment_duration': 0
        })),
        'panthera_species': defaultdict(lambda: {
            'count': 0,
            'total_duration': 0,
            'segment_duration': 0,
            'duration_list': [],
            'segment_list': [],
            'duration_distribution': {},
            'segment_distribution': {}
        }),
        'panthera_species_behavior': defaultdict(lambda: defaultdict(lambda: {
            'count': 0,
            'segment_duration': 0,
            'segment_list': []
        })),
        'all_video_durations': [],  # 所有视频时长
        'all_segment_durations': [],  # 所有片段时长
        'overall_video_distribution': {},  # 总体视频时长分布
        'overall_segment_distribution': {},  # 总体片段时长分布
        'video_details': [],  # 详细视频信息
        'resolution_categories': defaultdict(int),
        'fps_stats': defaultdict(int),
        'by_subset': defaultdict(int)
    }

    # 遍历所有视频
    for video_id, video_info in data.items():
        # 检查是否为目标猫科动物
        taxonomy = video_info.get('taxnomy', [])
        if not taxonomy:
            continue

        genus = taxonomy[0].get('genus', '')
        keyword = taxonomy[0].get('keyword', '').lower()

        if genus in target_genera:
            stats['total_videos'] += 1

            # 获取视频元数据
            video_duration = video_info.get('duration', 0)
            resolution = video_info.get('resolution', '')
            fps = video_info.get('fps', 0)
            subset = video_info.get('subset', '')

            # 统计总时长
            stats['total_duration'] += video_duration
            stats['all_video_durations'].append(video_duration)

            # 物种识别
            species_key = f"{genus}_{keyword}" if keyword else f"{genus}_unknown"

            # 更新属统计
            stats['by_genus'][genus]['count'] += 1
            stats['by_genus'][genus]['total_duration'] += video_duration
            stats['by_genus'][genus]['duration_list'].append(video_duration)

            # 更新物种统计
            stats['by_species'][species_key]['count'] += 1
            stats['by_species'][species_key]['total_duration'] += video_duration
            stats['by_species'][species_key]['duration_list'].append(video_duration)

            # 特别处理panthera属
            if genus == 'panthera':
                # 识别具体物种
                identified_species = None
                for key, name in panthera_species_map.items():
                    if key in keyword:
                        identified_species = name
                        break

                if not identified_species:
                    identified_species = keyword if keyword else '未识别'

                stats['panthera_species'][identified_species]['count'] += 1
                stats['panthera_species'][identified_species]['total_duration'] += video_duration
                stats['panthera_species'][identified_species]['duration_list'].append(video_duration)

            # 统计行为及其时间片段
            annotations = video_info.get('annotations', [])
            for ann in annotations:
                behavior = ann.get('label', '')
                segment = ann.get('segment', [])

                if len(segment) == 2:
                    segment_length = segment[1] - segment[0]

                    # 统计总片段时长
                    stats['total_segment_duration'] += segment_length
                    stats['all_segment_durations'].append(segment_length)

                    # 更新行为统计
                    stats['by_behavior'][behavior]['count'] += 1
                    stats['by_behavior'][behavior]['total_duration'] += video_duration
                    stats['by_behavior'][behavior]['segment_duration'] += segment_length
                    stats['by_behavior'][behavior]['duration_list'].append(video_duration)
                    stats['by_behavior'][behavior]['segment_list'].append(segment_length)

                    # 更新属的片段统计
                    stats['by_genus'][genus]['segment_duration'] += segment_length
                    stats['by_genus'][genus]['segment_list'].append(segment_length)

                    # 更新物种的片段统计
                    stats['by_species'][species_key]['segment_duration'] += segment_length
                    stats['by_species'][species_key]['segment_list'].append(segment_length)

                    # 更新物种-行为统计
                    stats['species_behavior'][species_key][behavior]['count'] += 1
                    stats['species_behavior'][species_key][behavior]['total_duration'] += video_duration
                    stats['species_behavior'][species_key][behavior]['segment_duration'] += segment_length
                    stats['species_behavior'][species_key][behavior]['duration_list'].append(video_duration)
                    stats['species_behavior'][species_key][behavior]['segment_list'].append(segment_length)

                    # 更新行为-物种统计
                    stats['behavior_species'][behavior][species_key]['count'] += 1
                    stats['behavior_species'][behavior][species_key]['segment_duration'] += segment_length

                    # Panthera物种的行为统计
                    if genus == 'panthera' and identified_species:
                        stats['panthera_species'][identified_species]['segment_duration'] += segment_length
                        stats['panthera_species'][identified_species]['segment_list'].append(segment_length)

                        stats['panthera_species_behavior'][identified_species][behavior]['count'] += 1
                        stats['panthera_species_behavior'][identified_species][behavior][
                            'segment_duration'] += segment_length
                        stats['panthera_species_behavior'][identified_species][behavior]['segment_list'].append(
                            segment_length)

                    # 保存详细信息
                    stats['video_details'].append({
                        'video_id': video_id,
                        'genus': genus,
                        'species_key': species_key,
                        'keyword': keyword,
                        'behavior': behavior,
                        'video_duration': video_duration,
                        'segment_start': segment[0],
                        'segment_end': segment[1],
                        'segment_length': segment_length,
                        'resolution': resolution,
                        'fps': fps,
                        'subset': subset,
                        'url': video_info.get('url', '')
                    })

            # 统计其他信息
            stats['by_subset'][subset] += 1
            stats['resolution_categories'][categorize_resolution(resolution)] += 1
            stats['fps_stats'][fps] += 1

    # 计算所有分布
    # 总体分布
    if stats['all_video_durations']:
        stats['overall_video_distribution'] = calculate_duration_distribution(stats['all_video_durations'])
    if stats['all_segment_durations']:
        stats['overall_segment_distribution'] = calculate_duration_distribution(stats['all_segment_durations'])

    # 属级别分布
    for genus, data in stats['by_genus'].items():
        if data['duration_list']:
            data['duration_distribution'] = calculate_duration_distribution(data['duration_list'])
        if data['segment_list']:
            data['segment_distribution'] = calculate_duration_distribution(data['segment_list'])
            data['avg_segment'] = statistics.mean(data['segment_list'])

    # 物种级别分布
    for species, data in stats['by_species'].items():
        if data['duration_list']:
            data['duration_distribution'] = calculate_duration_distribution(data['duration_list'])
        if data['segment_list']:
            data['segment_distribution'] = calculate_duration_distribution(data['segment_list'])
            data['avg_segment'] = statistics.mean(data['segment_list'])

    # 行为级别分布
    for behavior, data in stats['by_behavior'].items():
        if data['duration_list']:
            data['duration_distribution'] = calculate_duration_distribution(data['duration_list'])
        if data['segment_list']:
            data['segment_distribution'] = calculate_duration_distribution(data['segment_list'])
            data['avg_segment'] = statistics.mean(data['segment_list'])

    # Panthera物种分布
    for species, data in stats['panthera_species'].items():
        if data['duration_list']:
            data['duration_distribution'] = calculate_duration_distribution(data['duration_list'])
        if data['segment_list']:
            data['segment_distribution'] = calculate_duration_distribution(data['segment_list'])
            data['avg_segment'] = statistics.mean(data['segment_list']) if data['segment_list'] else 0

    # 物种-行为的平均片段时长
    for species in stats['species_behavior']:
        for behavior in stats['species_behavior'][species]:
            data = stats['species_behavior'][species][behavior]
            if data['segment_list']:
                data['avg_segment'] = statistics.mean(data['segment_list'])
                data['segment_distribution'] = calculate_duration_distribution(data['segment_list'])

    return stats, target_genera


def categorize_resolution(resolution: str) -> str:
    """将分辨率分类为质量等级"""
    if not resolution:
        return 'unknown'

    try:
        width, height = map(int, resolution.split('x'))
        pixels = width * height

        if pixels >= 1920 * 1080:
            return 'HD_1080p+'
        elif pixels >= 1280 * 720:
            return 'HD_720p'
        elif pixels >= 854 * 480:
            return 'SD_480p'
        else:
            return 'Low_360p-'
    except:
        return 'unknown'


def print_time_analysis(stats: Dict):
    """打印详细的时间分析"""

    print("\n" + "=" * 100)
    print("⏱️ 时间维度详细分析")
    print("=" * 100)

    # 1. 总体时间统计
    print("\n📊 数据集总体时间统计：")
    print("-" * 60)
    print(f"总视频数: {stats['total_videos']} 个")
    print(f"视频总时长: {format_duration(stats['total_duration'])} ({stats['total_duration']:.2f}秒)")
    print(f"片段总时长: {format_duration(stats['total_segment_duration'])} ({stats['total_segment_duration']:.2f}秒)")

    if stats['total_videos'] > 0:
        print(f"平均视频时长: {stats['total_duration'] / stats['total_videos']:.2f}秒")
        print(f"平均片段时长: {stats['total_segment_duration'] / stats['total_videos']:.2f}秒")

    # 2. 时长分布对比
    print("\n📈 时长分布对比：")
    print("-" * 80)
    print("范围        | 视频分布                              | 片段分布")
    print("-" * 80)

    for range_key in ['<3s', '3-5s', '5-10s', '10-20s', '20-60s', '>60s']:
        video_count = stats['overall_video_distribution'].get(range_key, 0)
        segment_count = stats['overall_segment_distribution'].get(range_key, 0)

        video_pct = (video_count / len(stats['all_video_durations']) * 100) if stats['all_video_durations'] else 0
        segment_pct = (segment_count / len(stats['all_segment_durations']) * 100) if stats[
            'all_segment_durations'] else 0

        video_bar = '█' * int(video_pct / 2)
        segment_bar = '█' * int(segment_pct / 2)

        print(
            f"{range_key:8} | {video_count:4}个 ({video_pct:5.1f}%) {video_bar:20} | {segment_count:4}个 ({segment_pct:5.1f}%) {segment_bar}")

    # 3. 属级别时长统计
    print("\n\n🐾 各属时长统计：")
    print("-" * 100)
    print(f"{'属名':<10} {'视频数':>8} {'视频总时长':>12} {'片段总时长':>12} {'平均片段':>10} {'片段分布'}")
    print("-" * 100)

    sorted_genera = sorted(stats['by_genus'].items(),
                           key=lambda x: x[1]['count'],
                           reverse=True)

    for genus, info in sorted_genera:
        if info['count'] > 0:
            avg_segment = info['avg_segment'] if 'avg_segment' in info else 0
            segment_dist = format_distribution_string(info['segment_distribution'], len(info['segment_list']))

            print(f"{genus:<10} {info['count']:>8} "
                  f"{format_duration(info['total_duration']):>12} "
                  f"{format_duration(info['segment_duration']):>12} "
                  f"{avg_segment:>9.2f}s")
            print(f"{'':10} 片段分布: {segment_dist}")

    # 4. Panthera属物种时长统计
    print("\n\n🦁 Panthera属物种时长详细统计：")
    print("-" * 100)

    sorted_panthera = sorted(stats['panthera_species'].items(),
                             key=lambda x: x[1]['count'],
                             reverse=True)

    for species, info in sorted_panthera:
        if info['count'] > 0:
            print(f"\n{species} ({info['count']}个视频)")
            print(f"  • 视频总时长: {format_duration(info['total_duration'])}")
            print(f"  • 片段总时长: {format_duration(info['segment_duration'])}")

            if info.get('avg_segment'):
                print(f"  • 平均片段时长: {info['avg_segment']:.2f}秒")

            if info.get('segment_distribution'):
                dist_str = format_distribution_string(info['segment_distribution'], len(info['segment_list']))
                print(f"  • 片段分布: {dist_str}")

            # 该物种的行为时长
            behaviors = stats['panthera_species_behavior'].get(species, {})
            if behaviors:
                print(f"  • 行为时长统计:")
                sorted_behaviors = sorted(behaviors.items(),
                                          key=lambda x: x[1]['count'],
                                          reverse=True)[:5]

                for behavior, b_info in sorted_behaviors:
                    avg_seg = statistics.mean(b_info['segment_list']) if b_info['segment_list'] else 0
                    print(f"    - {behavior}: {b_info['count']}次, 平均片段{avg_seg:.2f}秒")

    # 5. 行为时长统计
    print("\n\n🎯 行为类型时长统计 (TOP 10)：")
    print("-" * 100)
    print(f"{'行为':<35} {'次数':>6} {'片段总时长':>12} {'平均片段':>10}")
    print("-" * 100)

    sorted_behaviors = sorted(stats['by_behavior'].items(),
                              key=lambda x: x[1]['count'],
                              reverse=True)[:10]

    for behavior, info in sorted_behaviors:
        if info['count'] > 0:
            avg_segment = info.get('avg_segment', 0)
            print(f"{behavior:<35} {info['count']:>6} "
                  f"{format_duration(info['segment_duration']):>12} "
                  f"{avg_segment:>9.2f}s")

            # 显示该行为的片段分布
            if info.get('segment_distribution'):
                dist_str = format_distribution_string(info['segment_distribution'], len(info['segment_list']))
                print(f"{'':35} 分布: {dist_str}")


def generate_enhanced_report_with_time(stats: Dict, target_genera: Dict, output_file: str):
    """生成包含详细时间分析的增强版报告"""

    with open(output_file, 'w', encoding='utf-8') as f:
        # 报告标题
        f.write("=" * 100 + "\n")
        f.write(f"MammalNet {stats['cat_type']}综合统计报告（增强版）\n")
        f.write("=" * 100 + "\n\n")

        # 1. 执行摘要
        f.write("【执行摘要】\n")
        f.write("-" * 50 + "\n")
        f.write(f"• 数据集规模: {stats['total_videos']} 个视频\n")
        f.write(f"• 涵盖动物属: {len(stats['by_genus'])} 个\n")
        f.write(f"• 识别物种数: {len(stats['by_species'])} 个\n")
        f.write(f"• 行为类型: {len(stats['by_behavior'])} 种\n")
        f.write(f"• 视频总时长: {format_duration(stats['total_duration'])} ({stats['total_duration']:.2f}秒)\n")
        f.write(
            f"• 片段总时长: {format_duration(stats['total_segment_duration'])} ({stats['total_segment_duration']:.2f}秒)\n")

        if stats['total_videos'] > 0:
            f.write(f"• 平均视频时长: {stats['total_duration'] / stats['total_videos']:.2f}秒\n")
            f.write(f"• 平均片段时长: {stats['total_segment_duration'] / stats['total_videos']:.2f}秒\n")
        f.write("\n")

        # 2. 总体时长分布
        f.write("【总体时长分布分析】\n")
        f.write("-" * 50 + "\n")

        f.write("视频时长分布:\n")
        for range_key in ['<3s', '3-5s', '5-10s', '10-20s', '20-60s', '>60s']:
            count = stats['overall_video_distribution'].get(range_key, 0)
            if len(stats['all_video_durations']) > 0:
                pct = (count / len(stats['all_video_durations'])) * 100
                f.write(f"  • {range_key}: {count}个视频 ({pct:.1f}%)\n")

        f.write("\n片段时长分布:\n")
        for range_key in ['<3s', '3-5s', '5-10s', '10-20s', '20-60s', '>60s']:
            count = stats['overall_segment_distribution'].get(range_key, 0)
            if len(stats['all_segment_durations']) > 0:
                pct = (count / len(stats['all_segment_durations'])) * 100
                f.write(f"  • {range_key}: {count}个片段 ({pct:.1f}%)\n")
        f.write("\n")

        # 3. 动物分布统计（含时长）
        f.write("【动物分布统计】\n")
        f.write("-" * 50 + "\n\n")

        f.write("属级别统计（含完整行为列表）：\n")
        f.write(f"{'属名':<15} {'中文名':<25} {'视频数':>8} {'占比':>8}\n")
        f.write("-" * 100 + "\n")

        sorted_genera = sorted(stats['by_genus'].items(),
                               key=lambda x: x[1]['count'],
                               reverse=True)

        for genus, info in sorted_genera:
            genus_name = target_genera.get(genus, genus)
            percentage = (info['count'] / stats['total_videos'] * 100) if stats['total_videos'] > 0 else 0

            f.write(f"{genus:<15} {genus_name:<25} {info['count']:>8} {percentage:>7.1f}%\n")
            f.write(f"  • 视频总时长: {format_duration(info['total_duration'])}\n")
            f.write(f"  • 片段总时长: {format_duration(info['segment_duration'])}\n")

            if info.get('avg_segment'):
                f.write(f"  • 平均片段时长: {info['avg_segment']:.2f}秒\n")

            if info.get('segment_distribution'):
                dist_str = format_distribution_string(info['segment_distribution'], len(info['segment_list']))
                f.write(f"  • 片段分布: {dist_str}\n")

            # 列出该属的行为统计（含时间分布）
            behaviors = defaultdict(lambda: {'count': 0, 'duration': 0.0, 'segments': []})
            for species_key in stats['species_behavior']:
                if species_key.startswith(genus + "_"):
                    for behavior, b_info in stats['species_behavior'][species_key].items():
                        behaviors[behavior]['count'] += b_info['count']
                        behaviors[behavior]['duration'] += b_info['segment_duration']
                        # 累积该行为的片段时长列表，用于做分布
                        behaviors[behavior]['segments'].extend(b_info.get('segment_list', []))

            if behaviors:
                f.write("  行为统计（含时间分布）：\n")
                sorted_behaviors = sorted(behaviors.items(), key=lambda x: x[1]['count'], reverse=True)
                total_b = sum(v['count'] for v in behaviors.values())

                for behavior, b_data in sorted_behaviors:
                    avg_time = (b_data['duration'] / b_data['count']) if b_data['count'] > 0 else 0
                    pct = (b_data['count'] / total_b * 100) if total_b > 0 else 0
                    f.write(f"    • {behavior:<35} {b_data['count']:>4} 次 ({pct:5.1f}%), 平均片段{avg_time:.1f}秒\n")

                    # 该行为的片段时长分布
                    dist = calculate_duration_distribution(b_data['segments'])
                    dist_str = format_distribution_string(dist, len(b_data['segments']))
                    f.write(f"      分布: {dist_str}\n")
                f.write("\n")

        # 4. Panthera属物种详细分析（含时长）
        f.write("\n【Panthera属（豹属）物种详细分析】\n")
        f.write("-" * 50 + "\n")

        total_panthera = sum(info['count'] for info in stats['panthera_species'].values())
        f.write(f"Panthera属总计: {total_panthera} 个视频\n\n")

        sorted_panthera = sorted(stats['panthera_species'].items(),
                                 key=lambda x: x[1]['count'],
                                 reverse=True)

        for species, info in sorted_panthera:
            if info['count'] > 0:
                percentage = (info['count'] / total_panthera * 100) if total_panthera > 0 else 0

                f.write(f"{species:<15} {info['count']:>6} 个视频 ({percentage:>5.1f}%)\n")
                f.write(f"  • 视频总时长: {format_duration(info['total_duration'])}\n")
                f.write(f"  • 片段总时长: {format_duration(info['segment_duration'])}\n")

                if info.get('avg_segment'):
                    f.write(f"  • 平均片段时长: {info['avg_segment']:.2f}秒\n")

                if info.get('segment_distribution'):
                    dist_str = format_distribution_string(info['segment_distribution'], len(info['segment_list']))
                    f.write(f"  • 片段分布: {dist_str}\n")

                # 列出该物种的行为分布
                behaviors = stats['panthera_species_behavior'].get(species, {})
                if behaviors:
                    f.write("  行为分布：\n")
                    sorted_behaviors = sorted(behaviors.items(),
                                              key=lambda x: x[1]['count'],
                                              reverse=True)

                    for behavior, b_info in sorted_behaviors:
                        b_percentage = (b_info['count'] / info['count'] * 100)
                        avg_seg = statistics.mean(b_info['segment_list']) if b_info['segment_list'] else 0
                        f.write(
                            f"    • {behavior:<35} {b_info['count']:>3} 次 ({b_percentage:>5.1f}%), 平均片段{avg_seg:.1f}秒\n")

                        # 显示该行为的片段分布
                        if b_info.get('segment_list'):
                            dist = calculate_duration_distribution(b_info['segment_list'])
                            dist_str = format_distribution_string(dist, len(b_info['segment_list']))
                            f.write(f"      分布: {dist_str}\n")
                f.write("\n")

        # 5. 行为类型统计与物种分析（含时长）
        f.write("\n【行为类型统计与物种分析】\n")
        f.write("-" * 80 + "\n")

        sorted_behaviors = sorted(stats['by_behavior'].items(),
                                  key=lambda x: x[1]['count'],
                                  reverse=True)

        for idx, (behavior, info) in enumerate(sorted_behaviors):
            percentage = (info['count'] / stats['total_videos'] * 100) if stats['total_videos'] > 0 else 0

            f.write("\n" + "=" * 120 + "\n")
            f.write(f"◆ {behavior:<40} 总计: {info['count']} 次 ({percentage:.1f}%)\n")
            f.write("=" * 120 + "\n")

            f.write(f"  • 片段总时长: {format_duration(info['segment_duration'])}\n")
            f.write(f"  • 平均片段时长: {info.get('avg_segment', 0):.2f}秒\n")

            if info.get('segment_distribution'):
                dist_str = format_distribution_string(info['segment_distribution'], len(info['segment_list']))
                f.write(f"  • 片段分布: {dist_str}\n")

            # 统计该行为在各属中的分布
            genus_dist = defaultdict(lambda: {'count': 0, 'duration': 0})
            species_dist = stats['behavior_species'][behavior]

            for species_key, s_info in species_dist.items():
                genus = species_key.split('_')[0]
                genus_dist[genus]['count'] += s_info['count']
                genus_dist[genus]['duration'] += s_info['segment_duration']

            # 显示属级别分布
            f.write("\n  ▶ 属分布：\n")
            for genus, g_data in sorted(genus_dist.items(),
                                        key=lambda x: x[1]['count'],
                                        reverse=True):
                g_percentage = (g_data['count'] / info['count'] * 100) if info['count'] > 0 else 0
                genus_name = target_genera.get(genus, genus)
                avg_duration = g_data['duration'] / g_data['count'] if g_data['count'] > 0 else 0
                f.write(
                    f"    • {genus:<10} ({genus_name:<15}): {g_data['count']:>4} 次 ({g_percentage:>5.1f}%), 平均{avg_duration:>6.1f}秒\n")

                # 如果是panthera，显示物种细分
                if genus == 'panthera':
                    panthera_species_for_behavior = {}
                    for species_key, s_info in species_dist.items():
                        if species_key.startswith('panthera_'):
                            keyword = species_key.replace('panthera_', '')
                            # 映射到中文名
                            for eng, chn in {'lion': '狮', 'tiger': '虎', 'leopard': '豹',
                                             'jaguar': '美洲豹', 'snow leopard': '雪豹',
                                             'amur leopard': '远东豹'}.items():
                                if eng in keyword:
                                    keyword = chn
                                    break
                            panthera_species_for_behavior[keyword] = s_info

                    if panthera_species_for_behavior:
                        for sp, sp_info in sorted(panthera_species_for_behavior.items(),
                                                  key=lambda x: x[1]['count'],
                                                  reverse=True):
                            sp_percentage = (sp_info['count'] / g_data['count'] * 100) if g_data['count'] > 0 else 0
                            sp_avg = sp_info['segment_duration'] / sp_info['count'] if sp_info['count'] > 0 else 0
                            f.write(
                                f"      - {sp:<10}: {sp_info['count']:>3} 次 ({sp_percentage:>5.1f}%), 平均{sp_avg:>6.1f}秒\n")

            # 只显示前10个行为的详细信息
            if idx >= 9:
                break

        # 5. 数据质量分析
        f.write("\n\n【数据质量分析】\n")
        f.write("-" * 80 + "\n")
        f.write("分辨率质量分布：\n")
        for quality in ['HD_1080p+', 'HD_720p', 'SD_480p', 'Low_360p-', 'unknown']:
            if quality in stats['resolution_categories']:
                count = stats['resolution_categories'][quality]
                percentage = (count / stats['total_videos'] * 100) if stats['total_videos'] > 0 else 0
                f.write(f"  • {quality:12s}: {count:5d} ({percentage:5.1f}%)\n")

        # 6. 关键发现
        f.write("\n【关键发现】\n")
        f.write("-" * 80 + "\n")

        # Panthera属占比
        panthera_info = stats['by_genus'].get('panthera', {})
        if panthera_info and stats['total_videos'] > 0:
            panthera_count = panthera_info['count']
            f.write(f"• Panthera属占比: {panthera_count}/{stats['total_videos']} ")
            f.write(f"({panthera_count / stats['total_videos'] * 100:.1f}%)\n")

        # 最常见的Panthera物种
        if stats['panthera_species']:
            top_panthera = max(stats['panthera_species'].items(),
                               key=lambda x: x[1]['count'])
            f.write(f"• Panthera属最常见物种: {top_panthera[0]} ({top_panthera[1]['count']} 个视频)\n")

        # 最常见的行为
        if stats['by_behavior']:
            top_behavior = max(stats['by_behavior'].items(),
                               key=lambda x: x[1]['count'])
            f.write(f"• 最常见的行为: {top_behavior[0]} ({top_behavior[1]['count']} 次)\n")

        # 平均片段时长最长的行为
        longest_segment_behavior = None
        longest_avg = 0
        for behavior, info in stats['by_behavior'].items():
            if info.get('avg_segment', 0) > longest_avg:
                longest_avg = info['avg_segment']
                longest_segment_behavior = behavior

        if longest_segment_behavior:
            f.write(f"• 平均片段最长的行为: {longest_segment_behavior} ({longest_avg:.2f}秒)\n")

        # 平均片段时长最短的行为
        shortest_segment_behavior = None
        shortest_avg = float('inf')
        for behavior, info in stats['by_behavior'].items():
            avg = info.get('avg_segment', float('inf'))
            if avg > 0 and avg < shortest_avg:
                shortest_avg = avg
                shortest_segment_behavior = behavior

        if shortest_segment_behavior and shortest_avg != float('inf'):
            f.write(f"• 平均片段最短的行为: {shortest_segment_behavior} ({shortest_avg:.2f}秒)\n")

        # 物种多样性
        f.write(f"• 识别的物种/变体总数: {len(stats['by_species'])} 个\n")

        # 时长分布特点
        if stats['overall_segment_distribution']:
            max_range = max(stats['overall_segment_distribution'].items(), key=lambda x: x[1])
            f.write(f"• 片段时长主要分布: {max_range[0]}范围\n")

        # 报告生成信息
        f.write("\n" + "=" * 120 + "\n")
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 120 + "\n")

    print(f"\n📊 增强版报告已生成: {output_file}")


def export_detailed_csv(stats: Dict, output_file: str):
    """导出详细的CSV文件，包含时间信息"""

    df = pd.DataFrame(stats['video_details'])

    if not df.empty:
        # 添加额外的计算列
        df['segment_category'] = df['segment_length'].apply(lambda x:
                                                            '<3s' if x < 3 else
                                                            '3-5s' if x < 5 else
                                                            '5-10s' if x < 10 else
                                                            '10-20s' if x < 20 else
                                                            '20-60s' if x < 60 else
                                                            '>60s')

        df['video_category'] = df['video_duration'].apply(lambda x:
                                                          '<3s' if x < 3 else
                                                          '3-5s' if x < 5 else
                                                          '5-10s' if x < 10 else
                                                          '10-20s' if x < 20 else
                                                          '20-60s' if x < 60 else
                                                          '>60s')

        # 按物种和行为排序
        df = df.sort_values(['species_key', 'behavior', 'segment_length'])

        # 导出CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✅ 详细数据已导出: {output_file}")


def export_time_statistics_json(stats: Dict, output_file: str):
    """导出时间统计数据到JSON"""

    time_stats = {
        'dataset_summary': {
            'total_videos': stats['total_videos'],
            'total_video_duration': stats['total_duration'],
            'total_segment_duration': stats['total_segment_duration'],
            'avg_video_duration': stats['total_duration'] / stats['total_videos'] if stats['total_videos'] > 0 else 0,
            'avg_segment_duration': stats['total_segment_duration'] / stats['total_videos'] if stats[
                                                                                                   'total_videos'] > 0 else 0,
            'video_distribution': stats['overall_video_distribution'],
            'segment_distribution': stats['overall_segment_distribution']
        },
        'genus_stats': {},
        'species_stats': {},
        'behavior_stats': {},
        'panthera_species_stats': {}
    }

    # 属级别统计
    for genus, info in stats['by_genus'].items():
        time_stats['genus_stats'][genus] = {
            'count': info['count'],
            'total_duration': info['total_duration'],
            'segment_duration': info['segment_duration'],
            'avg_segment': info.get('avg_segment', 0),
            'segment_distribution': info.get('segment_distribution', {})
        }

    # 物种级别统计
    for species, info in stats['by_species'].items():
        time_stats['species_stats'][species] = {
            'count': info['count'],
            'total_duration': info['total_duration'],
            'segment_duration': info['segment_duration'],
            'avg_segment': info.get('avg_segment', 0),
            'segment_distribution': info.get('segment_distribution', {})
        }

    # 行为统计
    for behavior, info in stats['by_behavior'].items():
        time_stats['behavior_stats'][behavior] = {
            'count': info['count'],
            'total_duration': info['total_duration'],
            'segment_duration': info['segment_duration'],
            'avg_segment': info.get('avg_segment', 0),
            'segment_distribution': info.get('segment_distribution', {})
        }

    # Panthera物种统计
    for species, info in stats['panthera_species'].items():
        time_stats['panthera_species_stats'][species] = {
            'count': info['count'],
            'total_duration': info['total_duration'],
            'segment_duration': info['segment_duration'],
            'avg_segment': info.get('avg_segment', 0),
            'segment_distribution': info.get('segment_distribution', {})
        }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(time_stats, f, ensure_ascii=False, indent=2)

    print(f"✅ 时间统计JSON已导出: {output_file}")


# 主程序
if __name__ == "__main__":
    # 文件路径
    json_file = r"H:\big_cat_dataset\MammalNet(videos)\annotation\annotation-\annotation\detection_annotations.json"

    try:
        print("\n" + "🐯" * 30)
        print("MammalNet大型猫科动物时长详细分析")
        print("🐯" * 30)

        # 执行分析
        stats, genera = analyze_big_cats_with_time(json_file, include_medium_cats=False)

        # 打印时间分析
        print_time_analysis(stats)

        # 生成报告文件
        print("\n" + "=" * 80)
        print("📝 正在生成报告文件...")
        print("=" * 80)

        # 生成增强版报告
        generate_enhanced_report_with_time(stats, genera, 'mammalnet_time_report.txt')

        # 导出详细CSV
        export_detailed_csv(stats, 'mammalnet_video_details.csv')

        # 导出时间统计JSON
        export_time_statistics_json(stats, 'mammalnet_time_statistics.json')

        print("\n" + "=" * 80)
        print("✅ 分析完成！")
        print("=" * 80)
        print("\n📚 生成的文件列表：")
        print("📄 mammalnet_time_report.txt      - 包含时长分析的综合报告")
        print("📊 mammalnet_video_details.csv    - 所有视频片段的详细信息")
        print("📈 mammalnet_time_statistics.json - 时间统计数据JSON")
        print("=" * 80)

        # 打印关键统计摘要
        print("\n🎯 关键统计摘要：")
        print(f"• 总视频数: {stats['total_videos']} 个")
        print(f"• 视频总时长: {format_duration(stats['total_duration'])}")
        print(f"• 片段总时长: {format_duration(stats['total_segment_duration'])}")

        if stats['total_videos'] > 0:
            print(f"• 平均视频时长: {stats['total_duration'] / stats['total_videos']:.2f}秒")
            print(f"• 平均片段时长: {stats['total_segment_duration'] / stats['total_videos']:.2f}秒")

        # 找出片段最长和最短的行为
        if stats['by_behavior']:
            longest = max(stats['by_behavior'].items(),
                          key=lambda x: x[1].get('avg_segment', 0))
            shortest = min(stats['by_behavior'].items(),
                           key=lambda x: x[1].get('avg_segment', float('inf')) if x[1].get('avg_segment',
                                                                                           0) > 0 else float('inf'))

            if longest[1].get('avg_segment'):
                print(f"• 平均片段最长的行为: {longest[0]} ({longest[1]['avg_segment']:.2f}秒)")
            if shortest[1].get('avg_segment'):
                print(f"• 平均片段最短的行为: {shortest[0]} ({shortest[1]['avg_segment']:.2f}秒)")

        print("=" * 80)

    except FileNotFoundError:
        print(f"错误: 找不到文件 {json_file}")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback

        traceback.print_exc()