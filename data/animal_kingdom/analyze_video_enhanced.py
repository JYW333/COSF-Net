import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import statistics
from datetime import datetime, timedelta
import json
import os
import re


def parse_duration_string(duration_str):
    """
    解析时长字符串，如 "GZTAMICI: 4.02s; PJCNAICI: 14.00s"
    返回视频ID和时长的字典
    """
    durations = {}
    if pd.isna(duration_str):
        return durations

    # 分割多个视频
    parts = duration_str.split('; ')
    for part in parts:
        if ':' in part:
            video_id, time = part.split(': ')
            # 移除's'并转换为浮点数
            time_value = float(time.replace('s', ''))
            durations[video_id] = time_value

    return durations


def parse_total_duration(duration_str):
    """
    解析总时长字符串，如 "61.45s"
    返回秒数
    """
    if pd.isna(duration_str):
        return 0
    return float(duration_str.replace('s', ''))


def format_duration(seconds):
    """
    格式化时长显示
    """
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}小时"


def calculate_duration_distribution(durations):
    """
    计算时长分布
    返回分布字典和总数
    """
    distribution = {
        '<3s': 0,
        '3-5s': 0,
        '5-10s': 0,
        '10-20s': 0,
        '>20s': 0
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
        else:
            distribution['>20s'] += 1

    return distribution


def analyze_video_dataset_with_detailed_time(xlsx_file: str, csv_file: str):
    """
    分析视频数据集的增强版分析工具（包含详细时间维度）

    Args:
        xlsx_file: Excel文件路径
        csv_file: CSV文件路径
    """

    print("\n" + "=" * 100)
    print("🎬 视频数据集增强版分析（含详细时长统计）")
    print("=" * 100)

    # 读取数据
    csv_data = pd.read_csv(csv_file)
    xlsx_data = pd.ExcelFile(xlsx_file)

    # 初始化统计变量
    stats = {
        'total_videos': 0,
        'total_duration': 0,  # 总时长（秒）
        'by_species': {},
        'by_action': defaultdict(lambda: {
            'count': 0,
            'duration': 0,
            'videos': [],
            'duration_list': []  # 存储所有时长用于分布计算
        }),
        'by_category': defaultdict(lambda: {'count': 0, 'duration': 0}),
        'species_actions': defaultdict(lambda: defaultdict(lambda: {
            'count': 0,
            'duration': 0,
            'duration_list': []  # 每个物种-动作的时长列表
        })),
        'action_species': defaultdict(lambda: defaultdict(lambda: {'count': 0, 'duration': 0})),
        'category_species': defaultdict(lambda: defaultdict(lambda: {'count': 0, 'duration': 0})),
        'species_categories': defaultdict(lambda: defaultdict(lambda: {'count': 0, 'duration': 0})),
        'video_details': [],  # 存储每个视频的详细信息
        'duration_distribution': calculate_duration_distribution([]),  # 总体时长分布
        'all_durations': []  # 存储所有视频时长
    }

    # 1. 分析CSV数据（总体统计）
    print("\n📊 处理总体统计数据...")
    species_mapping = {
        'Lion (狮子)': '狮',
        'Tiger (老虎)': '虎',
        'Leopard (豹)': '豹',
        'Cheetah (猎豹）': '猎豹',
        'Jaguarundi Cat (美洲豹猫）': '美洲豹猫',
        'Cougar、Puma (美洲狮)': '美洲狮'
    }

    # 2. 读取详细视频数据
    print("📋 处理详细视频时长数据...")

    # 读取All_Video_IDs_Summary - 这里有时长信息
    df_video_ids = pd.read_excel(xlsx_file, sheet_name='All_Video_IDs_Summary')
    df_video_ids = df_video_ids.dropna(subset=['Action'])

    # 初始化每个物种的统计
    for sheet_name in species_mapping.keys():
        species_cn = species_mapping[sheet_name]
        stats['by_species'][species_cn] = {
            'total_videos': 0,
            'total_duration': 0,
            'avg_duration': 0,
            'min_duration': float('inf'),
            'max_duration': 0,
            'actions': {},
            'video_list': [],
            'duration_list': [],  # 该物种所有视频的时长列表
            'duration_distribution': {},  # 该物种的时长分布
            'original_name': sheet_name
        }

    # 3. 处理视频时长数据
    print("⏱️ 分析视频时长分布...")

    processed_combinations = set()  # 避免重复计数

    for _, row in df_video_ids.iterrows():
        if pd.notna(row['Sheet']) and pd.notna(row['Action']):
            sheet_name = row['Sheet'].strip()
            action = row['Action'].strip()

            # 映射到标准物种名称
            if sheet_name in species_mapping:
                species = species_mapping[sheet_name]
            else:
                continue

            # 创建唯一键
            key = f"{sheet_name}_{action}_{row.get('Type', '')}"

            # 只处理包含Duration_Total的汇总行
            if pd.notna(row.get('Duration_Total')) and key not in processed_combinations:
                processed_combinations.add(key)

                # 解析视频时长
                video_durations = parse_duration_string(row.get('Video_Durations', ''))
                total_duration = parse_total_duration(row['Duration_Total'])
                count = row.get('Count', 0)

                # 收集所有时长数据
                duration_values = list(video_durations.values())

                # 更新物种统计
                stats['by_species'][species]['total_videos'] += count
                stats['by_species'][species]['total_duration'] += total_duration
                stats['by_species'][species]['duration_list'].extend(duration_values)

                # 记录每个视频的详细信息
                for video_id, duration in video_durations.items():
                    stats['video_details'].append({
                        'video_id': video_id,
                        'species': species,
                        'action': action,
                        'duration': duration
                    })

                    # 添加到总体时长列表
                    stats['all_durations'].append(duration)

                    # 更新最小/最大时长
                    if duration < stats['by_species'][species]['min_duration']:
                        stats['by_species'][species]['min_duration'] = duration
                    if duration > stats['by_species'][species]['max_duration']:
                        stats['by_species'][species]['max_duration'] = duration

                # 更新行为统计
                stats['by_action'][action]['count'] += count
                stats['by_action'][action]['duration'] += total_duration
                stats['by_action'][action]['duration_list'].extend(duration_values)

                # 更新物种-行为关系（包含时长列表）
                stats['species_actions'][species][action]['count'] += count
                stats['species_actions'][species][action]['duration'] += total_duration
                stats['species_actions'][species][action]['duration_list'].extend(duration_values)

                # 更新行为-物种关系
                stats['action_species'][action][species]['count'] += count
                stats['action_species'][action][species]['duration'] += total_duration

    # 4. 计算分布和平均值
    for species in stats['by_species']:
        species_data = stats['by_species'][species]

        # 计算平均时长
        if species_data['total_videos'] > 0:
            species_data['avg_duration'] = (
                    species_data['total_duration'] /
                    species_data['total_videos']
            )

        # 修正最小值
        if species_data['min_duration'] == float('inf'):
            species_data['min_duration'] = 0

        # 计算该物种的时长分布
        if species_data['duration_list']:
            species_data['duration_distribution'] = calculate_duration_distribution(
                species_data['duration_list']
            )

        # 计算每个动作的时长分布
        for action in stats['species_actions'][species]:
            action_data = stats['species_actions'][species][action]
            if action_data['duration_list']:
                action_data['duration_distribution'] = calculate_duration_distribution(
                    action_data['duration_list']
                )
                action_data['avg_duration'] = (
                    action_data['duration'] / action_data['count']
                    if action_data['count'] > 0 else 0
                )

    # 计算总体分布
    if stats['all_durations']:
        stats['duration_distribution'] = calculate_duration_distribution(stats['all_durations'])

    # 计算总体统计
    stats['total_videos'] = sum(s['total_videos'] for s in stats['by_species'].values())
    stats['total_duration'] = sum(s['total_duration'] for s in stats['by_species'].values())

    # 读取其他表格
    df_actions_freq = pd.read_excel(xlsx_file, sheet_name='All_Actions_Freq_Summary')
    df_overall_stats = pd.read_excel(xlsx_file, sheet_name='Overall_Stats_Per_Sheet')
    df_global_category = pd.read_excel(xlsx_file, sheet_name='Global_Category_Summary')

    # 补充类别信息
    for _, row in df_actions_freq.iterrows():
        if pd.notna(row['Action']) and row['Action'] != '全局总计':
            action = row['Action']
            category = row['Category'] if pd.notna(row['Category']) else 'Other'

            if action in stats['by_action']:
                stats['by_action'][action]['category'] = category
                stats['by_category'][category]['count'] += stats['by_action'][action]['count']
                stats['by_category'][category]['duration'] += stats['by_action'][action]['duration']

    # 从CSV补充攻击性和移动性数据
    for _, row in csv_data.iterrows():
        species_name = row['工作表 (Sheet)']
        species_cn = species_mapping.get(species_name, species_name)

        if species_cn in stats['by_species']:
            stats['by_species'][species_cn]['aggressive_pct'] = float(row['Aggressive %'].strip('%'))
            stats['by_species'][species_cn]['movement_pct'] = float(row['Movement %'].strip('%'))

    return stats, species_mapping, df_global_category


def format_distribution_string(distribution, total):
    """格式化分布字符串"""
    if total == 0:
        return "无数据"

    parts = []
    for key, count in distribution.items():
        if count > 0:
            pct = (count / total) * 100
            parts.append(f"{key}:{count}个({pct:.1f}%)")

    return ", ".join(parts)


def generate_enhanced_report_with_detailed_time(stats: dict, species_mapping: dict,
                                                df_global: pd.DataFrame, output_file: str):
    """生成包含详细时间分析的增强版综合报告"""

    with open(output_file, 'w', encoding='utf-8') as f:
        # 报告标题
        f.write("=" * 100 + "\n")
        f.write("Animal_Kingdom视频数据集增强版分析报告\n")
        f.write("=" * 100 + "\n\n")

        # 1. 执行摘要
        f.write("【执行摘要】\n")
        f.write("-" * 50 + "\n")
        f.write(f"• 数据集规模: {stats['total_videos']} 个视频\n")
        f.write(f"• 总时长: {format_duration(stats['total_duration'])} ({stats['total_duration']:.2f}秒)\n")
        f.write(f"• 平均视频时长: {stats['total_duration'] / stats['total_videos']:.2f}秒\n")
        f.write(f"• 物种数量: {len(stats['by_species'])} 个\n")
        f.write(f"• 行为类型: {len(stats['by_action'])} 种\n")
        f.write(f"• 行为类别: {len(stats['by_category'])} 个\n\n")

        # 2. 总体时长分布分析
        f.write("【总体时长分布分析】\n")
        f.write("-" * 50 + "\n")

        total_videos_dist = sum(stats['duration_distribution'].values())
        for range_key in ['<3s', '3-5s', '5-10s', '10-20s', '>20s']:
            count = stats['duration_distribution'].get(range_key, 0)
            if total_videos_dist > 0:
                pct = (count / total_videos_dist) * 100
                f.write(f"  • {range_key}: {count}个视频 ({pct:.1f}%)\n")
        f.write("\n")

        # 3. 物种分布统计（含详细时长分析）
        f.write("【物种分布统计】\n")
        f.write("-" * 50 + "\n")

        sorted_species = sorted(stats['by_species'].items(),
                                key=lambda x: x[1]['total_videos'],
                                reverse=True)

        for species, info in sorted_species:
            percentage = (info['total_videos'] / stats['total_videos']) * 100 if stats['total_videos'] > 0 else 0
            duration_pct = (info['total_duration'] / stats['total_duration']) * 100 if stats[
                                                                                           'total_duration'] > 0 else 0

            f.write(f"\n{species} ({info['original_name']})\n")
            f.write(f"  • 视频数: {info['total_videos']} ({percentage:.1f}%)\n")
            f.write(f"  • 总时长: {format_duration(info['total_duration'])} ({duration_pct:.1f}%)\n")
            f.write(f"  • 平均时长: {info['avg_duration']:.2f}秒\n")
            f.write(f"  • 时长范围: {info['min_duration']:.2f}秒 - {info['max_duration']:.2f}秒\n")
            f.write(f"  • 攻击性: {info.get('aggressive_pct', 0):.1f}%\n")
            f.write(f"  • 移动性: {info.get('movement_pct', 0):.1f}%\n")

            # 该物种的时长分布
            if info.get('duration_distribution'):
                f.write(
                    f"  • 时长分布: {format_distribution_string(info['duration_distribution'], info['total_videos'])}\n")

            # 列出主要行为（含时长）
            behaviors = stats['species_actions'].get(species, {})
            if behaviors:
                top_behaviors = sorted(behaviors.items(),
                                       key=lambda x: x[1]['count'],
                                       reverse=True)[:10]
                f.write("  • 主要行为: ")
                behavior_strs = []
                for b, b_info in top_behaviors:
                    avg_time = b_info['duration'] / b_info['count'] if b_info['count'] > 0 else 0
                    behavior_strs.append(f"{b}({b_info['count']}次,{avg_time:.1f}秒)")
                f.write(", ".join(behavior_strs) + "\n")

            # 该物种每个动作的时长分布（详细展开）
            f.write("\n  【该物种行为时长分布】\n")
            if behaviors:
                sorted_behaviors = sorted(behaviors.items(),
                                          key=lambda x: x[1]['count'],
                                          reverse=True)

                for behavior, b_info in sorted_behaviors[:15]:  # 显示前15个行为
                    if b_info['count'] > 0:
                        avg_duration = b_info['duration'] / b_info['count']
                        f.write(f"    {behavior}:\n")
                        f.write(f"      - 次数: {b_info['count']}, 平均: {avg_duration:.2f}秒\n")

                        # 该行为的时长分布
                        if b_info.get('duration_distribution'):
                            dist_str = format_distribution_string(
                                b_info['duration_distribution'],
                                b_info['count']
                            )
                            f.write(f"      - 分布: {dist_str}\n")

            f.write("\n")

        # 4. 行为类型统计（含时长分布）
        f.write("\n【行为类型统计】\n")
        f.write("-" * 50 + "\n\n")

        # 显示前30个最常见行为的详细统计
        sorted_actions = sorted(stats['by_action'].items(),
                                key=lambda x: x[1]['count'],
                                reverse=True)[:30]

        f.write("Top 30 行为详细统计:\n\n")
        for action, info in sorted_actions:
            if info['count'] > 0:
                avg_duration = info['duration'] / info['count']
                category = info.get('category', 'Unknown')

                f.write(f"{action} (类别: {category})\n")
                f.write(f"  • 总次数: {info['count']}, 总时长: {format_duration(info['duration'])}\n")
                f.write(f"  • 平均时长: {avg_duration:.2f}秒\n")

                # 该行为的时长分布
                if info.get('duration_list'):
                    dist = calculate_duration_distribution(info['duration_list'])
                    dist_str = format_distribution_string(dist, len(info['duration_list']))
                    f.write(f"  • 时长分布: {dist_str}\n")

                # 显示该行为在各物种中的分布
                species_dist = stats['action_species'].get(action, {})
                if len(species_dist) > 1:
                    f.write("  • 物种分布: ")
                    species_strs = []
                    for sp, sp_info in sorted(species_dist.items(),
                                              key=lambda x: x[1]['count'],
                                              reverse=True):
                        species_strs.append(f"{sp}({sp_info['count']})")
                    f.write(", ".join(species_strs) + "\n")

                f.write("\n")

        # 5. 跨物种行为分析
        f.write("\n【跨物种行为分析】\n")
        f.write("-" * 50 + "\n\n")

        # 找出多个物种共有的行为
        f.write("多物种共有行为:\n")
        common_behaviors = []
        for action, species_dist in stats['action_species'].items():
            if len(species_dist) >= 3:
                total_count = sum(s['count'] for s in species_dist.values())
                total_duration = sum(s['duration'] for s in species_dist.values())
                avg_duration = total_duration / total_count if total_count > 0 else 0
                common_behaviors.append((action, len(species_dist), total_count, avg_duration))

        common_behaviors.sort(key=lambda x: x[1], reverse=True)

        for action, species_count, total_count, avg_duration in common_behaviors[:10]:
            f.write(f"  • {action}: {species_count}个物种, 共{total_count}次, ")
            f.write(f"平均{avg_duration:.2f}秒\n")

        # 6. 类别时长统计
        f.write("\n【行为类别时长统计】\n")
        f.write("-" * 50 + "\n")

        sorted_categories = sorted(stats['by_category'].items(),
                                   key=lambda x: x[1]['duration'],
                                   reverse=True)

        for category, info in sorted_categories:
            if stats['total_duration'] > 0:
                pct = (info['duration'] / stats['total_duration']) * 100
                f.write(f"  • {category:<20} {info['count']:>5}个视频  ")
                f.write(f"{format_duration(info['duration']):>12}  ({pct:>5.1f}%)\n")

        # 7. 关键发现
        f.write("\n【关键发现】\n")
        f.write("-" * 50 + "\n")

        # 最大物种群体
        if sorted_species:
            largest_species = sorted_species[0]
            f.write(f"• 最大物种群体: {largest_species[0]}, ")
            f.write(f"{largest_species[1]['total_videos']}个视频\n")

        # 最长平均时长物种
        if stats['by_species']:
            longest_avg = max(stats['by_species'].items(),
                              key=lambda x: x[1]['avg_duration'])
            f.write(f"• 平均时长最长: {longest_avg[0]} ")
            f.write(f"({longest_avg[1]['avg_duration']:.2f}秒)\n")

        # 最短平均时长物种
        if stats['by_species']:
            shortest_avg = min(stats['by_species'].items(),
                               key=lambda x: x[1]['avg_duration'])
            f.write(f"• 平均时长最短: {shortest_avg[0]} ")
            f.write(f"({shortest_avg[1]['avg_duration']:.2f}秒)\n")

        # 最常见行为
        if stats['by_action']:
            top_action = max(stats['by_action'].items(), key=lambda x: x[1]['count'])
            f.write(f"• 最常见行为: {top_action[0]} ({top_action[1]['count']}次)\n")

        # 时长分布特点
        f.write(f"• 时长分布特点: 主要集中在")
        max_range = max(stats['duration_distribution'].items(), key=lambda x: x[1])
        f.write(f"{max_range[0]}范围\n")

        # 报告生成信息
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 100 + "\n")

    print(f"\n📊 增强版报告已生成: {output_file}")


def export_detailed_statistics(stats: dict, output_prefix: str):
    """导出详细的统计数据到多个文件"""

    # 1. 导出物种-行为-时长详细矩阵
    species_action_details = []
    for species in stats['by_species'].keys():
        for action in stats['species_actions'].get(species, {}).keys():
            info = stats['species_actions'][species][action]
            if info['count'] > 0:
                row = {
                    '物种': species,
                    '行为': action,
                    '次数': info['count'],
                    '总时长(秒)': info['duration'],
                    '平均时长(秒)': info['duration'] / info['count'],
                    '<3s': info.get('duration_distribution', {}).get('<3s', 0),
                    '3-5s': info.get('duration_distribution', {}).get('3-5s', 0),
                    '5-10s': info.get('duration_distribution', {}).get('5-10s', 0),
                    '10-20s': info.get('duration_distribution', {}).get('10-20s', 0),
                    '>20s': info.get('duration_distribution', {}).get('>20s', 0)
                }
                species_action_details.append(row)

    if species_action_details:
        df = pd.DataFrame(species_action_details)
        df.to_csv(f'{output_prefix}_species_action_duration_matrix.csv',
                  index=False, encoding='utf-8')
        print(f"✅ 详细矩阵已导出: {output_prefix}_species_action_duration_matrix.csv")

    # 2. 导出每个物种的详细统计
    for species, info in stats['by_species'].items():
        species_data = {
            'summary': {
                'species': species,
                'total_videos': info['total_videos'],
                'total_duration': info['total_duration'],
                'avg_duration': info['avg_duration'],
                'min_duration': info['min_duration'],
                'max_duration': info['max_duration'],
                'duration_distribution': info.get('duration_distribution', {})
            },
            'behaviors': {}
        }

        for action, action_info in stats['species_actions'].get(species, {}).items():
            if action_info['count'] > 0:
                species_data['behaviors'][action] = {
                    'count': action_info['count'],
                    'total_duration': action_info['duration'],
                    'avg_duration': action_info['duration'] / action_info['count'],
                    'distribution': action_info.get('duration_distribution', {})
                }

        # 保存为JSON
        filename = f"{output_prefix}_{species}_detailed.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(species_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 各物种详细数据已导出")


# 主程序
if __name__ == "__main__":
    # 文件路径
    xlsx_file = r'H:\big_cat_dataset\Animal_Kingdom(download_video.tar.gz_not_download_image.tar.gz)\output_stats\comprehensive_video_summary.xlsx'
    csv_file = r'H:\big_cat_dataset\Animal_Kingdom(download_video.tar.gz_not_download_image.tar.gz)\output_stats\cross_sheet_summary.csv'

    try:
        print("\n" + "🎬" * 30)
        print("开始分析视频数据集（含详细时长维度）")
        print("🎬" * 30)

        # 执行分析
        stats, species_mapping, df_global = analyze_video_dataset_with_detailed_time(xlsx_file, csv_file)

        # 生成报告文件
        print("\n" + "=" * 80)
        print("📝 正在生成报告文件...")
        print("=" * 80)

        # 生成增强版报告（含详细时长分析）
        generate_enhanced_report_with_detailed_time(stats, species_mapping, df_global,
                                                    'video_dataset_enhanced_report.txt')

        # 导出详细统计数据
        export_detailed_statistics(stats, 'video_stats')

        # 导出汇总JSON
        summary_stats = {
            'total_videos': stats['total_videos'],
            'total_duration_seconds': stats['total_duration'],
            'total_duration_formatted': format_duration(stats['total_duration']),
            'average_duration_seconds': stats['total_duration'] / stats['total_videos'] if stats[
                                                                                               'total_videos'] > 0 else 0,
            'overall_distribution': stats['duration_distribution'],
            'species_summary': {}
        }

        for species, info in stats['by_species'].items():
            summary_stats['species_summary'][species] = {
                'total_videos': info['total_videos'],
                'total_duration': info['total_duration'],
                'avg_duration': info['avg_duration'],
                'distribution': info.get('duration_distribution', {}),
                'aggressive_pct': info.get('aggressive_pct', 0),
                'movement_pct': info.get('movement_pct', 0)
            }

        with open('video_dataset_summary_stats.json', 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 80)
        print("✅ 分析完成！")
        print("=" * 80)
        print("\n📚 生成的文件列表：")
        print("📄 video_dataset_enhanced_report.txt         - 增强版综合分析报告")
        print("📊 video_stats_species_action_duration_matrix.csv - 详细时长矩阵")
        print("📈 video_dataset_summary_stats.json          - 汇总统计数据")
        print("📁 video_stats_[物种]_detailed.json          - 每个物种的详细数据")
        print("=" * 80)

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback

        traceback.print_exc()