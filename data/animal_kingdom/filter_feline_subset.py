#!/usr/bin/env python3
"""
Animal Kingdom Top-12 行为提取脚本 V2
- 使用原有的Type分类（train/test）
- 将test改为val以符合Kinetics格式
- 生成标准Kinetics格式的标注文件
"""

import os
import json
import csv
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
import re

# Top 12 行为（按总时长排序，基于报告数据）
TOP_12_BEHAVIORS = {
    'Walking': {'total_minutes': 61.2, 'count': 490, 'avg_duration': 7.52},
    'Sensing': {'total_minutes': 40.29, 'count': 294, 'avg_duration': 8.22},
    'Keeping still': {'total_minutes': 21.77, 'count': 159, 'avg_duration': 8.22},
    'Barking': {'total_minutes': 19.51, 'count': 117, 'avg_duration': 10.01},
    'Attending': {'total_minutes': 17.26, 'count': 123, 'avg_duration': 8.42},
    'Sitting': {'total_minutes': 14.18, 'count': 79, 'avg_duration': 10.77},
    'Retreating': {'total_minutes': 13.65, 'count': 89, 'avg_duration': 9.20},
    'Shaking Head': {'total_minutes': 11.36, 'count': 56, 'avg_duration': 12.17},
    'Attacking': {'total_minutes': 10.87, 'count': 91, 'avg_duration': 7.17},
    'Startled': {'total_minutes': 10.75, 'count': 60, 'avg_duration': 10.75},
    'Standing': {'total_minutes': 10.74, 'count': 56, 'avg_duration': 11.51},
    'Running': {'total_minutes': 8.47, 'count': 87, 'avg_duration': 5.84}
}

# 行为到ID的映射（0-11）
BEHAVIOR_TO_ID = {
    'Walking': 0,
    'Sensing': 1,
    'Keeping still': 2,
    'Barking': 3,
    'Attending': 4,
    'Sitting': 5,
    'Retreating': 6,
    'Shaking Head': 7,
    'Attacking': 8,
    'Startled': 9,
    'Standing': 10,
    'Running': 11
}


class AnimalKingdomToKinetics:
    def __init__(self, excel_file, output_dir, video_dir=None):
        """
        初始化转换器

        Args:
            excel_file: Excel统计文件路径
            output_dir: 输出目录
            video_dir: 视频文件目录（可选）
        """
        self.excel_file = Path(excel_file)
        self.output_dir = Path(output_dir)
        self.video_dir = Path(video_dir) if video_dir else None

        # 创建输出目录结构
        self.setup_directories()

    def setup_directories(self):
        """创建Kinetics-400格式的目录结构"""
        # 创建基础目录
        for split in ['train', 'val']:
            split_dir = self.output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            # 为每个行为创建子目录
            for behavior in TOP_12_BEHAVIORS.keys():
                behavior_dir = split_dir / behavior.replace(' ', '_').lower()
                behavior_dir.mkdir(parents=True, exist_ok=True)

        print(f"✓ 目录结构创建完成: {self.output_dir}")

    def load_and_process_data(self):
        """加载Excel数据并处理"""
        print("\n1. 加载Excel数据...")

        # 读取All_Video_IDs_Summary sheet
        df = pd.read_excel(self.excel_file, sheet_name='All_Video_IDs_Summary')

        # 过滤出Top 12行为
        df_filtered = df[df['Action'].isin(TOP_12_BEHAVIORS.keys())].copy()

        # 过滤掉Type为'Total'的行（这些是汇总行）
        df_filtered = df_filtered[df_filtered['Type'].isin(['train', 'test'])]

        print(f"   原始数据: {len(df)} 行")
        print(f"   过滤后Top-12行为: {len(df_filtered)} 行")

        # 统计Type分布
        type_counts = df_filtered['Type'].value_counts()
        print(f"\n   Type分布:")
        print(f"     train: {type_counts.get('train', 0)} 行")
        print(f"     test→val: {type_counts.get('test', 0)} 行")

        return df_filtered

    def parse_video_ids(self, video_ids_str):
        """
        解析Video_IDs字段，提取视频ID列表

        Args:
            video_ids_str: Video_IDs字段的字符串

        Returns:
            list: 视频ID列表
        """
        if pd.isna(video_ids_str) or video_ids_str == '':
            return []

        # 处理不同的分隔符
        video_ids_str = str(video_ids_str)

        # 尝试用逗号、分号、空格等分隔
        video_ids = re.split('[,;\\s]+', video_ids_str)

        # 清理每个ID
        video_ids = [vid.strip() for vid in video_ids if vid.strip()]

        return video_ids

    def parse_durations(self, durations_str):
        """
        解析Video_Durations字段，提取时长列表

        Args:
            durations_str: Video_Durations字段的字符串

        Returns:
            list: 时长列表
        """
        if pd.isna(durations_str) or durations_str == '':
            return []

        durations_str = str(durations_str)

        # 提取所有数字（包括小数）
        durations = re.findall(r'[\d.]+', durations_str)

        # 转换为浮点数
        durations = [float(d) for d in durations]

        return durations

    def generate_kinetics_annotations(self, df):
        """
        生成Kinetics格式的标注文件
        格式: split/action_name/video_id.mp4 label_id
        """
        print("\n2. 生成Kinetics格式标注...")

        # 创建标注字典
        annotations = {
            'train': [],
            'val': []
        }

        # 统计信息
        stats = {
            'train': defaultdict(list),
            'val': defaultdict(list)
        }

        # 处理每一行数据
        for _, row in df.iterrows():
            action = row['Action']
            type_orig = row['Type']

            # train保持不变，test改为val
            split = 'train' if type_orig == 'train' else 'val'

            # 获取行为ID
            action_id = BEHAVIOR_TO_ID.get(action, -1)
            if action_id == -1:
                continue

            # 解析视频ID和时长
            video_ids = self.parse_video_ids(row['Video_IDs'])
            durations = self.parse_durations(row['Video_Durations'])

            # 确保视频ID和时长数量匹配
            if not video_ids:
                continue

            # 如果时长数量不匹配，使用平均时长
            if len(durations) != len(video_ids):
                avg_duration = TOP_12_BEHAVIORS.get(action, {}).get('avg_duration', 8.0)
                durations = [avg_duration] * len(video_ids)

            # 生成标注
            action_folder = action.replace(' ', '_').lower()

            for video_id, duration in zip(video_ids, durations):
                # Kinetics格式: split/action_name/video_id.mp4 label_id
                annotation_line = f"{split}/{action_folder}/{video_id}.mp4 {action_id}"
                annotations[split].append(annotation_line)

                # 统计信息
                stats[split][action].append({
                    'video_id': video_id,
                    'duration': duration
                })

        # 保存标注文件
        self.save_annotations(annotations, stats)

        return annotations, stats

    def save_annotations(self, annotations, stats):
        """保存标注文件和统计信息"""

        # 1. 保存Kinetics格式的.lst文件
        for split, lines in annotations.items():
            lst_file = self.output_dir / f'{split}.lst'
            with open(lst_file, 'w') as f:
                for line in sorted(lines):  # 排序以保持一致性
                    f.write(line + '\n')
            print(f"✓ 保存 {lst_file}: {len(lines)} 个视频")

        # 2. 保存标签映射文件
        labels_file = self.output_dir / 'labels.csv'
        with open(labels_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name'])
            for behavior, idx in sorted(BEHAVIOR_TO_ID.items(), key=lambda x: x[1]):
                writer.writerow([idx, behavior.replace(' ', '_').lower()])
        print(f"✓ 保存标签映射: {labels_file}")

        # 3. 保存JSON格式（额外的，方便其他框架使用）
        for split, lines in annotations.items():
            json_data = []
            for line in lines:
                parts = line.split()
                path_parts = parts[0].split('/')

                json_data.append({
                    'video': parts[0],
                    'label': int(parts[1]),
                    'label_name': path_parts[1],
                    'split': split
                })

            json_file = self.output_dir / f'{split}.json'
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"✓ 保存JSON标注: {json_file}")

        # 4. 保存统计信息
        self.save_statistics(stats)

    def save_statistics(self, stats):
        """保存详细的统计信息"""

        statistics = {
            'dataset': 'Animal Kingdom Top-12 Behaviors',
            'format': 'Kinetics-400',
            'num_classes': len(BEHAVIOR_TO_ID),
            'class_names': list(BEHAVIOR_TO_ID.keys()),
            'class_ids': BEHAVIOR_TO_ID,
            'splits': {}
        }

        print("\n3. 数据集统计:")
        print("=" * 50)

        for split, split_stats in stats.items():
            split_info = {
                'num_videos': sum(len(videos) for videos in split_stats.values()),
                'class_distribution': {}
            }

            print(f"\n{split.upper()}集:")
            print(f"  总视频数: {split_info['num_videos']}")
            print(f"  类别分布:")

            for behavior in sorted(BEHAVIOR_TO_ID.keys()):
                videos = split_stats.get(behavior, [])
                count = len(videos)

                if count > 0:
                    total_duration = sum(v['duration'] for v in videos)
                    avg_duration = total_duration / count

                    split_info['class_distribution'][behavior] = {
                        'count': count,
                        'total_duration': round(total_duration, 2),
                        'avg_duration': round(avg_duration, 2)
                    }

                    print(f"    {behavior:15s}: {count:4d} 个视频, "
                          f"平均时长 {avg_duration:.2f}秒")

            statistics['splits'][split] = split_info

        # 保存统计文件
        stats_file = self.output_dir / 'dataset_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 保存统计信息: {stats_file}")

    def copy_videos(self, stats):
        """复制视频文件到对应目录（如果video_dir存在）"""
        if not self.video_dir or not self.video_dir.exists():
            print("\n⚠️ 视频目录不存在，跳过视频复制")
            return

        print("\n4. 复制视频文件...")
        copied_count = 0
        missing_count = 0

        for split, split_stats in stats.items():
            for behavior, videos in split_stats.items():
                behavior_dir = behavior.replace(' ', '_').lower()
                target_dir = self.output_dir / split / behavior_dir

                for video_info in videos:
                    video_id = video_info['video_id']

                    # 尝试不同的文件扩展名
                    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                        source_file = self.video_dir / f"{video_id}{ext}"

                        if source_file.exists():
                            target_file = target_dir / f"{video_id}.mp4"
                            shutil.copy2(source_file, target_file)
                            copied_count += 1
                            break
                    else:
                        missing_count += 1

        print(f"✓ 复制了 {copied_count} 个视频文件")
        if missing_count > 0:
            print(f"⚠️ {missing_count} 个视频文件未找到")

    def process(self):
        """执行完整的转换流程"""
        print("=" * 60)
        print("Animal Kingdom -> Kinetics-400 格式转换 (V2)")
        print("=" * 60)

        # 1. 加载和处理数据
        df = self.load_and_process_data()

        # 2. 生成标注
        annotations, stats = self.generate_kinetics_annotations(df)

        # 3. 复制视频文件（如果可能）
        self.copy_videos(stats)

        print("\n" + "=" * 60)
        print("✅ 转换完成!")
        print(f"输出目录: {self.output_dir}")
        print("\n生成的文件:")
        print("  - train.lst: 训练集标注（Kinetics格式）")
        print("  - val.lst: 验证集标注（Kinetics格式）")
        print("  - labels.csv: 标签映射")
        print("  - train.json, val.json: JSON格式标注")
        print("  - dataset_statistics.json: 统计信息")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='将Animal Kingdom数据集转换为Kinetics-400格式（使用原有Type分类）'
    )
    parser.add_argument(
        '--excel', '-e',
        type=str,
        default=r'H:\big_cat_dataset\Animal_Kingdom(download_video.tar.gz_not_download_image.tar.gz)\output_stats\comprehensive_video_summary.xlsx',
        help='Excel统计文件路径'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./animal_kingdom_kinetics',
        help='输出目录路径'
    )
    parser.add_argument(
        '--video-dir', '-v',
        type=str,
        default=r'H:\big_cat_dataset\Animal_Kingdom(download_video.tar.gz_not_download_image.tar.gz)\action_recognition\dataset\video\video',
        help='视频文件目录（可选）'
    )

    args = parser.parse_args()

    # 创建转换器并执行
    converter = AnimalKingdomToKinetics(
        excel_file=args.excel,
        output_dir=args.output,
        video_dir=args.video_dir
    )

    converter.process()


if __name__ == '__main__':
    main()