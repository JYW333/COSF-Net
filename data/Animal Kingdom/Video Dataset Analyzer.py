import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm


class VideoDatasetAnalyzer:
    def __init__(self, dataset_path):
        """
        初始化视频数据集分析器

        Args:
            dataset_path: 数据集根目录路径
        """
        self.dataset_path = Path(dataset_path)
        self.stats = defaultdict(lambda: {
            'count': 0,
            'total_duration': 0,
            'durations': [],
            'file_names': []
        })

    def get_video_duration(self, video_path):
        """
        获取视频时长（秒）

        Args:
            video_path: 视频文件路径

        Returns:
            视频时长（秒）
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            return duration
        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            return 0

    def analyze_dataset(self):
        """
        分析整个数据集
        """
        print("开始分析数据集...")

        # 遍历所有分割（train, val, test等）
        for split_dir in self.dataset_path.iterdir():
            if not split_dir.is_dir():
                continue

            split_name = split_dir.name
            print(f"\n分析 {split_name} 集...")

            # 遍历每个行为类别
            for action_dir in split_dir.iterdir():
                if not action_dir.is_dir():
                    continue

                action_name = action_dir.name
                full_key = f"{split_name}/{action_name}"

                # 获取所有视频文件
                video_files = list(action_dir.glob("*.mp4"))

                print(f"  处理 {action_name} ({len(video_files)} 个视频)...")

                # 分析每个视频
                for video_file in tqdm(video_files, desc=f"    {action_name}", leave=False):
                    duration = self.get_video_duration(video_file)

                    self.stats[full_key]['count'] += 1
                    self.stats[full_key]['total_duration'] += duration
                    self.stats[full_key]['durations'].append(duration)
                    self.stats[full_key]['file_names'].append(video_file.name)

    def generate_report(self):
        """
        生成统计报告
        """
        print("\n" + "=" * 80)
        print("数据集统计报告")
        print("=" * 80)

        # 创建DataFrame用于更好的展示
        report_data = []

        for key, data in sorted(self.stats.items()):
            split, action = key.split('/')

            if data['count'] > 0:
                durations = data['durations']
                report_data.append({
                    '数据集': split,
                    '行为类别': action,
                    '视频数量': data['count'],
                    '总时长(秒)': round(data['total_duration'], 2),
                    '总时长(分钟)': round(data['total_duration'] / 60, 2),
                    '平均时长(秒)': round(np.mean(durations), 2),
                    '最短时长(秒)': round(min(durations), 2),
                    '最长时长(秒)': round(max(durations), 2),
                    '时长标准差(秒)': round(np.std(durations), 2)
                })

        df = pd.DataFrame(report_data)

        # 按数据集分组显示
        for split in df['数据集'].unique():
            split_df = df[df['数据集'] == split]
            print(f"\n【{split} 集统计】")
            print(split_df[['行为类别', '视频数量', '总时长(分钟)',
                            '平均时长(秒)', '最短时长(秒)', '最长时长(秒)']].to_string(index=False))

            # 汇总统计
            total_videos = split_df['视频数量'].sum()
            total_duration = split_df['总时长(分钟)'].sum()
            print(f"\n  {split}集总计: {total_videos} 个视频, 总时长 {total_duration:.2f} 分钟")

        # 总体统计
        print("\n" + "=" * 80)
        print("【总体统计】")
        print(f"总视频数量: {df['视频数量'].sum()}")
        print(f"总时长: {df['总时长(分钟)'].sum():.2f} 分钟 ({df['总时长(分钟)'].sum() / 60:.2f} 小时)")
        print(f"行为类别数: {df['行为类别'].nunique()}")

        return df

    def plot_duration_distribution(self, save_path=None):
        """
        绘制视频时长分布图

        Args:
            save_path: 保存图片的路径（可选）
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        # 准备数据
        all_durations = []
        labels = []

        for key, data in sorted(self.stats.items()):
            if data['count'] > 0:
                all_durations.extend(data['durations'])
                labels.extend([key] * len(data['durations']))

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 总体时长分布直方图
        ax1 = axes[0, 0]
        ax1.hist(all_durations, bins=30, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('视频时长 (秒)')
        ax1.set_ylabel('频次')
        ax1.set_title('所有视频时长分布')
        ax1.grid(True, alpha=0.3)

        # 2. 按行为类别的箱线图
        ax2 = axes[0, 1]
        duration_df = pd.DataFrame({'时长': all_durations, '类别': labels})

        # 重新组织数据用于箱线图
        categories = duration_df['类别'].unique()
        data_for_boxplot = [duration_df[duration_df['类别'] == cat]['时长'].values
                            for cat in categories]

        bp = ax2.boxplot(data_for_boxplot, labels=[cat.split('/')[-1] for cat in categories])
        ax2.set_xlabel('行为类别')
        ax2.set_ylabel('视频时长 (秒)')
        ax2.set_title('各行为类别时长分布箱线图')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        # 3. 按数据集分割的时长分布
        ax3 = axes[1, 0]
        splits = {}
        for key, data in self.stats.items():
            split = key.split('/')[0]
            if split not in splits:
                splits[split] = []
            splits[split].extend(data['durations'])

        for split, durations in splits.items():
            if durations:
                ax3.hist(durations, bins=20, alpha=0.5, label=split, edgecolor='black')

        ax3.set_xlabel('视频时长 (秒)')
        ax3.set_ylabel('频次')
        ax3.set_title('不同数据集的时长分布对比')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 累积分布函数(CDF)
        ax4 = axes[1, 1]
        sorted_durations = np.sort(all_durations)
        cdf = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
        ax4.plot(sorted_durations, cdf, linewidth=2)
        ax4.set_xlabel('视频时长 (秒)')
        ax4.set_ylabel('累积概率')
        ax4.set_title('视频时长累积分布函数')
        ax4.grid(True, alpha=0.3)

        # 添加一些统计线
        percentiles = [25, 50, 75]
        for p in percentiles:
            val = np.percentile(all_durations, p)
            ax4.axvline(val, color='r', linestyle='--', alpha=0.5)
            ax4.text(val, 0.05, f'{p}%: {val:.1f}s', rotation=90)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n图表已保存至: {save_path}")

        plt.show()

    def export_detailed_csv(self, output_path='dataset_details.csv'):
        """
        导出详细的CSV文件，包含每个视频的信息

        Args:
            output_path: CSV文件保存路径
        """
        detailed_data = []

        for key, data in self.stats.items():
            split, action = key.split('/')

            for i, (filename, duration) in enumerate(zip(data['file_names'], data['durations'])):
                detailed_data.append({
                    '数据集': split,
                    '行为类别': action,
                    '文件名': filename,
                    '时长(秒)': round(duration, 2),
                    '时长(分钟)': round(duration / 60, 2)
                })

        df = pd.DataFrame(detailed_data)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n详细数据已导出至: {output_path}")

        return df


# 使用示例
if __name__ == "__main__":
    # 设置你的数据集路径
    dataset_path = r"H:\big_cat_dataset\Animal_Kingdom(download_video.tar.gz_not_download_image.tar.gz)\animal_kingdom_kinetics\done"  # 根据你的实际路径修改

    # 创建分析器实例
    analyzer = VideoDatasetAnalyzer(dataset_path)

    # 分析数据集
    analyzer.analyze_dataset()

    # 生成报告
    report_df = analyzer.generate_report()

    # 绘制分布图
    analyzer.plot_duration_distribution(save_path='video_duration_distribution.png')

    # 导出详细CSV
    detailed_df = analyzer.export_detailed_csv('video_dataset_details.csv')

    print("\n分析完成！")