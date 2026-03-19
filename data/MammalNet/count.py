import os
import cv2
from pathlib import Path
from collections import defaultdict
import json


def get_video_info(video_path):
    """获取单个视频的详细信息"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 计算时长（秒）
        duration = frame_count / fps if fps > 0 else 0

        # 获取文件大小（MB）
        file_size = os.path.getsize(video_path) / (1024 * 1024)

        cap.release()

        return {
            'duration': duration,
            'fps': fps,
            'frame_count': frame_count,
            'resolution': (width, height),
            'file_size': file_size
        }
    except Exception as e:
        print(f"错误读取视频 {video_path}: {e}")
        return None


def analyze_dataset(root_path, video_extensions=['.mp4', '.avi', '.mov', '.mkv']):
    """分析整个数据集"""
    root = Path(root_path)

    # 存储统计信息
    stats = {
        'train': defaultdict(list),
        'val': defaultdict(list)
    }

    # 遍历train和val文件夹
    for split in ['train', 'val']:
        split_path = root / split

        if not split_path.exists():
            print(f"警告: {split_path} 不存在")
            continue

        # 遍历每个动作类别文件夹
        for action_folder in split_path.iterdir():
            if not action_folder.is_dir():
                continue

            action_name = action_folder.name
            print(f"正在处理: {split}/{action_name}")

            # 遍历该类别下的所有视频
            for video_file in action_folder.iterdir():
                if video_file.suffix.lower() in video_extensions:
                    info = get_video_info(video_file)
                    if info:
                        stats[split][action_name].append(info)

    return stats


def print_statistics(stats):
    """打印详细统计信息"""
    print("\n" + "=" * 80)
    print("数据集统计报告".center(80))
    print("=" * 80 + "\n")

    total_videos = 0
    total_duration = 0
    all_resolutions = set()
    all_fps = []

    for split in ['train', 'val']:
        if not stats[split]:
            continue

        print(f"\n{'【' + split.upper() + ' 集】':-^76}")

        split_videos = 0
        split_duration = 0

        # 按类别统计
        for action, videos in sorted(stats[split].items()):
            if not videos:
                continue

            num_videos = len(videos)
            durations = [v['duration'] for v in videos]
            file_sizes = [v['file_size'] for v in videos]
            resolutions = [v['resolution'] for v in videos]
            fps_list = [v['fps'] for v in videos]
            frame_counts = [v['frame_count'] for v in videos]

            avg_duration = sum(durations) / num_videos
            total_class_duration = sum(durations)
            avg_fps = sum(fps_list) / num_videos
            avg_frames = sum(frame_counts) / num_videos
            avg_size = sum(file_sizes) / num_videos

            # 统计分辨率分布
            resolution_counts = {}
            for res in resolutions:
                res_str = f"{res[0]}x{res[1]}"
                resolution_counts[res_str] = resolution_counts.get(res_str, 0) + 1
                all_resolutions.add(res_str)

            print(f"\n  类别: {action}")
            print(f"    视频数量: {num_videos}")
            print(f"    总时长: {total_class_duration:.2f}秒 ({total_class_duration / 60:.2f}分钟)")
            print(f"    平均时长: {avg_duration:.2f}秒")
            print(f"    时长范围: {min(durations):.2f}秒 ~ {max(durations):.2f}秒")
            print(f"    平均帧率: {avg_fps:.2f} FPS")
            print(f"    平均帧数: {avg_frames:.0f} 帧")
            print(f"    平均文件大小: {avg_size:.2f} MB")
            print(f"    分辨率分布: {', '.join([f'{k}({v}个)' for k, v in sorted(resolution_counts.items())])}")

            split_videos += num_videos
            split_duration += total_class_duration
            all_fps.extend(fps_list)

        print(f"\n  {split.upper()}集总计:")
        print(f"    类别数: {len(stats[split])}")
        print(f"    视频总数: {split_videos}")
        print(f"    总时长: {split_duration:.2f}秒 ({split_duration / 60:.2f}分钟, {split_duration / 3600:.2f}小时)")

        total_videos += split_videos
        total_duration += split_duration

    # 全局统计
    print(f"\n{'【全局统计】':-^76}")
    all_actions = set(stats['train'].keys()) | set(stats['val'].keys())
    print(f"  总类别数: {len(all_actions)}")
    print(f"  所有类别: {', '.join(sorted(all_actions))}")
    print(f"  视频总数: {total_videos}")
    print(f"  总时长: {total_duration:.2f}秒 ({total_duration / 60:.2f}分钟, {total_duration / 3600:.2f}小时)")
    print(f"  平均视频时长: {total_duration / total_videos:.2f}秒" if total_videos > 0 else "  平均视频时长: N/A")
    print(f"  所有分辨率: {', '.join(sorted(all_resolutions))}")
    if all_fps:
        print(f"  帧率范围: {min(all_fps):.2f} ~ {max(all_fps):.2f} FPS")

    print("\n" + "=" * 80 + "\n")


def save_to_json(stats, output_file='dataset_stats.json'):
    """保存统计信息到JSON文件"""
    # 转换为可JSON序列化的格式
    json_stats = {}
    for split in stats:
        json_stats[split] = {}
        for action, videos in stats[split].items():
            json_stats[split][action] = {
                'count': len(videos),
                'durations': [v['duration'] for v in videos],
                'avg_duration': sum(v['duration'] for v in videos) / len(videos) if videos else 0,
                'resolutions': [v['resolution'] for v in videos],
                'fps': [v['fps'] for v in videos],
            }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_stats, f, indent=2, ensure_ascii=False)

    print(f"统计信息已保存到: {output_file}")


def main():
    # 修改这里为你的数据集路径
    dataset_path = r"H:\big_cat_dataset\Animal_Kingdom(download_video.tar.gz_not_download_image.tar.gz)\animal_kingdom_kinetics\done\kinetics-4"  # 例如: "/path/to/dataset" 或 "D:/datasets/my_dataset"

    print("开始分析数据集...")
    print(f"数据集路径: {dataset_path}\n")

    # 分析数据集
    stats = analyze_dataset(dataset_path)

    # 打印统计信息
    print_statistics(stats)

    # 保存到JSON文件
    save_to_json(stats)


if __name__ == "__main__":
    main()