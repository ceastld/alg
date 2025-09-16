"""
LeetCode 621. Task Scheduler

题目描述：
给定一个任务列表tasks和一个整数n，表示相同任务之间需要间隔n个时间单位。
每个时间单位可以执行一个任务或保持空闲。
求完成所有任务的最少时间。

示例：
tasks = ["A","A","A","B","B","B"], n = 2
输出：8
解释：A -> B -> idle -> A -> B -> idle -> A -> B

数据范围：
- 1 <= tasks.length <= 10^4
- tasks[i] 是大写英文字母
- 0 <= n <= 100
"""

class Solution:
    def leastInterval(self, tasks: list[str], n: int) -> int:
        from collections import Counter
        
        # 统计每个任务的频次
        task_counts = Counter(tasks)
        max_count = max(task_counts.values())
        
        # 计算具有最大频次的任务数量
        max_count_tasks = sum(1 for count in task_counts.values() if count == max_count)
        
        # 计算最小时间间隔
        min_intervals = (max_count - 1) * (n + 1) + max_count_tasks
        
        return max(min_intervals, len(tasks))
