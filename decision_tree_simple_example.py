import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

class SimpleDecisionTree:
    """
    最简单的决策树实现，用于理解概念
    """
    
    def __init__(self):
        self.tree = None
    
    def create_simple_tree(self):
        """
        创建一个简单的决策树示例
        问题：根据天气和温度决定是否去公园
        """
        # 手动构建一个简单的决策树
        self.tree = {
            'question': '天气如何？',
            'answers': {
                '晴天': {
                    'question': '温度如何？',
                    'answers': {
                        '热': '不去公园',
                        '温暖': '去公园',
                        '凉爽': '去公园'
                    }
                },
                '阴天': {
                    'question': '温度如何？',
                    'answers': {
                        '热': '不去公园',
                        '温暖': '去公园',
                        '凉爽': '不去公园'
                    }
                },
                '雨天': '不去公园'
            }
        }
    
    def predict(self, weather: str, temperature: str) -> str:
        """
        根据天气和温度预测是否去公园
        """
        if self.tree is None:
            self.create_simple_tree()
        
        # 第一层判断：天气
        weather_answers = self.tree['answers']
        if weather not in weather_answers:
            return "未知天气"
        
        weather_result = weather_answers[weather]
        
        # 如果是叶子节点（直接给出答案）
        if isinstance(weather_result, str):
            return weather_result
        
        # 第二层判断：温度
        temp_answers = weather_result['answers']
        if temperature not in temp_answers:
            return "未知温度"
        
        return temp_answers[temperature]
    
    def print_tree(self, node=None, indent=0):
        """
        打印决策树结构
        """
        if node is None:
            node = self.tree
            if node is None:
                self.create_simple_tree()
                node = self.tree
        
        if isinstance(node, str):
            print("  " * indent + f"结果: {node}")
        else:
            print("  " * indent + f"问题: {node['question']}")
            for answer, next_node in node['answers'].items():
                print("  " * (indent + 1) + f"如果 {answer}:")
                self.print_tree(next_node, indent + 2)

def demonstrate_decision_tree():
    """
    演示决策树的工作原理
    """
    print("=== 决策树是什么？ ===\n")
    
    print("决策树就像是一系列的问题和答案：")
    print("1. 你问一个问题")
    print("2. 根据答案，你问下一个问题")
    print("3. 重复这个过程，直到得到最终答案")
    print()
    
    # 创建决策树
    tree = SimpleDecisionTree()
    tree.create_simple_tree()
    
    print("=== 决策树结构 ===")
    tree.print_tree()
    print()
    
    print("=== 实际预测示例 ===")
    test_cases = [
        ("晴天", "温暖"),
        ("晴天", "热"),
        ("阴天", "凉爽"),
        ("雨天", "温暖"),
        ("晴天", "凉爽")
    ]
    
    for weather, temp in test_cases:
        result = tree.predict(weather, temp)
        print(f"天气: {weather}, 温度: {temp} → 决定: {result}")
    
    print("\n=== 决策树的优点 ===")
    print("1. 容易理解 - 就像人类做决定的过程")
    print("2. 可以处理多种条件")
    print("3. 结果可以解释")
    print("4. 不需要复杂的数学计算")

def create_visual_decision_tree():
    """
    创建决策树的可视化
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 绘制决策树结构
    # 根节点
    ax.text(0.5, 0.9, "天气如何？", ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"), fontsize=12)
    
    # 第一层分支
    ax.text(0.2, 0.7, "晴天", ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen"), fontsize=10)
    ax.text(0.5, 0.7, "阴天", ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen"), fontsize=10)
    ax.text(0.8, 0.7, "雨天", ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen"), fontsize=10)
    
    # 第二层问题
    ax.text(0.2, 0.5, "温度如何？", ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"), fontsize=10)
    ax.text(0.5, 0.5, "温度如何？", ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"), fontsize=10)
    ax.text(0.8, 0.5, "不去公园", ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcoral"), fontsize=10)
    
    # 最终结果
    ax.text(0.1, 0.3, "热\n不去", ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcoral"), fontsize=9)
    ax.text(0.2, 0.3, "温暖\n去公园", ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen"), fontsize=9)
    ax.text(0.3, 0.3, "凉爽\n去公园", ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen"), fontsize=9)
    
    ax.text(0.4, 0.3, "热\n不去", ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcoral"), fontsize=9)
    ax.text(0.5, 0.3, "温暖\n去公园", ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen"), fontsize=9)
    ax.text(0.6, 0.3, "凉爽\n不去", ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcoral"), fontsize=9)
    
    # 绘制连接线
    # 从根节点到第一层
    ax.plot([0.5, 0.2], [0.85, 0.75], 'k-', linewidth=2)
    ax.plot([0.5, 0.5], [0.85, 0.75], 'k-', linewidth=2)
    ax.plot([0.5, 0.8], [0.85, 0.75], 'k-', linewidth=2)
    
    # 从第一层到第二层
    ax.plot([0.2, 0.2], [0.65, 0.55], 'k-', linewidth=2)
    ax.plot([0.5, 0.5], [0.65, 0.55], 'k-', linewidth=2)
    
    # 从第二层到结果
    ax.plot([0.2, 0.1], [0.45, 0.35], 'k-', linewidth=1)
    ax.plot([0.2, 0.2], [0.45, 0.35], 'k-', linewidth=1)
    ax.plot([0.2, 0.3], [0.45, 0.35], 'k-', linewidth=1)
    ax.plot([0.5, 0.4], [0.45, 0.35], 'k-', linewidth=1)
    ax.plot([0.5, 0.5], [0.45, 0.35], 'k-', linewidth=1)
    ax.plot([0.5, 0.6], [0.45, 0.35], 'k-', linewidth=1)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 1)
    ax.set_title("决策树示例：是否去公园", fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def real_world_examples():
    """
    现实世界中的决策树例子
    """
    print("=== 现实世界中的决策树例子 ===\n")
    
    examples = [
        {
            "场景": "医生诊断",
            "问题": "患者是否患有某种疾病？",
            "决策过程": [
                "体温是否超过38度？",
                "是否有咳嗽症状？",
                "是否接触过感染者？"
            ],
            "结果": "诊断结果和治疗建议"
        },
        {
            "场景": "银行贷款",
            "问题": "是否批准贷款申请？",
            "决策过程": [
                "收入是否足够？",
                "信用记录是否良好？",
                "是否有抵押品？"
            ],
            "结果": "批准或拒绝贷款"
        },
        {
            "场景": "推荐系统",
            "问题": "推荐什么类型的电影？",
            "决策过程": [
                "用户年龄？",
                "喜欢的电影类型？",
                "观看时间偏好？"
            ],
            "结果": "推荐特定电影"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['场景']}")
        print(f"   问题: {example['问题']}")
        print("   决策过程:")
        for step in example['决策过程']:
            print(f"     - {step}")
        print(f"   结果: {example['结果']}")
        print()

def main():
    """
    主函数：完整演示决策树概念
    """
    print("🎯 决策树完全理解指南\n")
    
    # 1. 基本概念演示
    demonstrate_decision_tree()
    
    print("\n" + "="*50 + "\n")
    
    # 2. 现实世界例子
    real_world_examples()
    
    print("\n" + "="*50 + "\n")
    
    # 3. 可视化
    print("=== 决策树可视化 ===")
    create_visual_decision_tree()
    
    print("\n=== 总结 ===")
    print("决策树就是：")
    print("🌳 像树一样的结构")
    print("❓ 每个节点是一个问题")
    print("🌿 每个分支是一个答案")
    print("🍃 每个叶子是最终结果")
    print("🤔 从根到叶子的路径就是决策过程")

if __name__ == "__main__":
    main()
