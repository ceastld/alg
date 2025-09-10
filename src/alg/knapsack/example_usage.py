"""
背包问题使用示例

展示如何在实际问题中应用各种背包算法
"""

from .knapsack import (
    solve_01_knapsack, solve_complete_knapsack, solve_multiple_knapsack,
    solve_group_knapsack, solve_2d_knapsack, ZeroOneKnapsack
)


def example_shopping_optimization():
    """示例：购物优化问题"""
    print("=== 购物优化问题 ===")
    
    # 问题：在预算限制下选择最优商品组合
    budget = 1000  # 预算1000元
    products = [
        (200, 300),  # 商品1: 价格200，满意度300
        (300, 400),  # 商品2: 价格300，满意度400
        (150, 200),  # 商品3: 价格150，满意度200
        (400, 500),  # 商品4: 价格400，满意度500
        (100, 150),  # 商品5: 价格100，满意度150
    ]
    
    print(f"预算: {budget}元")
    print("商品列表:")
    for i, (price, satisfaction) in enumerate(products):
        print(f"  商品{i+1}: 价格{price}元, 满意度{satisfaction}")
    
    # 使用01背包求解
    max_satisfaction = solve_01_knapsack(budget, products)
    print(f"最大满意度: {max_satisfaction}")
    
    # 获取具体选择方案
    solver = ZeroOneKnapsack()
    satisfaction, selected_items = solver.solve_with_path(budget, products)
    print(f"选择的商品: {[i+1 for i in selected_items]}")
    total_cost = sum(products[i][0] for i in selected_items)
    print(f"总花费: {total_cost}元")
    print()


def example_investment_portfolio():
    """示例：投资组合问题"""
    print("=== 投资组合问题 ===")
    
    # 问题：在资金限制下选择最优投资项目
    capital = 500000  # 资金50万
    investments = [
        (100000, 150000),  # 项目1: 投资10万，预期收益15万
        (200000, 250000),  # 项目2: 投资20万，预期收益25万
        (150000, 180000),  # 项目3: 投资15万，预期收益18万
        (80000, 100000),   # 项目4: 投资8万，预期收益10万
        (120000, 140000),  # 项目5: 投资12万，预期收益14万
    ]
    
    print(f"可用资金: {capital}元")
    print("投资项目:")
    for i, (cost, return_val) in enumerate(investments):
        roi = (return_val - cost) / cost * 100
        print(f"  项目{i+1}: 投资{cost}元, 预期收益{return_val}元, ROI={roi:.1f}%")
    
    # 使用01背包求解（每个项目只能投资一次）
    max_return = solve_01_knapsack(capital, investments)
    print(f"最大预期收益: {max_return}元")
    print(f"净收益: {max_return - capital}元")
    print()


def example_resource_allocation():
    """示例：资源分配问题"""
    print("=== 资源分配问题 ===")
    
    # 问题：在时间和人力限制下选择最优任务组合
    time_limit = 40  # 时间限制40小时
    manpower_limit = 20  # 人力限制20人
    tasks = [
        (10, 5, 1000),   # 任务1: 需要10小时，5人，价值1000
        (15, 8, 1500),   # 任务2: 需要15小时，8人，价值1500
        (8, 3, 800),     # 任务3: 需要8小时，3人，价值800
        (12, 6, 1200),   # 任务4: 需要12小时，6人，价值1200
        (6, 4, 600),     # 任务5: 需要6小时，4人，价值600
    ]
    
    print(f"时间限制: {time_limit}小时")
    print(f"人力限制: {manpower_limit}人")
    print("任务列表:")
    for i, (time, people, value) in enumerate(tasks):
        print(f"  任务{i+1}: 时间{time}小时, 人力{people}人, 价值{value}")
    
    # 使用二维背包求解
    max_value = solve_2d_knapsack(time_limit, manpower_limit, tasks)
    print(f"最大价值: {max_value}")
    print()


def example_equipment_selection():
    """示例：装备选择问题（分组背包）"""
    print("=== 装备选择问题 ===")
    
    # 问题：在预算限制下选择最优装备组合
    # 每个部位只能选择一件装备
    budget = 10000
    equipment_groups = [
        # 武器组
        [(2000, 500), (3000, 700), (1500, 400)],
        # 防具组
        [(1800, 450), (2500, 600), (1200, 300)],
        # 饰品组
        [(1000, 300), (1500, 400), (800, 250)],
        # 鞋子组
        [(1200, 350), (2000, 500), (900, 280)],
    ]
    
    print(f"预算: {budget}元")
    print("装备分组:")
    group_names = ["武器", "防具", "饰品", "鞋子"]
    for i, (group_name, group) in enumerate(zip(group_names, equipment_groups)):
        print(f"  {group_name}组:")
        for j, (cost, value) in enumerate(group):
            print(f"    装备{j+1}: 价格{cost}元, 属性{value}")
    
    # 使用分组背包求解
    max_value = solve_group_knapsack(budget, equipment_groups)
    print(f"最大属性值: {max_value}")
    print()


def example_inventory_management():
    """示例：库存管理问题（多重背包）"""
    print("=== 库存管理问题 ===")
    
    # 问题：在仓库容量限制下选择最优商品库存
    warehouse_capacity = 1000  # 仓库容量1000单位
    products = [
        (50, 100, 10),   # 商品1: 体积50，价值100，库存10个
        (30, 80, 15),    # 商品2: 体积30，价值80，库存15个
        (20, 60, 20),    # 商品3: 体积20，价值60，库存20个
        (40, 90, 8),     # 商品4: 体积40，价值90，库存8个
    ]
    
    print(f"仓库容量: {warehouse_capacity}单位")
    print("商品列表:")
    for i, (volume, value, stock) in enumerate(products):
        print(f"  商品{i+1}: 体积{volume}, 价值{value}, 库存{stock}个")
    
    # 使用多重背包求解
    max_value = solve_multiple_knapsack(warehouse_capacity, products)
    print(f"最大价值: {max_value}")
    print()


def main():
    """主函数：运行所有示例"""
    print("背包问题实际应用示例")
    print("=" * 50)
    
    example_shopping_optimization()
    example_investment_portfolio()
    example_resource_allocation()
    example_equipment_selection()
    example_inventory_management()
    
    print("所有示例运行完成！")


if __name__ == "__main__":
    main()
