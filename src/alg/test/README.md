# 智能测试框架

## 目录结构
```
src/alg/test/
├── __init__.py          # 包初始化文件，导出主要接口
├── smart_test.py        # 核心测试框架实现
├── example.py          # 使用示例
└── README.md           # 本说明文档
```

## 快速开始

### 1. 基本使用
```python
from src.alg.test import test_cases, run_tests

@test_cases(
    ("4\n2 5 6 13", "2", "Example 1"),
    ("4\n1 2 2 2", "1", "Example 2"),
    ("2\n3 6", "0", "Example 3")
)
def hj28_solution():
    from niuke.hj28 import solve_prime_partners
    n = int(input())
    nums = list(map(int, input().split()))
    result = solve_prime_partners(nums)
    print(result)

# 运行测试
run_tests(hj28_solution)
```

### 2. 程序文件测试
```python
from src.alg.test import create_test_suite, run_program_tests

suite = create_test_suite("HJ28 Tests")
suite.add_cases([
    ("4\n2 5 6 13", "2", "Example 1"),
    ("4\n1 2 2 2", "1", "Example 2"),
    ("2\n3 6", "0", "Example 3")
])

run_program_tests("niuke/hj28.py", suite)
```

## 主要特性

- ✅ **装饰器语法**: 使用 `@test_cases()` 直观定义测试用例
- ✅ **自动输入输出重定向**: 无需手动处理输入输出
- ✅ **智能结果比较**: 支持数字和字符串比较
- ✅ **详细错误报告**: 显示输入、期望输出、实际输出
- ✅ **超时保护**: 10秒超时防止程序卡死
- ✅ **支持多行输入**: 自动处理复杂的输入格式

## 测试用例格式

每个测试用例包含：
- `input_data`: 输入数据（字符串，支持多行，使用 `\n` 表示换行）
- `expected_output`: 期望输出（字符串）
- `description`: 测试描述（可选）

## 运行示例

查看 `example.py` 文件获取完整的使用示例。

## 注意事项

1. 函数必须使用 `input()` 和 `print()` 进行输入输出
2. 测试用例的输入数据使用 `\n` 表示换行
3. 期望输出必须是字符串格式
4. 程序运行超时时间为10秒
