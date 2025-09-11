# 智能测试框架使用说明

## 简介
`smart_test.py` 提供了一个优雅的测试框架，使用装饰器来定义测试用例，支持函数测试和程序文件测试。

## 核心功能

### 1. 装饰器测试（推荐）
直接在函数上使用 `@test_cases` 装饰器定义测试用例：

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
测试独立的程序文件：

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

## 测试用例格式

每个测试用例包含：
- `input_data`: 输入数据（字符串，支持多行）
- `expected_output`: 期望输出（字符串）
- `description`: 测试描述（可选）

## 使用示例

### 简单示例
```python
@test_cases(
    ("5", "3", "Simple test"),
    ("10", "4", "Another test")
)
def my_solution():
    n = int(input())
    result = n // 2 + 1
    print(result)

run_tests(my_solution)
```

### 多行输入示例
```python
@test_cases(
    ("3\n1 2 3\n4 5 6", "21", "Matrix sum"),
    ("2\n10 20", "30", "Simple sum")
)
def matrix_solution():
    n = int(input())
    total = 0
    for _ in range(n):
        row = list(map(int, input().split()))
        total += sum(row)
    print(total)

run_tests(matrix_solution)
```

## 特性

- ✅ **装饰器语法**: 直观的测试用例定义
- ✅ **自动输入输出重定向**: 无需手动处理
- ✅ **智能结果比较**: 支持数字和字符串比较
- ✅ **详细错误报告**: 显示输入、期望输出、实际输出
- ✅ **超时保护**: 10秒超时防止程序卡死
- ✅ **支持多行输入**: 自动处理复杂的输入格式

## 注意事项

1. 函数必须使用 `input()` 和 `print()` 进行输入输出
2. 测试用例的输入数据使用 `\n` 表示换行
3. 期望输出必须是字符串格式
4. 程序运行超时时间为10秒

## 快速开始

1. 导入测试框架：
```python
from src.alg.test import test_cases, run_tests
```

2. 定义测试函数：
```python
@test_cases(
    ("input1", "output1", "description1"),
    ("input2", "output2", "description2")
)
def your_solution():
    # 你的解决方案代码
    pass
```

3. 运行测试：
```python
run_tests(your_solution)
```

就这么简单！🎉
