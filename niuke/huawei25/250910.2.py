import math


def mean(nums):
    """Calculate mean of a list of numbers"""
    if not nums:
        return 0.0
    return sum(nums) / len(nums)


def std(nums):
    """Calculate standard deviation of a list of numbers"""
    if len(nums) <= 1:
        return 0.0
    
    mean_val = mean(nums)
    variance = sum((x - mean_val) ** 2 for x in nums) / (len(nums) - 1)
    return math.sqrt(variance)


def min_val(nums):
    """Find minimum value in a list"""
    if not nums:
        return 0
    return min(nums)


def max_val(nums):
    """Find maximum value in a list"""
    if not nums:
        return 0
    return max(nums)


def slope(nums):
    """Calculate slope using least squares method"""
    if len(nums) <= 1:
        return 0.0
    
    # Check if all values are the same
    if len(set(nums)) == 1:
        return 0.0
    
    try:
        n = len(nums)
        x = list(range(n))  # [0, 1, 2, ..., n-1]
        y = nums
        
        # Calculate means
        x_mean = mean(x)
        y_mean = mean(y)
        
        # Calculate slope using least squares formula
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    except:
        return 0.0


def process_array(nums):
    """Process array and return [mean, std, min, max, slope]"""
    mean_val = mean(nums)
    std_val = std(nums)
    min_val_result = min_val(nums)
    max_val_result = max_val(nums)
    slope_val = slope(nums)
    
    return [mean_val, std_val, min_val_result, max_val_result, slope_val]


def format_number(num):
    """Format number to avoid unnecessary decimal points and limit to 3 decimal places"""
    if isinstance(num, float):
        if num.is_integer():
            return int(num)
        else:
            # Round to 3 decimal places and remove trailing zeros
            rounded = round(num, 3)
            if rounded == int(rounded):
                return int(rounded)
            return rounded
    return num


def main():
    nums, windows = eval(input())
    max_window = max(windows)
    n = len(nums) - max_window + 1
    if n <= 0:
        print([])
        return
    for i in range(n):
        res = []
        for w in windows:
            processed = process_array(nums[i+max_window-w : i + max_window])
            # Format each number in the result
            formatted = [format_number(x) for x in processed]
            res.extend(formatted)
        print(res)


if __name__ == "__main__":
    main()
