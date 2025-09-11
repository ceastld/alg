
"""
MOE Top-k Routing Problem Solution

Given n experts distributed across m NPU cards, select k experts with routing
limited to at most p NPU cards to minimize cross-card communication.
"""

from typing import List, Tuple, Optional


def solve_moe_topk(n: int, m: int, p: int, k: int, probabilities: List[float]) -> Optional[List[int]]:
    """
    Solve the MOE Top-k routing problem.
    
    Args:
        n: Number of experts (0 to n-1)
        m: Number of NPU cards
        p: Maximum number of NPU cards to use
        k: Number of experts to select
        probabilities: List of probabilities for each expert
        
    Returns:
        List of selected expert indices in ascending order, or None if error
    """
    # Check if n is divisible by m
    if n % m != 0:
        return None
    
    # Check if p > m
    if p > m:
        return None
    
    # Calculate group size
    g = n // m
    
    # Check if we have enough experts to select k
    if p * g < k:
        return None
    
    # Step 1: Find representative (max probability) for each group
    group_reps: List[Tuple[float, int, int]] = []  # (max_prob, expert_idx, group_idx)
    
    for group_idx in range(m):
        start_idx = group_idx * g
        end_idx = start_idx + g
        
        # Find max probability in this group
        max_prob = -1
        max_expert_idx = -1
        
        for expert_idx in range(start_idx, end_idx):
            if probabilities[expert_idx] > max_prob:
                max_prob = probabilities[expert_idx]
                max_expert_idx = expert_idx
        
        group_reps.append((max_prob, max_expert_idx, group_idx))
    
    # Step 2: Sort groups by representative probability (descending, then by group index)
    group_reps.sort(key=lambda x: (-x[0], x[2]))
    
    # Step 3: Select top p groups
    selected_groups = group_reps[:p]
    
    # Step 4: Collect all experts from selected groups
    candidate_experts: List[Tuple[float, int]] = []  # (prob, expert_idx)
    
    for _, _, group_idx in selected_groups:
        start_idx = group_idx * g
        end_idx = start_idx + g
        
        for expert_idx in range(start_idx, end_idx):
            candidate_experts.append((probabilities[expert_idx], expert_idx))
    
    # Step 5: Sort candidates by probability (descending), then by expert index (ascending)
    candidate_experts.sort(key=lambda x: (-x[0], x[1]))
    
    # Step 6: Select top k experts
    selected_experts = [expert_idx for _, expert_idx in candidate_experts[:k]]
    
    # Step 7: Sort final result in ascending order
    selected_experts.sort()
    
    return selected_experts


def main():
    """Main function to handle input and output."""
    try:
        # Read input
        line1 = input().strip().split()
        n, m, p, k = map(int, line1)
        
        line2 = input().strip().split()
        probabilities = list(map(float, line2))
        
        # Solve the problem
        result = solve_moe_topk(n, m, p, k, probabilities)
        
        # Output result
        if result is None:
            print("error")
        else:
            print(" ".join(map(str, result)))
            
    except Exception:
        print("error")


if __name__ == "__main__":
    main()
