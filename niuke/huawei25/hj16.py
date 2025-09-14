def solve_shopping_problem(n: int, m: int, items: list[tuple[int, int, int]]) -> int:
    """
    Solve the shopping problem with main items and attachments.
    
    Args:
        n: budget (money available)
        m: total number of items
        items: list of (price, importance, parent_id) tuples
               parent_id = 0 means main item, otherwise attachment of item parent_id
    
    Returns:
        Maximum satisfaction value
    """
    # Group items by main items
    main_items = {}  # main_item_id -> (price, importance)
    attachments = {}  # main_item_id -> list of (price, importance)
    
    # First pass: identify all main items
    for i, (price, importance, parent_id) in enumerate(items, 1):
        if parent_id == 0:  # Main item
            main_items[i] = (price, importance)
            attachments[i] = []
    
    # Second pass: add attachments to their main items
    for i, (price, importance, parent_id) in enumerate(items, 1):
        if parent_id != 0:  # Attachment
            if parent_id in attachments:
                attachments[parent_id].append((price, importance))
    
    
    # DP: dp[i] = maximum satisfaction with budget i
    dp = [0] * (n + 1)
    
    # Process each main item group
    for main_id, (main_price, main_importance) in main_items.items():
        # Get attachments for this main item
        main_attachments = attachments.get(main_id, [])
        
        # Try all combinations of main item + attachments
        combinations = []
        
        # Option 1: Don't buy this main item
        combinations.append((0, 0))
        
        # Option 2: Buy only main item
        combinations.append((main_price, main_price * main_importance))
        
        # Option 3: Buy main item + each attachment
        for att_price, att_importance in main_attachments:
            total_price = main_price + att_price
            total_value = main_price * main_importance + att_price * att_importance
            combinations.append((total_price, total_value))
        
        # Option 4: Buy main item + both attachments (if there are 2)
        if len(main_attachments) == 2:
            att1_price, att1_importance = main_attachments[0]
            att2_price, att2_importance = main_attachments[1]
            total_price = main_price + att1_price + att2_price
            total_value = (main_price * main_importance + 
                          att1_price * att1_importance + 
                          att2_price * att2_importance)
            combinations.append((total_price, total_value))
        
        # Update DP using these combinations
        # Process in reverse order to avoid using updated values
        for budget in range(n, -1, -1):
            for price, value in combinations:
                if budget >= price:
                    dp[budget] = max(dp[budget], dp[budget - price] + value)
    
    return dp[n]


def main():
    # Read input
    n, m = map(int, input().split())
    items = []
    for _ in range(m):
        v, w, q = map(int, input().split())
        items.append((v, w, q))
    
    # Solve and output
    result = solve_shopping_problem(n, m, items)
    print(result)


if __name__ == "__main__":
    main()
