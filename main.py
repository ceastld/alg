from alg import __version__
from alg.utils import add


def main() -> None:
    """Main entry point for the algorithm project."""
    print(f"Hello from alg! Version: {__version__}")
    
    # Test the add function
    result = add(5, 3)
    print(f"5 + 3 = {result}")

if __name__ == "__main__":
    main()


