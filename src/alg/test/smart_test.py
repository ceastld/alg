#!/usr/bin/env python3
"""
Smart testing framework for algorithm problems
Provides decorators and utilities for easy input/output testing
"""

import sys
import io
import subprocess
import tempfile
import os
from typing import List, Tuple, Callable, Any, Union
from functools import wraps
from contextlib import redirect_stdout


class TestCase:
    """Represents a single test case"""
    
    def __init__(self, input_data: str, expected_output: str, description: str = ""):
        self.input_data = input_data
        self.expected_output = expected_output
        self.description = description
    
    def __repr__(self):
        return f"TestCase(input={repr(self.input_data)}, expected={repr(self.expected_output)})"


class TestSuite:
    """Collection of test cases"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.test_cases: List[TestCase] = []
    
    def add_case(self, input_data: str, expected_output: str, description: str = ""):
        """Add a test case"""
        self.test_cases.append(TestCase(input_data, expected_output, description))
        return self
    
    def add_cases(self, cases: List[Tuple[str, str, str]]):
        """Add multiple test cases"""
        for case in cases:
            if len(case) == 2:
                self.add_case(case[0], case[1])
            elif len(case) == 3:
                self.add_case(case[0], case[1], case[2])
        return self
    
    def run_tests(self, program_file: str, verbose: bool = True) -> Tuple[int, int]:
        """Run all test cases against a program file"""
        passed = 0
        total = len(self.test_cases)
        
        if verbose:
            print(f"Running {total} test cases for {program_file}")
            print("=" * 60)
        
        for i, test_case in enumerate(self.test_cases, 1):
            success, actual_output, error = self._run_single_test(program_file, test_case)
            
            if success and actual_output == test_case.expected_output:
                passed += 1
                if verbose:
                    print(f"✓ Test {i}: PASSED")
                    if test_case.description:
                        print(f"  {test_case.description}")
            else:
                if verbose:
                    print(f"✗ Test {i}: FAILED")
                    if test_case.description:
                        print(f"  {test_case.description}")
                    print(f"  Input: {repr(test_case.input_data)}")
                    print(f"  Expected: {repr(test_case.expected_output)}")
                    print(f"  Got: {repr(actual_output)}")
                    if error:
                        print(f"  Error: {error}")
                    print()
        
        if verbose:
            print("=" * 60)
            print(f"Results: {passed}/{total} tests passed")
            if passed == total:
                print("🎉 All tests passed!")
            else:
                print(f"❌ {total - passed} tests failed")
        
        return passed, total
    
    def _run_single_test(self, program_file: str, test_case: TestCase) -> Tuple[bool, str, str]:
        """Run a single test case against a program file"""
        try:
            result = subprocess.run(
                [sys.executable, program_file],
                input=test_case.input_data,
                text=True,
                capture_output=True,
                timeout=10
            )
            
            return True, result.stdout.strip(), result.stderr.strip()
            
        except subprocess.TimeoutExpired:
            return False, "", "Timeout (10 seconds)"
        except Exception as e:
            return False, "", str(e)


class SmartTester:
    """Smart testing framework with decorators"""
    
    def __init__(self):
        self.test_suites: dict = {}
    
    def test_cases(self, *cases: Union[Tuple[str, str], Tuple[str, str, str]]):
        """Decorator to define test cases for a function"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            # Store test cases in the function
            wrapper._test_cases = []
            for case in cases:
                if len(case) == 2:
                    wrapper._test_cases.append(TestCase(case[0], case[1]))
                elif len(case) == 3:
                    wrapper._test_cases.append(TestCase(case[0], case[1], case[2]))
            
            return wrapper
        return decorator
    
    def run_function_tests(self, func: Callable, verbose: bool = True) -> Tuple[int, int]:
        """Run test cases for a decorated function"""
        if not hasattr(func, '_test_cases'):
            print("No test cases found for this function")
            return 0, 0
        
        passed = 0
        total = len(func._test_cases)
        
        if verbose:
            print(f"Running {total} test cases for {func.__name__}")
            print("=" * 60)
        
        for i, test_case in enumerate(func._test_cases, 1):
            try:
                # Capture stdout
                old_stdout = sys.stdout
                sys.stdout = captured_output = io.StringIO()
                
                # Set stdin
                old_stdin = sys.stdin
                sys.stdin = io.StringIO(test_case.input_data)
                
                # Run the function
                func()
                
                # Get output
                actual_output = captured_output.getvalue().strip()
                
                # Restore stdout and stdin
                sys.stdout = old_stdout
                sys.stdin = old_stdin
                
                if actual_output == test_case.expected_output:
                    passed += 1
                    if verbose:
                        print(f"✓ Test {i}: PASSED")
                        if test_case.description:
                            print(f"  {test_case.description}")
                else:
                    if verbose:
                        print(f"✗ Test {i}: FAILED")
                        if test_case.description:
                            print(f"  {test_case.description}")
                        print(f"  Input: {repr(test_case.input_data)}")
                        print(f"  Expected: {repr(test_case.expected_output)}")
                        print(f"  Got: {repr(actual_output)}")
                        print()
                        
            except Exception as e:
                # Restore stdout and stdin
                sys.stdout = old_stdout
                sys.stdin = old_stdin
                
                if verbose:
                    print(f"✗ Test {i}: ERROR - {str(e)}")
                    print(f"  Input: {repr(test_case.input_data)}")
                    print()
        
        if verbose:
            print("=" * 60)
            print(f"Results: {passed}/{total} tests passed")
            if passed == total:
                print("🎉 All tests passed!")
            else:
                print(f"❌ {total - passed} tests failed")
        
        return passed, total
    
    def _run_single_test(self, program_file: str, test_case: TestCase) -> Tuple[bool, str, str]:
        """Run a single test case against a program file"""
        try:
            result = subprocess.run(
                [sys.executable, program_file],
                input=test_case.input_data,
                text=True,
                capture_output=True,
                timeout=10
            )
            
            return True, result.stdout.strip(), result.stderr.strip()
            
        except subprocess.TimeoutExpired:
            return False, "", "Timeout (10 seconds)"
        except Exception as e:
            return False, "", str(e)


# Global instance
tester = SmartTester()

# Convenience functions
def test_cases(*cases):
    """Decorator to define test cases"""
    return tester.test_cases(*cases)

def run_tests(func: Callable, verbose: bool = True) -> Tuple[int, int]:
    """Run tests for a decorated function"""
    return tester.run_function_tests(func, verbose)

def create_test_suite(name: str = "") -> TestSuite:
    """Create a new test suite"""
    return TestSuite(name)

def run_program_tests(program_file: str, test_suite: TestSuite, verbose: bool = True) -> Tuple[int, int]:
    """Run test suite against a program file"""
    return test_suite.run_tests(program_file, verbose)
