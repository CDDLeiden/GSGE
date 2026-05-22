import argparse
import sys
import subprocess

from GSGE import get_tests_dir

"""
CLI

Commands:
- run_test: Run all or specific test(s) from the tests directory.
"""

def main():
    parser = argparse.ArgumentParser(description='Package command-line interface')

    # Define subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Sub-command to execute')

    # Subparser for 'run_test' command
    test_parser = subparsers.add_parser('run_test', help='Run tests in the test directory')
    test_parser.add_argument('--file', type=str, help='Specific test file to run (e.g., test_example.py)')

    args = parser.parse_args()

    if args.command == 'run_test':
        run_tests(args.file)
    else:
        print(f'Error: Unknown command "{args.command}"')

def run_tests(filename=None):
    # Use GSGE path utilities to find tests directory
    test_dir = get_tests_dir()
    if test_dir is None:
        print("Error: tests directory not found.")
        print("  This command only works when running from a source checkout.")
        print("  Make sure you're in the GSGE repository with pip install -e .")
        sys.exit(1)

    if filename:
        test_path = test_dir / filename
        if test_path.is_file():
            print(f"Running specific test: {filename}")
            subprocess.run(['python', str(test_path)])
        else:
            print(f"Error: Test file '{filename}' not found in '{test_dir}'.")
    else:
        for file in test_dir.iterdir():
            if file.name.startswith('test_') and file.name.endswith('.py'):
                print(f"Found test file: {file.name}")
                try:
                    subprocess.run(['python', str(file)])
                    print(f"Test '{file.name}' executed successfully.")
                except Exception as e:
                    print(f"Error running test '{file.name}': {e}")

if __name__ == "__main__":
    main()