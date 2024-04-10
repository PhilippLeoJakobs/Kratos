import argparse
import os
import subprocess
import sys

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onlyFastSuite", action='store_true')
    args = parser.parse_args()

    command = [os.path.join("test", "KratosGeoMechanicsCoreTest")]
    if args.onlyFastSuite:
        command.append("--gtest_filter=KratosGeoMechanicsFastSuite.*")

    print(f"cwd = {os.getcwd()}")
    print(f"command = {command}")

    return subprocess.run(command).returncode

if __name__ == '__main__':
    sys.exit(run())
