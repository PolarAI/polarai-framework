import argparse
import os.path
import re
import subprocess
import sys
from os import walk


def main():
    parser = argparse.ArgumentParser(description="Athena Test Runner")
    parser.add_argument('test_dir', type=str)
    parser.add_argument('--bin-path', type=str)
    parser.add_argument('--llvm-path', type=str)
    args = parser.parse_args()

    commands = {
        'mlir_opt': os.path.join(args.llvm_path, 'bin', 'mlir-opt'),
        'llvm_opt': os.path.join(args.llvm_path, 'bin', 'opt'),
        'chaoscc': os.path.join(args.bin_path, 'utils', 'chaos', 'tools', 'Driver', 'chaoscc'),
        'check': os.path.join(args.bin_path, 'utils', 'test_runner', 'athena_check'),
        'test_dir': args.test_dir
    }

    test_files = []
    for (dirpath, dirnames, filenames) in walk(args.test_dir):
        for fn in filenames:
            if fn.endswith(".cpp"):
                test_files.append(dirpath + os.path.sep + fn)

    is_failed = False
    for path in test_files:
        with open(path) as f:
            test = f.read()
            runs = re.findall(r"(?<=RUN: ).*", test)
            for run in runs:
                for k, v in commands.items():
                    run = run.replace('%' + k, v)
                run = run.replace('%t', path)
                print(run)
                ret_code = subprocess.call(["bash", "-c", run])
                if ret_code != 0:
                    is_failed = True
    if is_failed:
        print("Tests failure")
        sys.exit(-1)


if __name__ == '__main__':
    main()
