#!/usr/bin/env python

import argparse
import subprocess
import os.path


def build(args):
    options = ["cmake"]

    if args.ninja:
        options.append("-G")
        options.append("Ninja")

    if args.docs_only:
        options.append("-DATHENA_DOCS_ONLY=ON")

    if args.disable_tests:
        options.append("-DATENA_DISABLE_TESTS=ON")

    if args.enable_coverage:
        options.append("-DENABLE_COVERAGE=ON")

    if args.compile_commands:
        options.append("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")

    if "" != args.install_dir:
        options.append("-DCMAKE_INSTALL_PREFIX=" + args.install_dir)

    if not os.path.exists(args.destination):
        os.mkdir(args.destination)

    my_env = os.environ.copy()

    if args.use_ldd:
        if "LDFLAGS" in my_env:
            my_env["LDFLAGS"] += " -fuse-ld=lld"
        else:
            my_env["LDFLAGS"] = " -fuse-ld=lld"

    options.append("-H" + args.source)
    options.append("-B" + args.destination)

    subprocess.call(options, env=my_env)
    if not args.no_build:
        subprocess.call(["cmake", "--build", args.destination], env=my_env)


def main():
    parser = argparse.ArgumentParser(description='Athena build tool')
    parser.add_argument("destination", type=str, help="CMake build directory")
    parser.add_argument("source", type=str, help="Athena source location")
    parser.add_argument("--ninja", action='store_true', help="Use Ninja build system")
    parser.add_argument("--docs-only", action='store_true', help="Configure docs target only")
    parser.add_argument("--disable-tests", action='store_true', help="Disable tests")
    parser.add_argument("--enable-coverage", action='store_true', help="Create code coverage report")
    parser.add_argument("--build-type", type=str, default='Release', help="Build type")
    parser.add_argument("--compile-commands", action='store_true', help="Export compile commands")
    parser.add_argument("--use-ldd", action='store_true', help="Use LLVM ldd linker")
    parser.add_argument("--no-build", action='store_true', help="Do not perform actual build")
    parser.add_argument("--install-dir", type=str, default="", help="Installation path")

    args = parser.parse_args()

    build(args)


if __name__ == '__main__':
    main()
