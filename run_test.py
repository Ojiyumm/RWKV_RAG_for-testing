import sys
import importlib


if __name__ == '__main__':
    argv = sys.argv
    print(argv)
    for name in argv[1:]:
        module = 'src.test.' + name
        print(f'testing {module}.py')
        importlib.import_module(module)
