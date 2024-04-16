import unittest

from tests.input.filetest import FileUnitTest
from tests.unfolding.unfoldingtest import UnfoldingUnitTest


def main():
    unittest.TextTestRunner(verbosity=3).run(unittest.TestSuite())

print('Unit test run')

if __name__ == '__main__':
    unittest.main()
