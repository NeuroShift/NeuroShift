# Tests
the test directory contains the pytest tests, that are automatically checked by circleCI.

## Adding A Test

### creating a test directory
the directory names should be mirroring the directories of the project
if adding a new directory, add it in the `pyproject.toml`.
it should be appended to the list called testpaths:
```toml
testpaths = [
    ...
    "path/to/add",
]
```

### creating a test file
to add a test, depending on for which file the test is for, you will need to create a new test file in this directory.
The file should follow the following regular expressions:
  - `*_test.py`
  - `test_*.py`
(please use the latter to keep a better style)

in the file you should import the equivalent class in the project as:
```python
from src.<relative_path> import ClassToTest
```

### creating a test
a test is a standard function, it will be recognized as such by pytest, whether it is a class method or a static function.
the only important part for a test to be recognized is for the name of its function to contain the word test.
(here again please try to keep the test names conform to `test_*`, where ideally the rest correspond to the tested method)
