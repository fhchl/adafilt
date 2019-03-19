## Development

Run tests in base directory with

    pytest

Always run tests before pushing code.

Install for development with `pip install -e .` .

Comments should comply with the [Numpy/Scipy documentation style][1]. An
example can also be found [here][2]. Code should comply to the [pep8 coding style][3]. You can check if the code complies by executing

    pycodestyle
    pydocstyle

Git commit messages follow [Angular style][4]. Uses [python-semantic-release][5] for versioning on travis-ci for pushs to master branch.

Create documentation with

    pdoc --html --overwrite --html-dir build response.py

[1]: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
[2]: http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
[3]: https://www.python.org/dev/peps/pep-0008/
[4]: https://github.com/angular/angular.js/blob/master/DEVELOPERS.md#-git-commit-guidelines
[5]: https://github.com/relekang/python-semantic-release
