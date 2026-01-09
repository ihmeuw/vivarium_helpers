import json
import sys
from pathlib import Path

from packaging.version import parse
from setuptools import find_packages, setup

with open("python_versions.json", "r") as f:
    supported_python_versions = json.load(f)

python_versions = [parse(v) for v in supported_python_versions]
min_version = min(python_versions)
max_version = max(python_versions)

if not (
    min_version <= parse(".".join([str(v) for v in sys.version_info[:2]])) <= max_version
):
    # Python 3.5 does not support f-strings
    py_version = ".".join([str(v) for v in sys.version_info[:3]])
    error = (
        "\n----------------------------------------\n"
        "Error: Vivarium runs under python {min_version}-{max_version}.\n"
        "You are running python {py_version}".format(
            min_version=min_version, max_version=max_version, py_version=py_version
        )
    )
    print(error, file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    src_dir = base_dir / "src"

    about = {}
    with (src_dir / "vivarium_helpers" / "__about__.py").open() as f:
        exec(f.read(), about)

    with (base_dir / "README.rst").open() as f:
        long_description = f.read()

    setup_requires = ["setuptools_scm"]

    install_requirements = [
        "vivarium_dependencies[numpy,pandas,db_queries,loguru,scipy]",
        "vivarium_build_utils>=2.1.1,<3.0.0",
    ]
    test_requirements = ["vivarium_dependencies[pytest]"]
    lint_requirements = ["vivarium_dependencies[lint]"]
    doc_requirements = ["vivarium_dependencies[sphinx]"]

    setup(
        name=about["__title__"],
        description=about["__summary__"],
        long_description=long_description,
        license=about["__license__"],
        url=about["__uri__"],
        author=about["__author__"],
        author_email=about["__email__"],

        package_dir={"": "src"},
        packages=find_packages(where="src"),
        include_package_data=True,

        install_requires=install_requirements,
        test_require=test_requirements,
	    extras_require={
            "docs": doc_requirements,
            "test": test_requirements,
            "dev": doc_requirements + test_requirements + lint_requirements,
        },

        zip_safe=False,

        use_scm_version={
            "write_to": "src/vivarium_helpers/_version.py",
            "write_to_template": '__version__ = "{version}"\n',
            "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
        },
        setup_requires=setup_requires,
    )
