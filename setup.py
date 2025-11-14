import json
import sys

from pathlib import Path

from setuptools import find_packages, setup

with open("python_versions.json", "r") as f:
    supported_python_versions = json.load(f)

python_versions = [tuple(map(int, v.split('.'))) for v in supported_python_versions]
min_version = min(python_versions)
max_version = max(python_versions)
current_version = sys.version_info[:2]

if not (min_version <= current_version <= max_version):
    # Python 3.5 does not support f-strings
    py_version = ".".join([str(v) for v in sys.version_info[:3]])
    min_version_str = ".".join(map(str, min_version))
    max_version_str = ".".join(map(str, max_version))
    error = (
        "\n----------------------------------------\n"
        "Error: vivarium_helpers runs under python {min_version}-{max_version}.\n"
        "You are running python {py_version}".format(
            min_version=min_version_str, max_version=max_version_str, py_version=py_version
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

    install_requirements = [
        "vivarium_build_utils>=2.0.1,<3.0.0",
        "pandas",
        "db_queries",
        "numpy",
        "scipy",
    ]

    setup_requires = ["setuptools_scm"]
    
    test_requirements = [
	    "pytest",
    ]

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
            "test": test_requirements,
        },

        zip_safe=False,

        use_scm_version={
            "write_to": "src/vivarium_helpers/_version.py",
            "write_to_template": '__version__ = "{version}"\n',
            "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
        },
        setup_requires=setup_requires,
    )
