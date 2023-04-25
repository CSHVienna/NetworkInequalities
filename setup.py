# Copyright (c) 2023 NetInCSH.  All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# License.  (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

import os
import sys
from glob import glob

from setuptools import setup

with open("netin/__init__.py") as fid:
    for line in fid:
        if line.startswith("__version__"):
            version = line.strip().split()[-1][1:-1]
            break

if sys.version_info[:2] < (3, 8):
    error = (f"NetIn {version} requires Python 3.9 or later ({sys.version_info[:2]} detected). \n")
    sys.stderr.write(error + "\n")
    sys.exit(1)

name = "netin"
description = "Python package to study inequalities in social networks"
authors = {
    "Karimi": ("Fariba Karimi", "karimi@csh.ac.at"),
    "Espín-Noboa": ("Lisette Espín-Noboa", "espin@csh.ac.at"),
    "Bachmann": ("Jan Bachmann", "bachmann@csh.ac.at"),
}
maintainer = "NetIn Developers"
maintainer_email = "netin-dev@googlegroups.com"
url = "https://github.com/CSHVienna/NetworkInequalities"
platforms = ["Linux", "Mac OSX", "Windows", "Unix"]
keywords = [
    "Networks",
    "Inequalities",
    "Social Networks",
    "Ranking",
    "Inference"
    "Graph Theory",
    "Mathematics",
    "network",
    "undirected",
    "discrete mathematics",
    "math",
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Information Technology",
    "Intended Audience :: Science/Research",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.9",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics",
]

packages = [
    "netin",
    "netin.algorithms.sampling",
    "netin.generators",
    "netin.utils",
    "netin.stats",
    "netin.viz",
    "netin.generators.tests",
]

docdirbase = "share/doc/netin-%s" % version
# add basic documentation
data = [(docdirbase, glob("*.txt"))]
# add examples
for d in [
    ".",
    "advanced",
    "algorithms",
    "basic",
    "drawing",
    "undirected",
    "subclass",
]:
    dd = os.path.join(docdirbase, "examples", d)
    pp = os.path.join("examples", d)
    data.append((dd, glob(os.path.join(pp, "*.txt"))))
    data.append((dd, glob(os.path.join(pp, "*.py"))))
    data.append((dd, glob(os.path.join(pp, "*.bz2"))))
    data.append((dd, glob(os.path.join(pp, "*.gz"))))
    data.append((dd, glob(os.path.join(pp, "*.mbox"))))
    data.append((dd, glob(os.path.join(pp, "*.edgelist"))))
# add js force examples
dd = os.path.join(docdirbase, "examples", "javascript/force")
pp = os.path.join("examples", "javascript/force")
data.append((dd, glob(os.path.join(pp, "*"))))

# add the tests subpackage(s)
package_data = {
    "netin": ["tests/*.py"],
    "netin.generators": ["tests/*.py"],
}


def parse_requirements_file(filename):
    with open(filename) as fid:
        requires = [l.strip() for l in fid.readlines() if not l.startswith("#")]
    return requires


install_requires = parse_requirements_file("requirements/default.txt")

extras_require = {
    dep: parse_requirements_file("requirements/" + dep + ".txt")
    for dep in ["default", "test"]  # , "developer", "doc", "extra"]
}

with open("README.rst") as fh:
    long_description = fh.read()

if __name__ == "__main__":
    setup(
        name=name,
        version=version,
        maintainer=maintainer,
        maintainer_email=maintainer_email,
        author=authors["Espín-Noboa"][0],
        author_email=authors["Espín-Noboa"][1],
        description=description,
        keywords=keywords,
        long_description=long_description,
        long_description_content_type="text/x-rst",
        platforms=platforms,
        url=url,
        classifiers=classifiers,
        packages=packages,
        data_files=data,
        package_data=package_data,
        install_requires=install_requires,
        extras_require=extras_require,
        python_requires=">=3.9",
        zip_safe=False,
    )
