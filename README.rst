NetIn
========
NetIn is a python package for network inference.
It is based on the NetworkX package and provides a set of methods to study network inequalities.
The package is currently under development and will be updated regularly.

.. image:: https://github.com/CSHVienna/NetworkInequalities/actions/workflows/python-app.yml/badge.svg
  :target: https://github.com/CSHVienna/NetworkInequalities/actions/workflows/python-app.yml

.. image:: https://img.shields.io/badge/python-3.9-blue.svg
  :target: https://www.python.org/downloads/release/python-3916/

.. image:: https://img.shields.io/badge/NetworkX-3.1-blue.svg
    :target: https://networkx.org/

.. image:: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
    :target: https://creativecommons.org/licenses/by-nc-sa/4.0/



- **Website:** https://www.networkinequality.com
- **Source:** https://github.com/CSHVienna/NetworkInequalities
- **Bug reports:** https://github.com/CSHVienna/NetworkInequalities/issues
- **GitHub Discussions:** https://github.com/CSHVienna/NetworkInequalities/discussions

Simple examples
---------------

Create an undirected network with preferential attachment and homophily.

.. code:: pycon

    >>> from netin as PAH
    >>> G = PAH(n=200, k=2, f_m=0.2, h_MM=0.1, h_mm=0.9, seed=42)
    >>> G.generate()


Create a directed network with preferential attachment and homophily.

.. code:: pycon

    >>> from netin import DPAH
    >>> G = DPAH(n=200, f_m=0.2, d=0.02, h_MM=0.1, h_mm=0.6, plo_M=2.0, plo_m=2.0, seed=42)
    >>> G.generate()

Install
-------

Install the latest version of NetIn::

    $ pip install netin


Install from source::

        $ git clone
        $ cd NetworkInequalities
        $ pip install -e .


Bugs
----

Please report any bugs that you find `here <https://github.com/CSHVienna/NetworkInequalities/issues>`_.
Or, even better, fork the repository on `GitHub <https://github.com/CSHVienna/NetworkInequalities>`_
and create a pull request (PR). We welcome all changes, big or small, and we
will help you make the PR if you are new to `git`.

License
-------

Released under Creative Commons by-nc-sa 4.0 (see `LICENSE`)::

   Copyright (C) 2023-2024 NetIn Developers
   Fariba Karimi <karimi@csh.ac.at>
   Lisette Espin-Noboa <espin@csh.ac.at>
   Jan Bachmann <bachmann@csh.ac.at>

