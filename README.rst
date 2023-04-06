NetIn
========
NetIn is a python package for network inference.
It is based on the NetworkX package and provides a set of methods to study network inequalities.
The package is currently under development and will be updated regularly.

.. image:: https://github.com/CSHVienna/NetworkInequalities/netin/workflows/test/badge.svg?branch=main
  :target: https://github.com/CSHVienna/NetworkInequalities/netin/actions?query=workflow%3A%22test%22

- **Website (including documentation):** https://www.networkinequality.com
- **Source:** https://github.com/CSHVienna/NetworkInequalities
- **Bug reports:** https://github.com/CSHVienna/NetworkInequalities/issues
- **GitHub Discussions:** https://github.com/CSHVienna/NetworkInequalities/discussions

Simple example
--------------

Find the shortest path between two nodes in an undirected graph:

.. code:: pycon

    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> G.add_edge("A", "B", weight=4)
    >>> G.add_edge("B", "D", weight=2)
    >>> G.add_edge("A", "C", weight=3)
    >>> G.add_edge("C", "D", weight=4)
    >>> nx.shortest_path(G, "A", "D", weight="weight")
    ['A', 'B', 'D']

Install
-------

Install the latest version of NetIn::

    $ pip install netin

Install with all optional dependencies::

    $ pip install netin[all]

For additional details, please see `INSTALL.rst`.

Bugs
----

Please report any bugs that you find `here <https://github.com/CSHVienna/NetworkInequalities/issues>`_.
Or, even better, fork the repository on `GitHub <https://github.com/CSHVienna/NetworkInequalities>`_
and create a pull request (PR). We welcome all changes, big or small, and we
will help you make the PR if you are new to `git` (just ask on the issue and/or
see `CONTRIBUTING.rst`).

License
-------

Released under Creative Commons by-nc-sa (see `LICENSE`)::

   Copyright (C) 2023-2024 NetIn Developers
   Fariba Karimi <karimi@csh.ac.at>
   Lisette Espin-Noboa <espin@csh.ac.at>
   Jan Bachmann <bachmann@csh.ac.at>
