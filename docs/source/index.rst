.. NetIn documentation master file, created by
   sphinx-quickstart on Wed Apr 19 11:08:02 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Network Inequality
=================================

.. automodule:: netin

Existing models are defined by the following :class:`.Model` class implementations.


.. toctree::
   :maxdepth: 3

   models/index

Each model can be simulated by running the :meth:`.Model.simulate` method, returning the simulated network as a :class:`.Graph` instance.

.. toctree::
   :maxdepth: 2

   graphs

:class:`Graphs <.Graph>` typically contain node attributes in the form of :class:`NodeVectors <NodeVector>`.
For most models, these vectors contain the group assignments of nodes to a minority or majority class as :class:`.BinaryClassNodeVector`.

.. toctree::
   :maxdepth: 2

   node_vectors/index

Besides the implementation of models, `NetIn` offers functionality to analyze existing (or simulated) networks in terms of various inequalities they exhibit.

.. toctree::
   :maxdepth: 1

   algorithms/index
   statistics

If you want to create your own custom models or inject your own code during simulation of existing models, `NetIn` is highly extendible.
It also provides utility functions to visualize networks and inequality results.

.. toctree::
   :maxdepth: 1

   extensions/index
   visualizations
   utilities

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
