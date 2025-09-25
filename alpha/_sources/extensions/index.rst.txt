Extending NetIn
===============

The package provides multiple interfaces to extend its functionalities.

Several classes implement the :class:`.HasEvents` interface which can be used to inject your own code at runtime.

.. toctree::
   :maxdepth: 2

   events

Alternatively, you can extend the existing class structure to facilitate the existing simulation code, changing only specific modeling details.
:class:`.Filter` and :class:`.LinkFormationMechanism` provide abstract classes that describe how target nodes are chosen during simulation.

.. toctree::
   :maxdepth: 2

   filters
   mechanisms

Both custom and existing implementations (e.g., :class:`.Homophily`) can be used in custom models to reuse the existing simulation logic.
In that case, only some of the simulation methods have to be reimplemented.
For this purpose, several abstract base classes define varying levels of modelling abstractions.

.. toctree::
   :maxdepth: 2

   models
