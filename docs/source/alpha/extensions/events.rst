Event Handling
==============

Several other classes implement the :class:`.HasEvents` interface.
In that case, they should provide a :attr:`.HasEvents.EVENTS` list that contains the :class:`Events <.Event>` that they implement.

If you wish to inject your own code, check :meth:`.HasEvents.register_event_handler`.
As an example, consider the (shortened) implementation of :class:`.SimulationTimer`:

.. code:: python

    class SimulationTimer:
        _start_time: float # Starting time

        time: Optional[float] = None # Final measure

        def __init__(self, model: Model) -> None:
            # Start timer with simulation
            model.register_event_handler(
                Event.SIMULATION_START, self._start_timer)

            # End timer with simulation
            model.register_event_handler(
                Event.SIMULATION_END, self._end_timer)

        def _start_timer(self):
            # Sets the starting time
            self._start_time = time.time()

        def _end_timer(self):
            # Computes the final simulation run
            self.time = time.time() - self._start_time

This code registers two functions to measure the simulation.
When the simulation starts, the :meth:`_start_timer` method is called to store the starting time.
Once the simulation ends, :meth:`_end_timer` takes again the time and subtracts the starting time.
After the simulation, the :attr:`time` attribute contains the runtime of the simulation.

.. autoclass:: netin.utils.Event
    :members:
    :undoc-members:
.. autoclass:: netin.utils.HasEvents
    :members: