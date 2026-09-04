"""
Every module in this folder is one vendor plugin (or, for generic.py, a
pair of fallback plugins). Importing THIS package is what actually
registers them -- each module uses the @register_plugin decorator from
data_import/plugin.py, which only runs when the module is imported.

To add support for a new instrument later: create a new file here (e.g.
zeiss_xradia.py), define a class inheriting from ImportPlugin, decorate
it with @register_plugin, and add an import line for it below. That's
the entire process -- nothing outside this folder needs to change.
"""
from data_import.vendors import generic  # noqa: F401
from data_import.vendors import synthetic  # noqa: F401
from data_import.vendors import hypothetical_synchrotron  # noqa: F401
from data_import.vendors import generic_parallel_synchrotron  # noqa: F401
