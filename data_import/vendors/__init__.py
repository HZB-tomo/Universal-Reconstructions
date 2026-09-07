"""
Each module in this folder is one vendor plugin (generic.py has a pair
of fallback plugins). Importing this package registers them via the
@register_plugin decorator in data_import/plugin.py.

To add a new vendor: create a file here, define a class inheriting
from ImportPlugin, decorate it with @register_plugin, and add an import
line below.
"""
from data_import.vendors import generic  # noqa: F401
from data_import.vendors import synthetic  # noqa: F401
from data_import.vendors import hypothetical_synchrotron  # noqa: F401
from data_import.vendors import generic_parallel_synchrotron  # noqa: F401
