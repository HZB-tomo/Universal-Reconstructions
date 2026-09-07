"""
data_import/ turns a file on disk into a ProjectObject (see
core/project.py).

Layout:
    path_map.py   PathMap: where things are, once located
    signature.py  Signature/Hdf5Signature/TiffSignature: how a plugin
                  declares "this is my kind of file" (a deterministic
                  checklist, not a confidence score)
    plugin.py     ImportPlugin: the contract every vendor plugin
                  implements, plus the plugin registry
    backend.py    FormatBackend/Hdf5Backend/TiffBackend: physically
                  reading hdf5/tiff, independent of vendor
    importer.py   DataImporter: detection -> plugin -> backend -> ProjectObject
    errors.py     exception types raised during import
    vendors/      one file per vendor plugin (or fallback pair)

Importing this package (or data_import.vendors directly) registers all
built-in plugins; see vendors/__init__.py.
"""
from data_import import vendors  # registers built-in plugins
