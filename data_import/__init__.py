"""
data_import/ contains everything related to turning a file on disk into
a ProjectObject (see core/project.py). This is the one place in the
codebase that uses a formal "plugin" pattern, because it's the one place
where all three conditions for needing plugins are actually true (see
our discussion): multiple vendors write files differently, the software
must pick the right interpretation automatically, and third parties
should be able to add support for their own instrument without editing
this package's code.

Layout:
    path_map.py   - PathMap: where things are, once located
    signature.py  - Signature/Hdf5Signature/TiffSignature: how a plugin
                     declares "this is my kind of file" (deterministic
                     checklist, not a confidence score -- see signature.py)
    plugin.py     - ImportPlugin: the contract every vendor plugin
                     implements, plus the registry plugins sign up with
    backend.py    - FormatBackend/Hdf5Backend/TiffBackend: how to
                     physically read hdf5/tiff, independent of vendor
    importer.py   - DataImporter: orchestrates detection -> plugin ->
                     backend -> ProjectObject
    errors.py     - exception types raised during import
    vendors/      - one file per vendor plugin (or fallback pair)

Importing this package (or data_import.vendors directly) registers all
built-in plugins as a side effect -- see vendors/__init__.py.
"""
from data_import import vendors  #   (registers built-in plugins)
