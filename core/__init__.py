"""
core/ holds the shared data model used by every part of the software:
ProjectObject (the container that flows through the whole pipeline),
DataInterface (how you read data out of it), DataHandle (RAM vs disk
storage), and MetadataBundle (the four-tier metadata model).

Nothing in this package should ever import from data_import/ or any
future sibling package (reconstruct/, view/, gui/, ...) -- core is meant
to be the stable foundation everything else builds on, not the other
way around. If you ever feel tempted to import, say, data_import.plugin
from inside core/, that's a sign the thing you're adding belongs
somewhere else.
"""
