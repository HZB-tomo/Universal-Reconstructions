"""
core/ holds the shared data model: ProjectObject (the container that
flows through the pipeline), DataInterface (how data is read out of
it), DataHandle (RAM vs disk storage), and MetadataBundle (the
four-tier metadata model).

Nothing in this package imports from data_import/ or any sibling
package.
"""
