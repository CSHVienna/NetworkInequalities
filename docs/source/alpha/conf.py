from netin import __version__ as _version_netin
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NetIn'
copyright = '2023, Fariba Karimi, Lisette Espin-Noboa, Jan Bachmann'
author = 'Fariba Karimi, Lisette Espin-Noboa, Jan Bachmann'
release = _version_netin

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # https://www.sphinx-doc.org/en/master/tutorial/automatic-doc-generation.html
    # Generate documentation from files
    'sphinx.ext.autodoc',
    # https://www.sphinx-doc.org/en/master/tutorial/automatic-doc-generation.html
    # Generate a summary of the project
    'sphinx.ext.autosummary',
    # https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
    # Make numpy style docstrings interpretable by sphinx
    # 'sphinx.ext.napoleon',
    'numpydoc'
]

templates_path = ['_templates']
exclude_patterns = []

suppress_warnings = ["ref.citation", "ref.footnote"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_logo = 'netin-logo.png'
html_favicon = 'netin-logo.png'
html_theme_options = {
}
html_css_files = [
    'style.css',
]

numpydoc_show_class_members = False
