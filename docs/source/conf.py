# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NetIn'
copyright = '2023, Fariba Karimi, Lisette Espin-Noboa, Jan Bachmann'
author = 'Fariba Karimi, Lisette Espin-Noboa, Jan Bachmann'
release = '1.0.5'

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
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
