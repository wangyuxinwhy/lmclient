# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import sys
from pathlib import Path

project_dir = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_dir))
project = 'lmclient'
copyright = '2023, wangyuxin'
author = 'wangyuxin'
release = '0.9.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
]
templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_theme_options = {
    'light_css_variables': {
        'font-stack': 'Arial, sans-serif',
        'font-stack--monospace': 'Courier, monospace',
    },
}

intersphinx_mapping = {
    'pydantic': ('https://docs.pydantic.dev/latest/', None),
}