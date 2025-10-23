
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LLM-Learn'
copyright = '2024, Kai Feng'
author = 'Kai Feng'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.githubpages",
    "sphinxcontrib.mermaid",
]

templates_path = ['_templates']
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']

# MyST-Parser configuration
myst_enable_extensions = [
    "colon_fence",  # Enables admonitions
    "deflist",
    "dollarmath",   # Enables TeX-style math blocks
    "html_image",
]
myst_heading_anchors = 3  # Auto-generate header anchors up to level 3

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_title = "LLM-Learn"

html_theme_options = {
    "repository_url": "https://github.com/kaifeng/llm-learn",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "launch_buttons": {
        "colab_url": "https://colab.research.google.com/github/kaifeng/llm-learn/blob/main/source"
    },
}
