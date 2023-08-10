# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import shutil
import sys
import pathlib
import jupytext  # type: ignore

sys.path.insert(0, os.path.abspath("../../optson"))

SCRIPT_DIR = pathlib.Path(__file__).parent
MODULE_DIR = SCRIPT_DIR.parent / "optson"


def _convert_tutorials():
    """Convert jupytext to ipynb."""
    tut_dir = SCRIPT_DIR / "jupytext_notebooks"
    output_dir = SCRIPT_DIR / "ipynb_notebooks"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    for jupytext_file in tut_dir.glob("*.py"):
        ipynb_file = output_dir / f"{jupytext_file.stem}.ipynb"
        print(f"Converting {jupytext_file}")
        jupytext.write(jupytext.read(jupytext_file), ipynb_file)


# -- Project information -----------------------------------------------------

project = "Optson"
copyright = "2023, Dirk-Philip van Herwaarden & Vaclav Hapla"
author = "Dirk-Philip van Herwaarden & Vaclav Hapla"
language = "en"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinx.ext.autodoc",  # Autodoc extension for documenting classes. We use apidoc for now
    "sphinx.ext.intersphinx",  # Link to other docs
    "sphinx.ext.mathjax",  # Support for equations
    "sphinx.ext.viewcode",  # Provides links to source code.
    "myst_nb",  # Markdown instead of RST and ipynb notebook reading.
    # docstrings
    "sphinx.ext.napoleon",  # Support for Google-style docstrings.
    # "sphinx_autodoc_typehints", # additional support for type hints
    # "sphinx_toolbox.more_autodoc.typevars", # may be interesting to use
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes
html_theme = "sphinx_book_theme"
html_logo = "static/LOGO_DARK.png"
html_theme_options = {
    "repository_branch": "master",
    "path_to_docs": "docs",
    "repository_url": "https://gitlab.com/swp_ethz/projects/inversionson/optson",
    "logo": {
        "image_dark": "static/LOGO_BLUE.png",
    },
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["static"]

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# myst_nb notebook settings
nb_execution_mode = "cache"  # Only execute when the notebooks change
nb_execution_raise_on_error = True  # Stop the build when a notebook raises an exception

# Napoleon settings
napoleon_include_init_with_doc = False

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "special-members": "__call__",
    "undoc-members": True,
    "show_inheritance": True,
}

autodoc_type_aliases = {
    "InstanceOrType": "optson.utils.InstanceOrType",
    "optson.utils.InstanceOrType": "optson.utils.InstanceOrType",
    "Vec": "optson.vector.Vec",
    "optson.vector.Vec": "optson.vector.Vec",
    "InVec": "optson.vector.InVec",
    "optson.vector.InVec": "optson.vector.InVec",
    "Scalar": "optson.vector.Scalar",
    "optson.vector.Scalar": "optson.vector.Scalar",
    "T": "optson.utils.T",
    "optson.utils.T": "optson.utils.T",
    "h5py.File": "h5py.File",
    "h5py.Group": "h5py.Group",
}

# This provides the links to classes defined in other documentations
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "h5py": ("https://docs.h5py.org/en/latest/", None),
}

nitpicky = True  # raise warnings for all missing references
nitpick_ignore = [("py:class", "optional")]


def run_custom_scripts(*args, **kwargs):  # NOQA
    # Generate the API documentation.
    _convert_tutorials()


def resolve_type_aliases(app, env, node, contnode):
    """Resolve :class: references to our type aliases as :data: instead."""
    if (
        node["refdomain"] == "py"
        and node["reftype"] == "class"
        and node["reftarget"] in autodoc_type_aliases.keys()
    ):
        return app.env.get_domain("py").resolve_xref(
            env,
            node["refdoc"],
            app.builder,
            "data",
            node["reftarget"],
            node,
            contnode,
        )


def setup(app):
    # Run custom scripts during the doc building stage.
    app.connect("builder-inited", run_custom_scripts)

    # app.connect("missing-reference", turn_off_wrap)
    app.connect("missing-reference", resolve_type_aliases)
