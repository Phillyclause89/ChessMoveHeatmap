Installation
============

Follow these steps to set up the ChessMoveHeatmap project on your local machine.

Prerequisites
-------------
Ensure you have the following installed:
- Python **3.7 - 3.10** (Python **3.11+** is not supported; see `issue #16` for details).
- Git for cloning the repository.

Setup Instructions
------------------
1. **Clone the Repository**:
    Clone the project from GitHub and navigate to the project directory:

    .. code-block:: bash

        git clone https://github.com/Phillyclause89/ChessMoveHeatmap.git
        cd ChessMoveHeatmap

2. **Set Up a Virtual Environment**:
    Create and activate a virtual environment:

    .. code-block:: bash

        python -m venv .venv
        source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

3. **Install Dependencies**:
    Use `requirements.txt` to install the required Python packages:

    .. code-block:: bash

        pip install -r requirements.txt

    > **Note**: Using `.toml` files for dependency installation is not officially supported and may not work as expected.

Optional: Compile for Performance
----------------------------------
For optimal performance, compile the project and its dependencies using Cython:

.. code-block:: bash

    python setup.py

This will improve the speed of recursive calculations and other performance-critical operations.
