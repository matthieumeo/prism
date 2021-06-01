Installation
============

Prism requires Python 3.6 or greater. It was developed and tested on x86_64 systems running MacOS and Linux.


Dependencies
------------


The package extra dependencies are listed in the files ``requirements.txt`` and ``requirements-conda.txt``. 
It is recommended to install those extra dependencies using `Miniconda <https://conda.io/miniconda.html>`_ or
`Anaconda <https://www.anaconda.com/download/#linux>`_. This
is not just a pure stylistic choice but comes with some *hidden* advantages, such as the linking to
``Intel MKL`` library (a highly optimized BLAS library created by Intel).

.. code-block:: bash
   
   >> conda install --channel=conda-forge --file=requirements-conda.txt


Quick Install
-------------

Prism can be installed very simply via the command: 

.. code-block:: bash
   
   >> pip install git+git://github.com/matthieumeo/prism.git@master#egg=prism

If you have previously activated your conda environment ``pip`` will install Prism in said environment. Otherwise it will install it in your base environment together with the various dependencies obtained from the file ``requirements.txt``.


Developper Install
------------------

It is also possible to install Prism from the source for developpers: 


.. code-block:: bash
   
   >> git clone https://github.com/matthieumeo/prism
   >> cd <repository_dir>/
   >> pip install -e .

The package documentation can be generated with: 

.. code-block:: bash
   
   >> conda install sphinx=='2.1.*'            
   >> python3 setup.py build_sphinx -b singlehtml