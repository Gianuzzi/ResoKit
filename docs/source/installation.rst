Installation
============

You need Python 3.10 or greater to run `ResoKit`.


Installing  with pip
^^^^^^^^^^^^^^^^^^^^

Make sure that the Python interpreter can load `ResoKit` code.
The most convenient way to do this is to use virtualenv, virtualenvwrapper, and pip.

After setting up and activating the virtualenv, run the following command:

.. code-block:: console

   $ python -m pip install resokit

That should be it all.



Installing the development version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you’d like to be able to update your `ResoKit` code occasionally with the
latest improvements and bug fixes, follow these instructions:

Make sure that you have Git installed and that you can run its commands from a shell.
(Enter *git help* at a shell prompt to test this.)

Check out `ResoKit` main development branch like so:

.. code-block:: console

   $ git clone https://github.com/gianuzzi/resokit.git


This will create a directory *resokit* in your current directory.

Then you can proceed to install with the commands

.. code-block:: console

   $ cd resokit
   $ python -m pip install -e .
