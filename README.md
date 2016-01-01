ML and Conflict Prevention in Python
====================================

Hacking
-------

Python rewrite and extension of *Machine Learning and Conflict Prevention*.

This work uses data from
[ML and conflict prevention repository](https://github.com/byndcivilization/ML-and-conflict-prevention).

It's available as a submodule repository. In order to get this data, you need
to update with following commands:

```
git submodule init
git submodue update
```

This might take some time. After that run following script to merge data that
has been split into parts:

```
./merge_dpi.py
```

This code includes artificial neural networks extension on the basis work. To
run it, execute `./machine_learn.py`.

License
-------

GNU General Public License version 3 (or later). See COPYING file for more
information.

