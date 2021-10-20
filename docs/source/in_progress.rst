TODO
====

There are some qierd quicks when using ``nbox``, yes we all are aware of it. Here are some fixes
that we are working on:

1. Adding harware acceleration to ``nbox.Model`` where supported
2. Ability to serialise and deserialise ``nbox.Model`` using ``__setstate__`` and ``__getstate__``
   methods. This will be super crucial to portability of model across locations and simplifies
   the deployment process immensely and would reduce dependencies on the backend engineering team.
3. Support for all ``sklearn`` models will be done on a case by case basis, currenty ``nbox.Model``
   works good with ``RandomForest`` algorithm, we need to expand it to others.
4. Support for using ``nbox`` with ``Airflow``.
