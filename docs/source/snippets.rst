Snippets
========


There are couple of snippets we have built to test out small behaviour of things.
Here they are:


(Un)Pickling gotchas
--------------------

Only individual functions can be pickled and not the operations that incoporate. So the test
below will throw ``NameError: name 'fn_inner' is not defined``.

.. code-block::python

  from nbox.utils import to_pickle, from_pickle

  def fn_inner(x):
    return x + 2

  def fn(x):
    return fn_inner(x)


  print("Before pickle:", fn(1))
  to_pickle(fn, "./test.pkl")
  
  del fn_inner, fn # delete local refs
  
  fn = from_pickle("./test.pkl")
  print("After pickle:", fn(1))


Defining the function on the inside can work, but in honesty how much code can you
really write like this.

.. code-block::python

  def fn(x):
    def fn_inner(x):
      return x + 2
    return fn_inner(x)

  print("Before pickle:", fn(1)); to_pickle(fn, "./test.pkl")
  del fn
  print("After pickle:", from_pickle("./test.pkl")(1))


Effective strategy can be to stich inside a class or any other singular structure that works
for you.

.. code-block::python

  from nbox.utils import to_pickle, from_pickle

  class X:
    def fn_inner(self, x):
      return x + 2
    def fn(self, x):
      return self.fn_inner(x)

  x = X()

  print("Before pickle:", x.fn(1)); to_pickle(x, "./test.pkl")
  del x
  print("After pickle:", from_pickle("./test.pkl").fn(1))

