"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def mul(a: float, b: float) -> float:
    """Multiplies two numbers.

    Parameters
    ----------
    a
      A float to be multiplied.
    b
      A float to be multiplied.

    Returns
    -------
      A product of a and b

    """
    return a * b


def id(i: float) -> float:
    """Returns the input unchanged.

    Parameters
    ----------
    i
      float

    Returns
    -------
      i unchanged

    """
    return i


def add(a: float, b: float) -> float:
    """Adds two numbers.

    Parameters
    ----------
    a
      A float to be added.
    b
      A float to be added.

    Returns
    -------
      A sum of a and b

    """
    return a + b


def neg(a: float) -> float:
    """Negates a number.

    Parameters
    ----------
    a
      float

    Returns
    -------
      A negative float of a

    """
    return -a


def lt(a: float, b: float) -> float:
    """Checks if one number is less than another.

    Parameters
    ----------
    a
      An int to check if less than b
    b
      An int to check if greater than a

    Returns
    -------
      A bool

    """
    return 1.0 if a < b else 0.0


def eq(a: float, b: float) -> float:
    """Checks if two numbers are equal.

    Parameters
    ----------
    a
      An int to check if eq
    b
      An int to check if eq

    Returns
    -------
      A bool

    """
    return 1.0 if a == b else 0.0


def max(a: float, b: float) -> float:
    """Returns the larger of two numbers.

    Parameters
    ----------
    a
      An int.
    b
      An int

    Returns
    -------
      An int

    """
    return a if a > b else b


def is_close(a: float, b: float) -> bool:
    """Checks if two numbers are close in value.

    Parameters
    ----------
    a
      A float.
    b
      A float.

    Returns
    -------
      A bool if the numbers are close in value

    """
    return abs(a - b) < 1e-2


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function.

    Parameters
    ----------
    x
      A float.

    Returns
    -------
      A float

    """
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function.

    Parameters
    ----------
    x
      A float.

    Returns
    -------
      A float

    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Calculates the natural logarithm.

    Parameters
    ----------
    x
      A float.

    Returns
    -------
      A float

    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Calculates the exponential function.

    Parameters
    ----------
    x
      A float.

    Returns
    -------
      A float

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal.

    Parameters
    ----------
    x
      A float.

    Returns
    -------
      A float

    """
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Computes the derivative of log times a second argument.

    Parameters
    ----------
    x
      A float. The input value to the logarithm function. Must be a positive float.
    d
      A float. The scalar value by which to multiply the derivative.

    Returns
    -------
      A float

    """
    return d / (x + EPS)


def inv_back(x: float, d: float) -> float:
    """Computes the derivative of reciprocal times a second argument.

    Parameters
    ----------
    x
      A float.
    d
      A float.

    Returns
    -------
      A float

    """
    return -(1.0 / x**2) * d


def relu_back(x: float, d: float) -> float:
    """Computes the derivative of ReLU times a second argument.

    Parameters
    ----------
    x
      A float.
    d
      A float that is the derivative of ReLu

    Returns
    -------
      A float

    """
    return d if x > 0 else 0


# ## Task 0.3


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Applies a given function to each element of an iterable.

    Parameters
    ----------
    fn
      Function from one value to one value.
    iter
      Iterable object

    Returns
    -------
      A function that takes a list, applies fn to each element, and a new list

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order function that combines elements from two iterables using a given function.
    If one list is shorter than the other, the len of the shorter list is used
    and all other elements in longer list are ignored. If one list is empty, an
    empty list will be returned.

    Parameters
    ----------
    fn
      (two-arg function) -- combine two values
    iter_a
      iterable for first argument of fn. Must be same legnth as iter_b.
    iter_b
      iterable for second argument of fn. Must be same legnth as iter_a.

    Returns
    -------
      takes two lists ls1 and ls2, produce a new list by applying
      fn(x, y) on each pair of elements.

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def reduce(fn: Callable[[float,float],float], start: float) -> Callable[[Iterable[float]],float]:
    """Higher-order function that reduces an iterable to a single value using a given function

    Parameters
    ----------
    fn
      function from two values to one value
    iter
      iterable object containing floats
    init
      float initializing the function when iter is empty

    Returns
    -------
      A single float reduced down by applying fn to iter

    """
    def _reduce(ls:Iterable[float])-> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val
    return _reduce


def negList(iter: Iterable[float]) -> Iterable[float]:
    """Negates a list

    Parameters
    ----------
    iter
      An iterable list of floats

    Returns
    -------
      An iterable of negated values

    """
    return map(neg)(iter)


def addLists(iter_a: Iterable[float], iter_b: Iterable[float]) -> Iterable[float]:
    """Adds two lists together

    Parameters
    ----------
    iter_a
      An iterable list of numbers
    iter_b
      An iterable list of numbers

    Returns
    -------
      An iterable list of the two lists added together

    """
    return zipWith(add)(iter_a, iter_b)


def sum(ls: Iterable[float]) -> float:
    """Sums a list

    Parameters
    ----------
    iter
      An iterable list of numbers

    Returns
    -------
      An integer that is the sum of the iterable

    """
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Product of all values in iterable

    Parameters
    ----------
    ls
      An iterable list of numbers

    Returns
    -------
      An integer that is the product of the iterable

    """
    return reduce(mul, 1.0)(ls)
