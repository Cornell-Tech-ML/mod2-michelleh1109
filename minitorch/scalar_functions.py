from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies a `ScalarFunction` to the provided inputs by performing the forward pass and storing
        the history necessary for the backward pass.

        Args:
        ----
            *vals (ScalarLike): The input values to the function. These can be `Scalar` objects,
                                floats, or ints. If the inputs are not `Scalar` objects, they are
                                automatically converted into `Scalar`.

        Returns:
        -------
            Scalar: A new `Scalar` object resulting from the forward pass of the function.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Scalar Addition"""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass"""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Scalar Log Function"""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for log function"""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.
class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        """Multiplication forward pass

        Parameters
        ----------
        ctx
            Context class to store information in forward pass
        x
            A float to be multiplied.
        y
            A float to be multiplied.

        Returns
        -------
        A product of x and y

        """
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Multiplication backward pass

        Parameters
        ----------
        ctx
            Context class to store information in forward pass
        d_output
            A float

        Returns
        -------
        Derivative of x and y

        """
        x, y = ctx.saved_values
        return y * d_output, x * d_output


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Inverse forward pass

        Parameters
        ----------
        ctx
            Context class to store information in forward pass
        x
            A float

        Returns
        -------
        Inverse of x

        """
        ctx.save_for_backward(x)
        return operators.inv(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Inverse backward pass

        Parameters
        ----------
        ctx
            Context class to store information in forward pass
        d_output
            A float

        Returns
        -------
        Derivative of 1/x

        """
        (x,) = ctx.saved_values
        return operators.inv_back(x, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Negation forward pass

        Parameters
        ----------
        ctx
            Context class to store information in forward pass
        a
            A float

        Returns
        -------
        Negation of a

        """
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Negation backward pass

        Parameters
        ----------
        ctx
            Context class to store information in forward pass
        d_output
            A float

        Returns
        -------
        Derivative of -x

        """
        return -d_output


class Sigmoid(ScalarFunction):
    r"""Sigmoid function $f(x) = \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else
    $\frac{e^x}{(1.0 + e^{x})}$

    """

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Sigmoid forward pass

        Parameters
        ----------
        ctx
            Context class to store information in forward pass
        x
            A float

        Returns
        -------
        Sigmoid of x

        """
        out = operators.sigmoid(x)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Sigmoid backward pass

        Parameters
        ----------
        ctx
            Context class to store information in forward pass
        d_output
            A float

        Returns
        -------
        Derivative of the sigmoid function

        """
        sigma: float = ctx.saved_values[0]
        return sigma * d_output * (1 - sigma)


class ReLU(ScalarFunction):
    r"""ReLU function $f(x) = \max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """ReLU forward pass

        Parameters
        ----------
        ctx
            Context class to store information in forward pass
        x
            A float

        Returns
        -------
        ReLU of a

        """
        ctx.save_for_backward(x)
        return operators.relu(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """ReLU backward pass

        Parameters
        ----------
        ctx
            Context class to store information in forward pass
        d_output
            A float

        Returns
        -------
        Derivative of ReLU

        """
        (x,) = ctx.saved_values
        return operators.relu_back(x, d_output)


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Exponential forward pass

        Parameters
        ----------
        ctx
            Context class to store information in forward pass
        x
            A float

        Returns
        -------
        Exponential of x

        """
        result = operators.exp(x)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Exponential backward pass

        Parameters
        ----------
        ctx
            Context class to store information in forward pass
        d_output
            A float

        Returns
        -------
        Derivative of e^x

        """
        (result,) = ctx.saved_values
        return d_output * result


class LT(ScalarFunction):
    """Less-than function $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Less-than forward pass

        Parameters
        ----------
        ctx
            Context class to store information in forward pass
        a
            A float
        b
            A float

        Returns
        -------
        Result of the less-than comparison

        """
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Less-than backward pass

        Parameters
        ----------
        ctx
            Context class to store information in forward pass
        d_output
            A float

        Returns
        -------
        Tuple containing zero gradients

        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Equal forward pass

        Parameters
        ----------
        ctx
            Context class to store information in forward pass
        a
            A float
        b
            A float

        Returns
        -------
        Result of the equality comparison

        """
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Equal backward pass

        Parameters
        ----------
        ctx
            Context class to store information in forward pass
        d_output
            A float

        Returns
        -------
        Tuple containing zero gradients

        """
        return 0.0, 0.0
