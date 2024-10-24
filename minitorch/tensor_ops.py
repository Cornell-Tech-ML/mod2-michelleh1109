from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        """Call a map function"""
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Map placeholder"""
        ...

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Zip placeholder"""
        ...

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Reduce placeholder"""
        ...

    @staticmethod
    def expand(tensor: Tensor, shape: Shape) -> Tensor:
        """Expand placeholder"""
        ...

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiply"""
        # Ensure tensors are 2D
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise ValueError("Matrix multiplication requires 2D tensors.")

        # Ensure compatible dimensions for matrix multiplication
        if a.shape[1] != b.shape[0]:
            raise ValueError(
                f"Incompatible dimensions: {a.shape} and {b.shape} for matrix multiplication."
            )

        m, n = a.shape
        _, p = b.shape

        # Create an output tensor of shape (m, p)
        out = a.zeros((m, p))

        # Use tensor operations for matrix multiplication
        for i in range(m):
            for j in range(p):
                # Calculate dot product for position (i,j)
                sum_val = 0.0
                for k in range(n):
                    # Access elements directly from storage
                    a_pos = i * n + k
                    b_pos = k * p + j
                    sum_val += a._tensor._storage[a_pos] * b._tensor._storage[b_pos]
                out._tensor._storage[i * p + j] = sum_val

        return out

    # @staticmethod
    # def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
    #     print("------------------------")
    #     print(type(b))
    #     """Matrix multiply"""
    #     # Ensure tensors are 2D
    #     if len(a.shape) != 2 or len(b.shape) != 2:
    #         raise ValueError("Matrix multiplication requires 2D tensors.")

    #     # Ensure compatible dimensions for matrix multiplication
    #     if a.shape[1] != b.shape[0]:
    #         raise ValueError(
    #             f"Incompatible dimensions: {a.shape} and {b.shape} for matrix multiplication."
    #         )

    #     m, n = a.shape
    #     _, p = b.shape

    #     # Create an output tensor of shape (m, p)
    #     out = a.zeros((m, p))

    #     # Perform matrix multiplication using nested loops
    #     for i in range(m):  # Iterate over rows of a
    #         for j in range(p):  # Iterate over columns of b
    #             sum_val = 0.0
    #             for k in range(n):  # Iterate over shared dimension
    #                 sum_val += a[i, k] * b[k, j]
    #             out[i, j] = sum_val

    #     return out

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
        ----
            ops : tensor operations object see `tensor_ops.py`


        Returns:
        -------
            A collection of tensor functions

        """
        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda

        self.expand = ops.expand


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
        ----
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
        -------
            new tensor data

        """
        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
        -------
            :class:`TensorData` : new tensor data

        """
        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """Higher-order tensor reduce function. ::

          fn_reduce = reduce(fn)
          out = fn_reduce(a, dim)

        Simple version ::

            for j:
                out[1, j] = start
                for i:
                    out[1, j] = fn(out[1, j], a[i, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to reduce over
            dim (int): int of dim to reduce
            start: initiai value in storage

        Returns:
        -------
            :class:`TensorData` : new tensor

        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def expand(tensor: Tensor, s: Shape) -> Tensor:
        """Expand the input tensor to the given shape by broadcasting."""
        out = tensor.zeros(tuple(s))
        tensor.f.id_map(tensor, out)  # Map the values from `tensor` to `out`
        return out

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        """Matrix multiplication"""
        return TensorOps.matrix_multiply(a, b)

    is_cuda = False


# Implementations.


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
    ----
        fn: function from float-to-float to apply

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # elements in tensor
        n = 1
        for i in out_shape:
            n *= i

        # loop through every element in flattened array
        out_index = np.array([0] * len(out_shape))
        in_index = np.array([0] * len(in_shape))
        for i in range(n):
            # store out index
            to_index(i, in_shape, out_index)

            # use out index to get the original value in in_shape

            broadcast_index(
                out_index, out_shape, in_shape, in_index
            )  # in_index has been modified to contain index mapped to out_index

            # modify the value and store back into out (storage)
            out_offset = sum(o * s for o, s in zip(out_index, out_strides))
            in_offset = sum(i * s for i, s in zip(in_index, in_strides))

            # Apply the function and store the result
            out[out_offset] = fn(in_storage[in_offset])

    return _map


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
    ----
        fn: function mapping two floats to float to apply

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        n = 1
        for i in out_shape:
            n *= i

        out_index = np.zeros(len(out_shape), dtype=np.int32)
        a_index = np.zeros(len(a_shape), dtype=np.int32)
        b_index = np.zeros(len(b_shape), dtype=np.int32)

        for out_pos in range(n):
            # Convert the flat `out_pos` to the multi-dimensional `out_index`
            to_index(out_pos, out_shape, out_index)

            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            out_offset = sum(
                out_index[i] * out_strides[i] for i in range(len(out_index))
            )
            a_offset = sum(a_index[i] * a_strides[i] for i in range(len(a_index)))
            b_offset = sum(b_index[i] * b_strides[i] for i in range(len(b_index)))

            # Apply the function and store the result
            out[out_offset] = fn(a_storage[a_offset], b_storage[b_offset])

    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
    ----
        fn: reduction function mapping two floats to float

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        n = 1
        for i in out_shape:
            n *= i

        # loop over each element in out_shape
        out_index = np.array([n] * len(out_shape))
        a_index = np.array([n] * len(a_shape))

        for pos in range(n):
            # getting index of pos and storing into out_index
            to_index(pos, out_shape, out_index)

            # copy over everyting from a to out
            np.copyto(a_index, out_index)  # broadcasts as needed

            # getting first accumalator value from a
            a_index[reduce_dim] = 0
            acc = a_storage[index_to_position(a_index, a_strides)]

            for i in range(1, a_shape[reduce_dim]):
                a_index[reduce_dim] = i
                acc = fn(acc, a_storage[index_to_position(a_index, a_strides)])

            # storing reduction result in out
            offset = index_to_position(out_index, out_strides)
            out[offset] = acc

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
