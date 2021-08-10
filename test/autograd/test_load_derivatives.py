import dataclasses
import unittest

from tools.autograd import load_derivatives
import tools.codegen.model

class TestCreateDerivative(unittest.TestCase):

    def test_this_is_run(self):
        """Sanity check that this is run in CI: we expect a failure now."""
        # DO NOT MERGE
        assert False

    def test_named_grads(self):
        schema = tools.codegen.model.FunctionSchema.parse(
            'func(Tensor a, Tensor b) -> (Tensor x, Tensor y)')
        native_function = dataclasses.replace(DEFAULT_NATIVE_FUNCTION, func=schema)

        derivative = load_derivatives.create_derivative(
            native_function,
            formula='func_backward(grad_x, grad_y)',
            var_names=())
        self.assertDictEqual(derivative.named_grads, {'grad_x': 0, 'grad_y': 1})

    def test_named_grads(self):
        schema = tools.codegen.model.FunctionSchema.parse(
            'func(Tensor a, Tensor b) -> (Tensor x, Tensor y)')
        native_function = dataclasses.replace(DEFAULT_NATIVE_FUNCTION, func=schema)

        derivative = load_derivatives.create_derivative(
            native_function,
            formula='func_backward(grads[0], grads[1])',
            var_names=())
        self.assertDictEqual(derivative.named_grads, {})

    # TODO add tests demonstrating policy violations, e.g. specifying
    #      grads[1] and grad_x in the same formula or across formulas.


class TestInsertOrRaise(unittest.TestCase):
    def test_empty(self):
        d = {}
        load_derivatives.insert_or_raise(d, 'a', 1)
        self.assertDictEqual(d, {'a': 1})

    def test_match(self):
        d = {'a': 1}
        load_derivatives.insert_or_raise(d, 'a', 1)
        self.assertDictEqual(d, {'a': 1})

    def test_conflict(self):
        d = {'a': 1}
        with self.assertRaisesRegex(RuntimeError, 'maps to distinct values'):
            load_derivatives.insert_or_raise(d, 'a', 2)


class TestMergeDicts(unittest.TestCase):
    def test_empty(self):
        self.assertDictEqual(load_derivatives.merge_dicts(), {})

    def test_distinct_keys(self):
        d = {'a': 1}
        e = {'b': 1}
        self.assertDictEqual(load_derivatives.merge_dicts(d, e), {'a': 1, 'b': 1})

    def test_match(self):
        d = {'a': 1}
        e = {'a': 1}
        self.assertDictEqual(load_derivatives.merge_dicts(d, e), {'a': 1})

    def test_distinct_values(self):
        d = {'a': 1}
        e = {'a': 2}
        with self.assertRaisesRegex(RuntimeError, 'maps to distinct values'):
            load_derivatives.merge_dicts(d, e)


DEFAULT_NATIVE_FUNCTION, _ = tools.codegen.model.NativeFunction.from_yaml(
    {'func': 'func() -> bool'}, loc=tools.codegen.model.Location(__file__, 1))


if __name__ == '__main__':
    unittest.main()
