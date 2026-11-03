"""Trace-edge checks for class-first, unfused timm ViT; no graph arithmetic."""
from __future__ import annotations

import operator

from .export import _arguments, _under, operation_name


def calls_at(traced, path, *, children=True):
    return tuple(n for n in traced.calls if n in traced.live
                 and (_under(traced.path(n), path) if children else traced.path(n) == path))


class Difference(Exception):
    """An expected trace edge is absent."""


def require(condition, message):
    if not condition:
        raise Difference(message)


def one(traced, path, operation=None):
    found = [n for n in calls_at(traced, path, children=bool(path))
             if operation is None or operation_name(n.target) == operation]
    require(len(found) == 1, f"{path} requires one {operation or 'call'}; found {len(found)}")
    return found[0]


def parameters(traced, calls):
    """Pin affine parameter roles, which a set of parameter names cannot distinguish."""
    for node in calls:
        if operation_name(node.target) not in ('aten.linear.default', 'aten.conv2d.default',
                                              'aten.layer_norm.default'):
            continue
        arguments = _arguments(node)
        path = traced.path(node)
        for role in ('weight', 'bias'):
            value = arguments[role]
            if value is not None and traced.parameters.get(value.name) != path + '.' + role:
                return f'{path}: {role} must read its own {role} parameter'
    return None


def attention(path, dim, heads):
    def check(traced, calls):
        try:
            qkv = one(traced, path + '.qkv', 'aten.linear.default')
            owned = calls_at(traced, path)
            def step(op, source, **arguments):
                found = [n for n in owned if operation_name(n.target) == op and n.args[0] is source]
                require(len(found) == 1, f'{path}: {op} must read {source.name}')
                node = found[0]
                args = _arguments(node)
                require(all(args[k] == v for k, v in arguments.items()), f'{path}: {op} changes head layout')
                return node
            shaped = step('aten.reshape.default', qkv, shape=[1, 197, 3, heads, dim // heads])
            packed = step('aten.permute.default', shaped, dims=[2, 0, 3, 1, 4])
            split = step('aten.unbind.int', packed, dim=0)
            branches = []
            for i in range(3):
                found = [n for n in owned if n.target is operator.getitem and n.args == (split, i)]
                require(len(found) == 1, f'{path}: missing Q/K/V branch {i}')
                branches.append(found[0])
            q, k, v = branches
            scaled = step('aten.mul.Tensor', q)
            kt = step('aten.transpose.int', k, dim0=-2, dim1=-1)
            scores = step('aten.matmul.default', scaled)
            require(scores.args == (scaled, kt), f'{path}: scores must read scaled Q and transposed K')
            probabilities = step('aten.softmax.int', scores, dim=-1)
            dropped = step('aten.dropout.default', probabilities)
            context = step('aten.matmul.default', dropped)
            require(context.args == (dropped, v), f'{path}: context must read probabilities and V')
            joined = step('aten.transpose.int', context, dim0=1, dim1=2)
            joined = step('aten.reshape.default', joined, shape=[1, 197, dim])
            proj = one(traced, path + '.proj', 'aten.linear.default')
            require(proj.args[0] is joined, f'{path}: projection must read joined attention heads')
            drop = one(traced, path + '.proj_drop', 'aten.dropout.default')
            require(drop.args[0] is proj, f'{path}: projection dropout must read projection')
        except Difference as error:
            return str(error)
        return None
    return check


def block(path):
    def check(traced, calls):
        try:
            norms = [one(traced, path + '.' + n, 'aten.layer_norm.default') for n in ('norm1', 'norm2')]
            residuals = calls_at(traced, path, children=False)
            require(len(residuals) == 2, f'{path}: expected two residuals')
            first, second = residuals
            qkv = one(traced, path + '.attn.qkv', 'aten.linear.default')
            require(qkv.args[0] is norms[0], f'{path}: QKV must read norm1')
            proj = one(traced, path + '.attn.proj_drop', 'aten.dropout.default')
            require(first.args[:2] == (norms[0].args[0], proj), f'{path}: first residual order or state differs')
            require(norms[1].args[0] is first, f'{path}: norm2 must follow first residual')
            state = norms[1]
            for suffix, op in (('fc1', 'aten.linear.default'), ('act', 'aten.gelu.default'),
                               ('drop1', 'aten.dropout.default'), ('fc2', 'aten.linear.default'),
                               ('drop2', 'aten.dropout.default')):
                next_state = one(traced, path + '.mlp.' + suffix, op)
                require(next_state.args[0] is state, f'{path}: MLP {suffix} reads another state')
                state = next_state
            require(second.args[:2] == (first, state), f'{path}: second residual order or state differs')
        except Difference as error:
            return str(error)
        return None
    return check


def position(depth, *, path="", final=True, position_path=None):
    def check(traced, calls):
        try:
            root = path + '.' if path else ''
            def local(op):
                found = [n for n in calls_at(traced, path if position_path is None else position_path, children=False)
                         if operation_name(n.target) == op]
                require(len(found) == 1, f'{path}: expected one {op}')
                return found[0]
            cat = local('aten.cat.default')
            require(cat.args[1] == 1 and len(cat.args[0]) == 2, 'class token must be concatenated on row axis 1')
            cls, spatial = cat.args[0]
            require(operation_name(cls.target) == 'aten.expand.default', 'class token must occupy row 0')
            require(traced.parameters.get(cls.args[0].name) == root + 'cls_token', 'row 0 must read cls_token')
            patch = calls_at(traced, root + 'patch_embed')[-1]
            require(spatial is patch, 'spatial rows must follow class row in row-major patch order')
            add = local('aten.add.Tensor')
            require(add.args[0] is cat and traced.parameters.get(add.args[1].name) == root + 'pos_embed',
                    'position addition must read the class-first concatenation and pos_embed')
            drop = one(traced, root + 'pos_drop', 'aten.dropout.default')
            require(drop.args[0] is add, 'position dropout must follow position addition')
            state = drop
            for i in range(depth):
                norm = one(traced, f'{root}blocks.{i}.norm1', 'aten.layer_norm.default')
                require(norm.args[0] is state, f'blocks.{i} must follow position addition / previous residual')
                state = calls_at(traced, f'{root}blocks.{i}', children=False)[-1]
            if final:
                norm = one(traced, root + 'norm', 'aten.layer_norm.default')
                require(norm.args[0] is state, 'final norm must follow the last block')
                output = one(traced, root + 'head_drop', 'aten.dropout.default')
                require(output.args[0] is norm and traced.output is output, 'encoder must return the final norm sequence')
        except Difference as error:
            return str(error)
        return None
    return check
