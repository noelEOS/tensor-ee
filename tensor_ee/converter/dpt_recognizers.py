"""SMP 0.5.0 DPT: checked mappings to existing window blocks and ViT pair units."""
from __future__ import annotations

from dataclasses import replace

from ..core.errors import UnsupportedOperation
from ..core.grid import TokenGrid
from . import recognizers as R
from . import vit_dataflow as flow
from .dpt_fingerprints import FINGERPRINTS, SOURCE, TRACED_WITH
from .export import (ClassTokenExport, Forward, NCHW, Through, Unit, _arguments, _shape,
                     operation_name, qualified_name)
from .graph import Edge
from .operations import get_operation

DPT = 'segmentation_models_pytorch.decoders.dpt'
ENCODER = 'segmentation_models_pytorch.encoders.timm_vit.TimmViTEncoder'
DECODER = DPT + '.decoder.'
NTC = ('N', 'T', 'C')


def _safe(check):
    def wrapped(t, calls):
        try:
            return check(t, calls)
        except (flow.Difference, UnsupportedOperation, AttributeError, IndexError, KeyError) as error:
            return str(error)
    return wrapped


def _forward(kind, check=None):
    def checked(t, calls):
        return flow.parameters(t, calls) or (None if check is None else _safe(check)(t, calls))
    return Forward(FINGERPRINTS[kind], SOURCE, checked, axes=True)


def _unit(path, op, settings, kind, *, owns=None, inputs=None, **fields):
    return Unit(path, op, settings,
                {r: path + '.' + r for r in get_operation(op).roles(settings)},
                owns=(path,) if owns is None else owns, inputs=(path,) if inputs is None else inputs,
                window=14, forward=_forward(kind), **fields)


def _full_check(unit, t, path, kind, check=None):
    """Attach an enclosing module's full fingerprint and dataflow to its first unit."""
    held = unit.forward
    def checked(traced, calls):
        return (held.difference(traced, calls)
                or _forward(kind, check).difference(traced, flow.calls_at(traced, path)))
    return replace(unit, forward=Forward(held.calls, SOURCE, checked, axes=True))


def _conv(module, path, *, kernel=3, stride=1, bias=False):
    R._conv2d(module, path, module.in_channels, module.out_channels, kernel, stride, kernel // 2, bias=bias)
    if kernel == 1:
        return _unit(path, 'window_linear', dict(in_features=module.in_channels, out_features=module.out_channels,
                     bias=bias, conv=True), 'conv')
    return _unit(path, 'window_conv3x3', dict(cin=module.in_channels, cout=module.out_channels,
                 stride=stride, bias=bias), 'conv')


def _bn(module, path, channels):
    R._batch_norm(module, path, channels)
    def roles(t, calls):
        args = _arguments(calls[0])
        for role in ('weight', 'bias', 'running_mean', 'running_var'):
            flow.require(t.parameters.get(args[role].name) == path + '.' + role,
                         f'{path}: BatchNorm {role} must read its own parameter or buffer')
    return replace(_unit(path, 'window_batch_norm', {'eps': module.eps}, 'bn'), forward=_forward('bn', roles))


def _binary(path, node, kind='add'):
    return _unit(path, 'window_add', {}, kind, owns=(path.rsplit('.', 1)[0],),
        select=lambda n, held=node: n is held,
        inputs=tuple(Through(f'operand {i}', lambda t, c, i=i: c[0].args[i]) for i in range(2)))


def _resize(path, node, out_window):
    def check(t, calls):
        a = _arguments(calls[0])
        flow.require(a['output_size'] is None and a['scale_factors'] == [2.0, 2.0]
                     and a['align_corners'] is True, f'{path}: resize must use x2 bilinear align_corners=True')
    unit = _unit(path + '.resize', 'dpt_resize', dict(factor=2, align_corners=True, out_window=out_window),
        'resize', owns=(path,), inputs=(path,), select=lambda n, held=node: n is held)
    return replace(unit, forward=_forward('resize', check))


def _exporter(t):
    model = t.modules['encoder.model']
    bindings = {flow.calls_at(t, f'encoder.model.blocks.{i}', children=False)[-1]:
                f'encoder.model.blocks.{i}.residual2' for i in t.modules['encoder']._output_indices}
    edges = {name: Edge('P', TokenGrid(14, 14, 1.), model.embed_dim, window=14) for name in bindings.values()}
    exporter = ClassTokenExport(t, edges)
    for node, name in bindings.items():
        exporter.bind(node, name)
    return exporter


def _encoder(module, path, t):
    model, p = R._child(module, path, 'model', R.VIT_MODEL)
    depth = len(model.blocks)
    expected = tuple(depth * i // 4 - 1 for i in range(1, 5))
    if depth < 4 or tuple(module._output_indices) != expected:
        R._refuse(path, module, f'output_indices must be the four uniform raw block indices {expected}')
    if (module._num_prefix_tokens != 1 or model.num_prefix_tokens != 1
            or not module.has_prefix_tokens):
        R._refuse(path, module, 'exactly one prefix token is required for DPT cat readout')
    if tuple(module.output_strides) != (16,) * 4 or tuple(module.out_channels) != (model.embed_dim,) * 4:
        R._refuse(path, module, 'four stride-16 outputs preserving embed_dim are required')
    units = list(R._vision_transformer(model, p, t, intermediates=True))
    # forward_intermediates is a method call, so torch records its root calls at
    # the SMP wrapper path rather than at encoder.model.
    position = units[1]
    position_calls = {n for n in flow.calls_at(t, path, children=False)
                      if operation_name(n.target) in ('aten.expand.default', 'aten.cat.default', 'aten.add.Tensor')}
    units[1] = replace(position, owns=(path, p + '.pos_drop'), inputs=(path,),
        select=lambda n: n in position_calls or t.path(n) == p + '.pos_drop',
        forward=R._vit_forward('position', flow.position(depth, path=p, final=False, position_path=path)))
    # Bind the known block ends to pair descriptions. ClassTokenExport checks the
    # exact 0:1/1:197 slices and row-major NCHW moves, without any graph arithmetic.
    bindings = {}
    for i in expected:
        b = p + f'.blocks.{i}'
        bindings[flow.calls_at(t, b, children=False)[-1]] = b + '.residual2'
    def validate(traced, endpoint, want):
        edges = {name: Edge('P', TokenGrid(14, 14, 1.), model.embed_dim, window=14) for name in bindings.values()}
        exporter = ClassTokenExport(traced, edges)
        for node, name in bindings.items():
            exporter.bind(node, name)
        flow.require(exporter.resolve(endpoint) == want, f'{path}: selected intermediate must read {want}')
    selections = [n for n in flow.calls_at(t, path, children=False) if operation_name(n.target) == 'aten.slice.Tensor']
    if len(selections) != 8:
        R._refuse(path, module, 'forward_intermediates must select four spatial/class pairs')
    for i, block_index in enumerate(expected):
        for component, rows in (('prefix', 1), ('spatial', 196)):
            source = next(node for node, name in bindings.items() if name == p + f'.blocks.{block_index}.residual2')
            candidates = [n for n in selections if _shape(n)[1] == rows]
            selected = candidates[i:i+1]
            if len(selected) != 1:
                R._refuse(path, module, f'one {component} selection from block {block_index} required')
            chain = [selected[0]]
            while True:
                next_calls = [n for n in chain[-1].users if n in t.live and t.path(n) == path]
                if not next_calls:
                    break
                if len(next_calls) != 1:
                    R._refuse(path, module, 'intermediate selection must have one layout chain')
                chain.append(next_calls[0])
            endpoint = chain[-1]
            wanted = bindings[source] + '.' + component
            def checked(traced, calls, endpoint=endpoint, wanted=wanted):
                validate(traced, endpoint, wanted)
            u = _unit(wanted, 'pair_select', {'component': component}, 'select_' + component,
                owns=(path,), inputs=(Through('raw intermediate pair', lambda t, c: c[0].args[0]),),
                axes=NTC, output_axes=NCHW if component == 'spatial' else NTC,
                select=lambda n, held=frozenset(chain): n in held)
            units.append(replace(u, forward=_forward('select_' + component, checked)))
    # Record the wrapper's complete own forward, independent of ViT depth.
    held = units[1].forward
    def wrapper(traced, calls):
        return (held.difference(traced, calls)
                or _forward('encoder').difference(traced, flow.calls_at(traced, path, children=False)))
    units[1] = replace(units[1], forward=Forward(held.calls, SOURCE, wrapper, axes=True))
    return tuple(units)


def _readout(module, path, t):
    project, p = R._child(module, path, 'project', R.SEQUENTIAL)
    if len(project) != 2:
        R._refuse(path, module, 'cat readout project must be Linear then exact GELU')
    fc, f = R._child(project, p, '0', R.LINEAR)
    R._linear(fc, f, 2 * fc.out_features, fc.out_features)
    act, a = R._child(project, p, '1', R.GELU)
    if act.approximate != 'none':
        R._refuse(a, act, 'GELU approximate must be none')
    direct = flow.calls_at(t, path, children=False)
    cut = t.order[flow.one(t, f, 'aten.linear.default')]
    before = frozenset(n for n in direct if t.order[n] < cut)
    after = frozenset(n for n in direct if t.order[n] > cut)
    def cat_inputs(traced, calls, index):
        # Feature view reads the original NCHW spatial tensor; alias/slice reads the prefix.
        return calls[0].args[0] if index == 0 else next(n for n in calls if operation_name(n.target)
                                                     in ('aten.alias.default', 'aten.slice.Tensor')).args[0]
    cat = _unit(path + '.cat', 'dpt_cat_readout', {}, 'cat', owns=(path,),
        select=lambda n: n in before, inputs=tuple(Through(f'readout input {i}',
        lambda t, c, i=i: cat_inputs(t, c, i)) for i in range(2)), axes=NTC, input_axes=(NCHW, NTC))
    units = [cat, _unit(f, 'window_linear', dict(in_features=fc.in_features, out_features=fc.out_features,
             bias=True, conv=False), 'linear', axes=NTC),
        _unit(a, 'window_gelu', {}, 'readout_gelu', owns=(a, path), inputs=(a,), axes=NTC,
              output_axes=NCHW, select=lambda n: n in after or t.path(n) == a)]
    def check(traced, calls):
        local = flow.calls_at(traced, path, children=False)
        view, trans, contiguous, prefix, expanded, cat_node, tail_trans, tail_view = local
        index = int(path.rsplit('.', 1)[1])
        block = traced.modules['encoder']._output_indices[index]
        expected = f'encoder.model.blocks.{block}.residual2'
        exporter = _exporter(traced)
        flow.require(exporter.resolve(view.args[0]) == expected + '.spatial'
                     and exporter.resolve(prefix.args[0]) == expected + '.prefix',
                     f'{path}: readout must use its corresponding spatial/class intermediate pair')
        flow.require(trans.args[0] is view and contiguous.args[0] is trans,
                     'readout spatial rows must stay row-major')
        flow.require(expanded.args == (prefix, contiguous) and cat_node.args == ([contiguous, expanded], 2),
                     'readout concatenation must place spatial columns before repeated class columns')
        flow.require(_shape(prefix) == (1, 1, fc.out_features), 'readout requires exactly one prefix row')
        linear_node = flow.one(traced, f, 'aten.linear.default')
        gelu_node = flow.one(traced, a, 'aten.gelu.default')
        flow.require(linear_node.args[0] is cat_node and gelu_node.args[0] is linear_node
                     and tail_trans.args[0] is gelu_node and tail_view.args[0] is tail_trans,
                     'readout projection and GELU must follow the ordered cat')
    units[0] = _full_check(units[0], t, path, 'readout', check)
    return tuple(units)


def _reassemble(module, path, t):
    index = int(path.rsplit('.', 1)[1])
    first, f = R._child(module, path, 'project_to_out_channel', R.CONV2D)
    last, l = R._child(module, path, 'project_to_feature_dim', R.CONV2D)
    units = [_conv(first, f, kernel=1, bias=True)]
    up = module.upsample
    u = path + '.upsample'
    if index < 2:
        factor = (4, 2)[index]
        if (qualified_name(up) != 'torch.nn.modules.conv.ConvTranspose2d'
                or up.in_channels != first.out_channels or up.out_channels != first.out_channels
                or tuple(up.kernel_size) != (factor, factor) or tuple(up.stride) != (factor, factor)
                or tuple(up.padding) != (0, 0) or tuple(up.output_padding) != (0, 0)
                or tuple(up.dilation) != (1, 1) or up.groups != 1 or up.bias is None or up.padding_mode != 'zeros'):
            R._refuse(u, up, f'upsampling must be biased non-overlapping ConvTranspose x{factor}')
        units.append(_unit(u, 'dpt_transpose_linear', dict(channels=up.in_channels, factor=factor), 'transpose',
            post_op=('window_depth_to_space', dict(factor=factor, out_window=7))))
    elif index == 2:
        R._child(module, path, 'upsample', R.IDENTITY)
    elif index == 3:
        R._child(module, path, 'upsample', R.CONV2D)
        if up.in_channels != first.out_channels or up.out_channels != first.out_channels:
            R._refuse(u, up, 'stride-2 reassemble must preserve intermediate channels')
        units.append(_conv(up, u, stride=2, bias=True))
    else:
        R._refuse(path, module, 'four reassemble levels required')
    if last.in_channels != first.out_channels:
        R._refuse(l, last, 'feature projection must read the intermediate width')
    units.append(_conv(last, l))
    def check(traced, calls):
        state = calls[0]
        readout = flow.calls_at(traced, f'decoder.projection_blocks.{index}')[-1]
        flow.require(state.args[0] is readout, f'{path}: reassemble must read its corresponding readout')
        for node in calls[1:]:
            flow.require(node.args[0] is state, f'{path}: reassemble must follow projection, upsample, feature projection')
            state = node
    kind = 'reassemble_up' if index < 2 else ('reassemble_identity' if index == 2 else 'reassemble_down')
    units[0] = _full_check(units[0], t, path, kind, check)
    return tuple(units)


def _residual(module, path, t):
    R._child(module, path, 'activation', R.RELU)
    activations = flow.calls_at(t, path + '.activation')
    if len(activations) != 2:
        R._refuse(path, module, 'residual block needs two ReLU calls')
    units = []
    channels = module.conv_1.in_channels
    for i in (1, 2):
        act = activations[i-1]
        units.append(_unit(path + f'.relu{i}', 'window_relu', {}, 'relu', owns=(path + '.activation',),
                            inputs=(path + '.activation',), select=lambda n, held=act: n is held))
        conv, c = R._child(module, path, f'conv_{i}', R.CONV2D)
        bn, b = R._child(module, path, f'batch_norm_{i}', R.BATCH_NORM)
        if (conv.in_channels, conv.out_channels) != (channels, channels):
            R._refuse(c, conv, 'residual convolution must preserve channels')
        units.extend((_conv(conv, c), _bn(bn, b, channels)))
    end = flow.calls_at(t, path, children=False)
    if len(end) != 1:
        R._refuse(path, module, 'one residual addition required')
    units.append(_binary(path + '.add', end[0]))
    def check(traced, calls):
        a, c, b, a2, c2, b2, add = calls
        flow.require(c.args[0] is a and b.args[0] is c and a2.args[0] is b and c2.args[0] is a2
                     and b2.args[0] is c2 and add.args[:2] == (b2, a.args[0]),
                     f'{path}: residual must add the original input after two ReLU/conv/BN branches')
    units[0] = _full_check(units[0], t, path, 'residual', check)
    return tuple(units)


def _fusion(module, path, t):
    index = int(path.rsplit('.', 1)[1])
    units = []
    for i in (1, 2):
        block, b = R._child(module, path, f'residual_conv_block{i}', DECODER + 'ResidualConvBlock')
        units.extend(_residual(block, b, t))
    direct = flow.calls_at(t, path, children=False)
    if index:
        units.append(_binary(path + '.add', direct[0]))
    resize = direct[-1]
    units.append(_resize(path, resize, (14, 7, None, None)[index]))
    project, p = R._child(module, path, 'project', R.CONV2D)
    channels = module.residual_conv_block1.conv_1.in_channels
    if project.in_channels != channels or project.out_channels != channels:
        R._refuse(p, project, 'fusion projection must preserve channels')
    units.append(_conv(project, p, kernel=1, bias=True))
    def check(traced, calls):
        first = flow.calls_at(traced, path + '.residual_conv_block1')
        second = flow.calls_at(traced, path + '.residual_conv_block2')
        expected = flow.calls_at(traced, f'decoder.reassemble_blocks.{3-index}.project_to_feature_dim')[-1]
        flow.require(first[0].args[0] is expected, f'{path}: fusion must read reassemble{3-index} (deepest first)')
        state = first[-1]
        if index:
            add = flow.calls_at(traced, path, children=False)[0]
            previous = flow.one(traced, f'decoder.fusion_blocks.{index-1}.project', 'aten.conv2d.default')
            flow.require(add.args[:2] == (state, previous), f'{path}: fusion must add the immediately previous fusion state')
            state = add
        flow.require(second[0].args[0] is state and resize.args[0] is second[-1]
                     and flow.one(traced, p, 'aten.conv2d.default').args[0] is resize,
                     f'{path}: fusion residual/resize/project state differs')
    units[0] = _full_check(units[0], t, path, 'fusion' if index else 'fusion_first', check)
    return tuple(units)


def _head(module, path, t):
    R._child(module, path, 'head', R.SEQUENTIAL)
    if len(module.head) != 5 or module.upsampling_factor != 2:
        R._refuse(path, module, 'DPT head requires five layers and upsampling_factor=2')
    activation, a = R._child(module, path, 'activation', R.ACTIVATION)
    R._child(activation, a, 'activation', R.IDENTITY)
    c0, p0 = R._child(module.head, path + '.head', '0', R.CONV2D)
    bn, b = R._child(module.head, path + '.head', '1', R.BATCH_NORM)
    R._child(module.head, path + '.head', '2', R.RELU)
    R._child(module.head, path + '.head', '3', R.DROPOUT)
    c4, p4 = R._child(module.head, path + '.head', '4', R.CONV2D)
    if c0.out_channels != c0.in_channels or c4.in_channels != c0.out_channels:
        R._refuse(path, module, 'head feature channels must match')
    units = [_conv(c0, p0), _bn(bn, b, c0.out_channels),
        _unit(path + '.head.2', 'window_relu', {}, 'head_relu', owns=(path + '.head.2', path + '.head.3')),
        _conv(c4, p4, kernel=1, bias=True)]
    resize = flow.calls_at(t, path, children=False)[0]
    units.append(_resize(path, resize, None))
    def check(traced, calls):
        flow.require(calls[0].args[0] is flow.one(traced, 'decoder.fusion_blocks.3.project', 'aten.conv2d.default'),
                     'head must read the last fusion')
        for previous, node in zip(calls, calls[1:]):
            flow.require(node.args[0] is previous, 'head must follow conv/BN/ReLU/dropout/conv/resize')
    units[0] = _full_check(units[0], t, path, 'head', check)
    return tuple(units)


def _dpt(module, path, t):
    if path or module.classification_head is not None:
        R._refuse(path, module, 'root DPT without an auxiliary classification head required')
    encoder, e = R._child(module, path, 'encoder', ENCODER)
    decoder, d = R._child(module, path, 'decoder', DECODER + 'DPTDecoder')
    head, h = R._child(module, path, 'segmentation_head', DECODER + 'DPTSegmentationHead')
    units = list(_encoder(encoder, e, t))
    if any(len(getattr(decoder, name)) != 4 for name in ('projection_blocks', 'reassemble_blocks', 'fusion_blocks')):
        R._refuse(d, decoder, 'four readout/reassemble/fusion levels required')
    for i in range(4):
        for name, cls, recognize in (('projection_blocks', 'ReadoutConcatBlock', _readout),
                                      ('reassemble_blocks', 'ReassembleBlock', _reassemble),
                                      ('fusion_blocks', 'FusionBlock', _fusion)):
            child, where = R._child(getattr(decoder, name), d + '.' + name, str(i), DECODER + cls)
            units.extend(recognize(child, where, t))
    units.extend(_head(head, h, t))
    return tuple(units)


_RECOGNIZERS = {DPT + '.model.DPT': _dpt, ENCODER: _encoder,
    DECODER + 'ReadoutConcatBlock': _readout, DECODER + 'ReassembleBlock': _reassemble,
    DECODER + 'FusionBlock': _fusion, DECODER + 'ResidualConvBlock': _residual,
    DECODER + 'DPTSegmentationHead': _head}
