"""Forward calls from timm 1.0.30 ViT, exported with torch 2.14.0+cpu.

Captured from an unfused, evaluation-mode VisionTransformer at 224/16, width
128, heads 2, depth 2, six bands, no classifier or pooling. Tests retrace both
this reduced twin and ViT-B (768/12/12). These are code fingerprints, not
numerical evidence. Integer layouts and Q/K/V roles also have dataflow checks.
"""

SOURCE = "timm 1.0.30 ViT"
TRACED_WITH = {"timm": "1.0.30", "torch": "2.14.0+cpu"}

FINGERPRINTS = {
    'position': (
        'aten.expand.default(implicit=False)',
        'aten.cat.default(dim=1)',
        'aten.add.Tensor',
        'aten.dropout.default(train=False)',
    ),
    'patch': (
        'aten.conv2d.default',
        'aten.flatten.using_ints(start_dim=2, end_dim=-1)',
        'aten.transpose.int(dim0=1, dim1=2)',
    ),
    'block': (
        'aten.layer_norm.default(cudnn_enable=False)',
        'aten.linear.default',
        'aten.reshape.default',
        'aten.permute.default(dims=[2, 0, 3, 1, 4])',
        'aten.unbind.int(dim=0)',
        'operator.getitem',
        'operator.getitem',
        'operator.getitem',
        'aten.mul.Tensor',
        'aten.transpose.int(dim0=-2, dim1=-1)',
        'aten.matmul.default',
        'aten.softmax.int(dim=-1)',
        'aten.dropout.default(train=False)',
        'aten.matmul.default',
        'aten.transpose.int(dim0=1, dim1=2)',
        'aten.reshape.default',
        'aten.linear.default',
        'aten.dropout.default(train=False)',
        'aten.add.Tensor',
        'aten.layer_norm.default(cudnn_enable=False)',
        'aten.linear.default',
        "aten.gelu.default(approximate='none')",
        'aten.dropout.default(train=False)',
        'aten.linear.default',
        'aten.dropout.default(train=False)',
        'aten.add.Tensor',
    ),
    'attention': (
        'aten.linear.default',
        'aten.reshape.default',
        'aten.permute.default(dims=[2, 0, 3, 1, 4])',
        'aten.unbind.int(dim=0)',
        'operator.getitem',
        'operator.getitem',
        'operator.getitem',
        'aten.mul.Tensor',
        'aten.transpose.int(dim0=-2, dim1=-1)',
        'aten.matmul.default',
        'aten.softmax.int(dim=-1)',
        'aten.dropout.default(train=False)',
        'aten.matmul.default',
        'aten.transpose.int(dim0=1, dim1=2)',
        'aten.reshape.default',
        'aten.linear.default',
        'aten.dropout.default(train=False)',
    ),
    'attention_core': (
        'aten.reshape.default',
        'aten.permute.default(dims=[2, 0, 3, 1, 4])',
        'aten.unbind.int(dim=0)',
        'operator.getitem',
        'operator.getitem',
        'operator.getitem',
        'aten.mul.Tensor',
        'aten.transpose.int(dim0=-2, dim1=-1)',
        'aten.matmul.default',
        'aten.softmax.int(dim=-1)',
        'aten.dropout.default(train=False)',
        'aten.matmul.default',
        'aten.transpose.int(dim0=1, dim1=2)',
        'aten.reshape.default',
    ),
    'mlp': (
        'aten.linear.default',
        "aten.gelu.default(approximate='none')",
        'aten.dropout.default(train=False)',
        'aten.linear.default',
        'aten.dropout.default(train=False)',
    ),
    'norm': (
        'aten.layer_norm.default(cudnn_enable=False)',
    ),
    'final_norm': (
        'aten.layer_norm.default(cudnn_enable=False)',
        'aten.dropout.default(train=False)',
    ),
    'linear': (
        'aten.linear.default',
    ),
    'projection': (
        'aten.linear.default',
        'aten.dropout.default(train=False)',
    ),
    'gelu': (
        "aten.gelu.default(approximate='none')",
        'aten.dropout.default(train=False)',
    ),
    'fc2': (
        'aten.linear.default',
        'aten.dropout.default(train=False)',
    ),
    'residual': (
        'aten.add.Tensor',
    ),
}
