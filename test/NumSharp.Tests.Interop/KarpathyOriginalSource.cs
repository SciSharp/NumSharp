using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using Python.Runtime;

namespace NumSharp.Tests.Interop;

/// <summary>Hash-pinned original definitions, isolated from each Gist's downloads/IO/training main.</summary>
internal static class KarpathyOriginalSource
{
    private static readonly IReadOnlyDictionary<string, string> Pins = new Dictionary<string, string>
    {
        ["rnn"] = "98a0eaa61092f9541e883f3e8b6c02dae28b5658dbc5f692ffe0881be37a390a",
        ["pong"] = "cf764d11a0ebebb46d02c482c5a9c7c31081960d24b7c332757362735ad71a14",
        ["lstm"] = "137a213cfe6ec6eb7c73dc73c18f356dfd676e6e36bb66d2be6667ca6be7a4c4",
        ["microgpt"] = "d47d88c2fd432c8ebdc1048beab7f7eb64ea7e0e664e11b812d72a6d95ebccee",
        ["nes"] = "4144833dab064335631040b233d19aea2392f1e460a2cae1491a5def5d912f3b",
        ["walk"] = "cd5c47d7fbfa6d09fafbd6d6938c8a3217bf734c30ce9fb222b238aeb88c8451"
    };

    internal static string Read(string key)
    {
        if (!Pins.TryGetValue(key, out string? expected)) throw new ArgumentException("Unknown pinned Gist.", nameof(key));
        using var source = typeof(KarpathyOriginalSource).Assembly.GetManifestResourceStream($"KarpathyOriginal.{key}.py")
            ?? throw new FileNotFoundException($"Missing embedded pinned source: {key}");
        using var memory = new MemoryStream(); source.CopyTo(memory);
        byte[] bytes = memory.ToArray();
        if (!Convert.ToHexString(SHA256.HashData(bytes)).Equals(expected, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"Original Gist bytes differ from the approved pin: {key}");
        return new UTF8Encoding(false, true).GetString(bytes);
    }

    // Installs an isolated dict named 'original'. Set its model/data globals explicitly in a test.
    // This helper owns its GIL; callers never receive a PyObject needing later disposal.
    internal static void Load(PyModule scope, string key)
    {
        string source = Read(key);
        using (Py.GIL())
        {
            using var text = new PyString(source);
            using var name = new PyString(key);
            using var hash = new PyString(Pins[key]);
            scope.Set("_karpathy_raw_source", text); scope.Set("_karpathy_source_key", name); scope.Set("_karpathy_source_hash", hash);
            scope.Exec(Loader);
        }
    }

    private const string Loader = """
        import ast as _karpathy_ast
        import math as _karpathy_math
        import numpy as _karpathy_np
        _karpathy_names = {
            'rnn': ('lossFun', 'sample'),
            'pong': ('sigmoid', 'prepro', 'discount_rewards', 'policy_forward', 'policy_backward'),
            'lstm': ('LSTM', 'checkSequentialMatchesBatch', 'checkBatchGradient'),
            'microgpt': ('Value', 'linear', 'softmax', 'rmsnorm', 'gpt'),
            'nes': ('f',),
            'walk': ('slerp',)
        }[_karpathy_source_key]
        _karpathy_code = _karpathy_raw_source
        _karpathy_changes = []
        if _karpathy_source_key in ('rnn', 'pong', 'lstm'):
            # CPython3.12 is the pinned short-run host. Fail explicitly if the legacy-syntax
            # converter is absent; never silently skip a purported original-source test.
            import warnings as _karpathy_warnings
            with _karpathy_warnings.catch_warnings():
                _karpathy_warnings.simplefilter('ignore', DeprecationWarning)
                from lib2to3.refactor import RefactoringTool as _KarpathyRefactor
                _karpathy_code = str(_KarpathyRefactor([
                    'lib2to3.fixes.fix_print', 'lib2to3.fixes.fix_xrange', 'lib2to3.fixes.fix_repr'
                ]).refactor_string(_karpathy_code.rstrip()+'\n', '<pinned-karpathy>'))
            _karpathy_changes.append('Python2 print/xrange/backtick syntax converted; numerical formulas unchanged')
        if _karpathy_source_key == 'lstm':
            if _karpathy_code.count('WLSTM.shape[1]/4') != 1:
                raise AssertionError('Unexpected LSTM integer-width expression')
            _karpathy_code = _karpathy_code.replace('WLSTM.shape[1]/4', 'WLSTM.shape[1]//4')
            _karpathy_changes.append('Python2 integer hidden-size division preserved using //4')
        _karpathy_tree = _karpathy_ast.parse(_karpathy_code, filename='<pinned-karpathy>')
        _karpathy_selected = [node for node in _karpathy_tree.body
            if isinstance(node, (_karpathy_ast.FunctionDef, _karpathy_ast.ClassDef)) and node.name in _karpathy_names]
        if tuple(node.name for node in _karpathy_selected) != _karpathy_names:
            raise AssertionError('Original definition inventory differs from the reviewed source')
        if _karpathy_source_key == 'walk':
            # Source bug already documented by the port: NumPy inputs never assign this local.
            # Initialize only that dispatch flag; no numerical expression is changed.
            _karpathy_selected[0].body.insert(1, _karpathy_ast.Assign(
                targets=[_karpathy_ast.Name(id='inputs_are_torch',ctx=_karpathy_ast.Store())], value=_karpathy_ast.Constant(False)))
            _karpathy_changes.append('Initialize inputs_are_torch=False for the otherwise broken original NumPy branch')
        if _karpathy_source_key == 'pong':
            class _KarpathyFloatAlias(_karpathy_ast.NodeTransformer):
                def visit_Attribute(self, node):
                    self.generic_visit(node)
                    if isinstance(node.value, _karpathy_ast.Name) and node.value.id == 'np' and node.attr == 'float':
                        node.attr = 'float64'
                    return node
            _karpathy_selected = [_KarpathyFloatAlias().visit(node) for node in _karpathy_selected]
            _karpathy_changes.append('Removed np.float alias is np.float64; NumPy module not monkey-patched')
        _karpathy_module = _karpathy_ast.fix_missing_locations(_karpathy_ast.Module(body=_karpathy_selected, type_ignores=[]))
        original = {'np': _karpathy_np, 'math': _karpathy_math, '__name__': '_pinned_gist_definitions',
                    '__source_sha256__': _karpathy_source_hash, '__loaded_names__': _karpathy_names,
                    '__compatibility__': tuple(_karpathy_changes)}
        exec(compile(_karpathy_module, '<pinned-karpathy-definitions>', 'exec'), original)
        # No source imports, assignments, loops, dataset reads/downloads or __main__ blocks execute.
        del _karpathy_raw_source
        """;
}
