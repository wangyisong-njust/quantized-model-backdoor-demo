import numpy as np
import torch
import torch.nn.intrinsic.qat as nniqat
from torch import fx, nn
from torch.fx import GraphModule, Node
from torch.nn import Module
from typing import List

from mqbench.utils.logger import logger
from mqbench.utils.hook import DataSaverHook, StopForwardException
from mqbench.utils import deepcopy_graphmodule, topology_order, getitem2node
from mqbench.utils.utils import _fix_succ_recursivly
from mqbench.utils.state import disable_all, enable_quantization
import mqbench.nn.intrinsic.qat as qnniqat

__all__ = ["ptq_reconstruction"]

_ADAROUND_SUPPORT_TYPE = (torch.nn.Conv2d, torch.nn.Linear)
_FUSED_TYPE = (
    nniqat.ConvBnReLU2d,
    nniqat.ConvBn2d,
    qnniqat.ConvFreezebn2d,
    qnniqat.ConvFreezebnReLU2d,
)
_WEIGHTS_MODULE_TYPE = (torch.nn.Conv2d, torch.nn.Linear)


def node2modules(name2modules, nodes):
    modules = {}
    for node in nodes:
        if node.target in name2modules:
            modules[node] = name2modules[node.target]
    return modules


def qnode2fpnode(quant_modules, fp32_modules):
    quant_named_nodes = {node.target: node for node in quant_modules}
    fp32_named_nodes = {node.target: node for node in fp32_modules}
    return {
        quant_named_nodes[key]: fp32_named_nodes[key]
        for key in quant_named_nodes
        if key in fp32_named_nodes
    }


def layer_has_weights(nodes, modules):
    for node in nodes:
        if node in modules and isinstance(modules[node], _WEIGHTS_MODULE_TYPE):
            return True
    return False


def lp_loss(pred, tgt, p=2.0):
    diff = (pred - tgt).abs().pow(p)
    if diff.ndim <= 1:
        return diff.mean()
    return diff.reshape(diff.shape[0], -1).sum(1).mean()


def to_device(data, device="cpu"):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {key: to_device(val, device) for key, val in data.items()}
    if isinstance(data, list):
        return [to_device(val, device) for val in data]
    if isinstance(data, tuple):
        return tuple(to_device(val, device) for val in data)
    return data


def tensor_detach(data):
    if isinstance(data, torch.Tensor):
        return data.detach()
    if isinstance(data, dict):
        return {key: tensor_detach(val) for key, val in data.items()}
    if isinstance(data, list):
        return [tensor_detach(val) for val in data]
    if isinstance(data, tuple):
        return tuple(tensor_detach(val) for val in data)
    return data


def _forward_model(model, batch, device):
    if isinstance(batch, dict):
        return model(**to_device(batch, device))
    return model(to_device(batch, device))


def save_inp_oup_data(
    model: GraphModule,
    inp_module: Module,
    oup_module: Module,
    cali_data: list,
    store_inp=True,
    store_oup=True,
    keep_gpu: bool = True,
):
    device = next(model.parameters()).device
    if store_inp:
        assert inp_module is not None
        inp_saver = DataSaverHook(
            store_input=store_inp,
            store_output=False,
            stop_forward=(not store_oup),
        )
        inp_handle = inp_module.register_forward_hook(inp_saver)
    if store_oup:
        assert oup_module is not None
        oup_saver = DataSaverHook(
            store_input=False,
            store_output=store_oup,
            stop_forward=True,
        )
        oup_handle = oup_module.register_forward_hook(oup_saver)

    cached = ([], [])
    with torch.no_grad():
        for batch in cali_data:
            try:
                _ = _forward_model(model, batch, device)
            except StopForwardException:
                pass
            if store_inp:
                if keep_gpu:
                    cached[0].append([tensor_detach(inp) for inp in inp_saver.input_store])
                else:
                    cached[0].append(
                        [to_device(tensor_detach(inp), "cpu") for inp in inp_saver.input_store]
                    )
            if store_oup:
                if keep_gpu:
                    cached[1].append(tensor_detach(oup_saver.output_store))
                else:
                    cached[1].append(to_device(tensor_detach(oup_saver.output_store), "cpu"))

    if store_inp:
        inp_handle.remove()
    if store_oup:
        oup_handle.remove()
    torch.cuda.empty_cache()
    return cached


class _NodeRecorder(fx.Interpreter):
    def __init__(self, module: GraphModule, target_names):
        super().__init__(module)
        self.target_names = set(target_names)
        self.records = {}

    def run_node(self, n):
        result = super().run_node(n)
        if n.name in self.target_names:
            self.records[n.name] = tensor_detach(result)
        return result


def save_nodes_data(model: GraphModule, target_nodes, cali_data: list, keep_gpu: bool = True):
    device = next(model.parameters()).device
    target_names = [node.name for node in target_nodes]
    cached = {name: [] for name in target_names}
    placeholder_names = [node.target for node in model.graph.nodes if node.op == "placeholder"]

    with torch.no_grad():
        for batch in cali_data:
            recorder = _NodeRecorder(model, target_names)
            if isinstance(batch, dict):
                batch = to_device(batch, device)
                values = [batch.get(name, None) for name in placeholder_names]
            else:
                batch = to_device(batch, device)
                if isinstance(batch, (list, tuple)):
                    values = list(batch)
                else:
                    values = [batch]
                if len(values) < len(placeholder_names):
                    values.extend([None] * (len(placeholder_names) - len(values)))
            recorder.run(*values)
            for name in target_names:
                value = recorder.records[name]
                if keep_gpu:
                    cached[name].append(value)
                else:
                    cached[name].append(to_device(value, "cpu"))

    return cached


class LossFunction:
    r"""EFRAP objective: activation preservation + error-guided flipped rounding."""

    def __init__(self, subgraph: Module, weight: float = 1.0, p: float = 2.0):
        self.subgraph = subgraph
        self.weight = weight
        self.p = p
        self.count = 0

    def __call__(self, pred, tgt):
        self.count += 1
        rec_loss = lp_loss(pred, tgt, p=self.p)
        round_loss = 0.0
        penalty_loss = 0.0
        b = 2

        for layer in self.subgraph.modules():
            if not isinstance(layer, _ADAROUND_SUPPORT_TYPE):
                continue
            round_vals = layer.weight_fake_quant.activate()
            round_loss += self.weight * (1 - ((round_vals - 0.5).abs() * 2).pow(b)).sum()

            expected_hv = layer.weight_fake_quant.get_reverse_round(layer.weight.data)
            hv = layer.weight_fake_quant.activate()
            rounding_error = layer.weight_fake_quant.get_error(layer.weight.data)
            cross_entropy = (
                -torch.log(hv + 1e-8) * expected_hv
                - torch.log(1 - hv + 1e-8) * (1 - expected_hv)
            )
            penalty_loss += (rounding_error * cross_entropy).sum()

        total_loss = rec_loss + penalty_loss + round_loss
        if self.count % 200 == 0:
            logger.info(
                "EFRAP loss: total=%.6f rec=%.6f penalty=%.6f round=%.6f count=%d",
                float(total_loss),
                float(rec_loss),
                float(penalty_loss),
                float(round_loss),
                self.count,
            )
        return total_loss


def _flatten_args(node):
    flattened_args = []
    if isinstance(node, dict):
        for value in node.values():
            flattened_args.extend(_flatten_args(value))
    elif isinstance(node, (tuple, list)):
        for value in node:
            flattened_args.extend(_flatten_args(value))
    else:
        flattened_args.append(node)
    return flattened_args


def find_used_times(nodes, target):
    return len([node for node in target.users if node in nodes])


def find_cur_node(layer_node_list):
    node_list = []
    used_later = []
    for idx, node in enumerate(layer_node_list):
        for later_node in layer_node_list[idx + 1:]:
            if node in _flatten_args(later_node.args):
                used_later.append(node)
                break
    not_used_later = [node for node in layer_node_list if node not in used_later]

    single_branch = {}
    for node in not_used_later:
        single_branch[node] = {node}
        queue = [node]
        while True:
            now_args = sum([_flatten_args(cur_node.args) for cur_node in queue], [])
            pending = [
                cur_node
                for cur_node in now_args
                if isinstance(cur_node, torch.fx.Node)
                and find_used_times(layer_node_list, cur_node) == 1
            ]
            single_branch[node] = single_branch[node].union(set(pending))
            if not pending:
                break
            queue = pending

    pivot = None
    for node in layer_node_list:
        if node.op in ("call_function", "call_method"):
            continue
        if node not in used_later:
            pivot = node
            break
    if pivot is None:
        return layer_node_list

    unwanted = set()
    for key, branch in single_branch.items():
        if key is pivot:
            continue
        unwanted = unwanted.union(branch)

    for node in layer_node_list:
        if node in unwanted:
            continue
        node_list.append(node)
        if node is pivot:
            return node_list
    return node_list


def extract_subgraph(orig_module, nodes: List[fx.Node], output: fx.Node, g2node: dict):
    new_graph = fx.Graph()
    env = {}
    inp_lst = []

    for node in nodes:
        for arg in _flatten_args(node.args):
            if isinstance(arg, torch.fx.Node) and arg not in nodes and arg not in inp_lst:
                inp_lst.append(node)
                arg_name = g2node[node].name if node in g2node else node.name
                env[node] = new_graph.placeholder(arg_name)
                break

    for node in nodes:
        if node in inp_lst:
            continue
        src_node = g2node[node] if node in g2node else node
        env[src_node] = new_graph.node_copy(src_node, lambda x: env[x])

    new_graph.output(env[output])
    new_graph.lint()
    return fx.GraphModule(orig_module, new_graph)


def find_num_nodes(nodes):
    return sum(1 for node in nodes if isinstance(node, Node))


def extract_layer(node, fp32_modules):
    layer_node_list = []
    cur_node = node
    is_next_block = False

    while True:
        layer_node_list.append(cur_node)
        stop = len(cur_node.users) == 0
        for user in cur_node.users:
            if user.target == "update":
                continue
            if user.target == "size":
                stop = True
            if user.op == "call_module" and isinstance(fp32_modules[user], _ADAROUND_SUPPORT_TYPE):
                stop = True
            if "add" in user.name and user.op in ("call_function", "call_method"):
                stop = True
            if user.op == "output":
                is_next_block = True
                stop = True
        if stop:
            break
        cur_node = list(cur_node.users.keys())[0]

    if find_num_nodes(cur_node.users) > 1:
        is_next_block = True
    return layer_node_list, is_next_block


def extract_block(input_nodes, fp32_modules, depth=0):
    if depth > 2:
        return []

    layer_node_list = []
    is_block = False
    cnt = {}
    queue, pending = [], []
    cur_node = None

    for input_node in input_nodes:
        for user in input_node.users:
            if user not in cnt:
                cnt[user] = find_num_nodes(user.args)
                if cnt[user] > 1:
                    is_block = True
                pending.append(user)
            cnt[user] -= 1
            if cnt[user] == 0:
                queue.append(user)
                pending.remove(user)

    while queue:
        cur_node = queue.pop(0)
        if cur_node.target == "update":
            continue
        if not pending and not queue:
            break
        layer_node_list.append(cur_node)
        for user in cur_node.users:
            if user not in cnt:
                cnt[user] = find_num_nodes(user.args)
                if cnt[user] > 1:
                    is_block = True
                pending.append(user)
            cnt[user] -= 1
            if cnt[user] == 0:
                queue.append(user)
                pending.remove(user)

    if not cur_node:
        return layer_node_list

    extra_nodes, is_next_block = extract_layer(cur_node, fp32_modules)
    if is_block or is_next_block:
        return layer_node_list + extra_nodes
    return layer_node_list + extra_nodes + extract_block([extra_nodes[-1]], fp32_modules, depth + 1)


def _hard_round_stats(layer):
    weight_quantizer = layer.weight_fake_quant
    nearest = 1 - weight_quantizer.get_reverse_round(layer.weight.data)
    hard = (weight_quantizer.alpha >= 0).float()
    return {
        "shape": list(layer.weight.shape),
        "flip_ratio": float((hard != nearest).float().mean().item()),
        "error_mean": float(weight_quantizer.get_error(layer.weight.data).mean().item()),
    }


def subgraph_reconstruction(subgraph, cached_inps, cached_oups, config, stats):
    device = next(subgraph.parameters()).device
    w_para = []
    a_para = []
    layer_stats = []

    for name, layer in subgraph.named_modules():
        if isinstance(layer, _ADAROUND_SUPPORT_TYPE):
            weight_quantizer = layer.weight_fake_quant
            weight_quantizer.init(layer.weight.data, config.round_mode)
            w_para.append(weight_quantizer.alpha)
        if isinstance(layer, torch.quantization.FakeQuantizeBase) and "post_act_fake_quantize" in name:
            if hasattr(config, "scale_lr"):
                a_para.append(layer.scale)
            layer.prob = config.prob

    if a_para:
        a_opt = torch.optim.Adam(a_para, lr=config.scale_lr)
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            a_opt,
            T_max=config.max_count,
            eta_min=0.0,
        )
    else:
        a_opt = None
        a_scheduler = None

    w_opt = torch.optim.Adam(w_para)
    loss_func = LossFunction(subgraph=subgraph, weight=config.weight)

    if config.prob < 1.0:
        batch_count = len(cached_inps[0][0])
        num_args = len(cached_inps[0])
    else:
        batch_count = len(cached_inps[0])
        num_args = len(cached_inps)

    for _ in range(config.max_count):
        idx = np.random.randint(0, batch_count)
        cur_args = []
        for arg_idx in range(num_args):
            if config.prob < 1.0:
                cur_inp = to_device(cached_inps[0][arg_idx][idx], device)
                cur_sym = to_device(cached_inps[1][arg_idx][idx], device)
                cur_inp = torch.where(torch.rand_like(cur_inp) < config.prob, cur_inp, cur_sym)
            else:
                cur_inp = to_device(cached_inps[arg_idx][idx], device)
            cur_args.append(cur_inp)
        cur_out = to_device(cached_oups[idx], device)

        if a_opt is not None:
            a_opt.zero_grad()
        w_opt.zero_grad()
        out_quant = subgraph(*tuple(cur_args))
        err = loss_func(out_quant, cur_out)
        err.backward()
        w_opt.step()
        if a_opt is not None:
            a_opt.step()
        if a_scheduler is not None:
            a_scheduler.step()

    torch.cuda.empty_cache()

    for name, layer in subgraph.named_modules():
        if isinstance(layer, _FUSED_TYPE):
            weight_quantizer = layer.weight_fake_quant
            scale_factor = layer.bn.weight / torch.sqrt(layer.bn.running_var + layer.bn.eps)
            merged_rounded_weight = weight_quantizer.get_hard_value(
                layer.weight.data * scale_factor.reshape([-1] + [1] * (len(layer.weight.shape) - 1))
            )
            layer.weight.data = merged_rounded_weight / scale_factor.reshape(
                [-1] + [1] * (len(merged_rounded_weight.shape) - 1)
            )
            weight_quantizer.adaround = False
        elif isinstance(layer, _ADAROUND_SUPPORT_TYPE):
            layer_stats.append({"name": name, "type": type(layer).__name__, **_hard_round_stats(layer)})
            layer.weight.data = layer.weight_fake_quant.get_hard_value(layer.weight.data)
            layer.weight_fake_quant.adaround = False
        if isinstance(layer, torch.quantization.FakeQuantizeBase) and "post_act_fake_quantize" in name:
            layer.prob = 1.0

    stats["subgraphs"].append({"layer_summaries": layer_stats})


def ptq_reconstruction(model: GraphModule, cali_data: list, config: dict):
    if not config.keep_gpu:
        cali_data = [to_device(inp, "cpu") for inp in cali_data]

    fp32_model = model
    fp32_model.eval()
    assert isinstance(fp32_model, torch.fx.GraphModule)

    quant_model = deepcopy_graphmodule(model)
    nodes = list(quant_model.graph.nodes)
    g2node = getitem2node(quant_model)
    fp32_modules = node2modules(dict(fp32_model.named_modules()), fp32_model.graph.nodes)
    quant_modules = node2modules(dict(quant_model.named_modules()), quant_model.graph.nodes)
    fp32_nodes_by_name = {node.name: node for node in fp32_model.graph.nodes}
    topology_order_by_node = topology_order(quant_model)
    qnode2fpnode_dict = qnode2fpnode(quant_modules, fp32_modules)

    quant_model.eval()
    disable_all(fp32_model)
    enable_quantization(quant_model)
    torch.cuda.empty_cache()

    checked_nodes = {}
    stats = {"optimized_targets": [], "subgraphs": []}
    max_layers = int(config.max_layers) if hasattr(config, "max_layers") else None

    for node in nodes:
        if max_layers is not None and len(stats["optimized_targets"]) >= max_layers:
            logger.info("Reached max_layers=%d, stop reconstruction early.", max_layers)
            break
        if node in checked_nodes:
            continue
        if node.op != "call_module" or not isinstance(quant_modules[node], _ADAROUND_SUPPORT_TYPE):
            continue

        logger.info("prepare %s reconstruction for %s", config.pattern, node)
        if config.pattern == "layer":
            layer_node_list, _ = extract_layer(node, quant_modules)
        elif config.pattern == "block":
            layer_node_list = extract_block(node.all_input_nodes, quant_modules)
        else:
            raise NotImplementedError(f"Unsupported reconstruction pattern: {config.pattern}")

        if not all(sub_node.target != "update" for sub_node in layer_node_list):
            remove_nodes = []
            for idx, sub_node in enumerate(layer_node_list):
                if sub_node.target != "update":
                    continue
                src = sub_node.args[0]
                remove = True
                for later_idx in range(idx + 1, len(layer_node_list)):
                    if src in _flatten_args(layer_node_list[later_idx].args):
                        remove = False
                        break
                if remove:
                    remove_nodes.append(sub_node)
            layer_node_list = [sub_node for sub_node in layer_node_list if sub_node not in remove_nodes]

        missing_inputs = []
        for sub_node in layer_node_list:
            for arg in _flatten_args(sub_node.args):
                if isinstance(arg, torch.fx.Node) and arg not in layer_node_list and arg not in missing_inputs:
                    missing_inputs.append(arg)
        layer_node_list.extend(missing_inputs)

        layer_node_list = [g2node[sub_node] if sub_node in g2node else sub_node for sub_node in layer_node_list]
        for sub_node in layer_node_list:
            src_nodes = [
                arg
                for arg in _flatten_args(sub_node.args)
                if not isinstance(arg, slice) and arg in g2node
            ]
            for arg in src_nodes:
                sub_node.args = _fix_succ_recursivly(sub_node.args, arg, g2node[arg])
        layer_node_list = sorted(layer_node_list, key=lambda cur_node: topology_order_by_node[cur_node])
        layer_node_list = find_cur_node(layer_node_list)

        if not layer_has_weights(layer_node_list, quant_modules):
            continue

        target_node = layer_node_list[-1]
        fp32_target_node = fp32_nodes_by_name.get(target_node.name)
        if fp32_target_node is None:
            logger.warning("Skip node %s because the fp32 node mapping is missing.", target_node)
            continue

        logger.info("reconstruct node list:")
        logger.info(layer_node_list)

        input_nodes = []
        fp32_input_nodes = []
        for sub_node in layer_node_list:
            if all(
                arg in layer_node_list
                for arg in _flatten_args(sub_node.args)
                if isinstance(arg, torch.fx.Node)
            ):
                continue
            fp32_sub_node = fp32_nodes_by_name.get(sub_node.name)
            if fp32_sub_node is None:
                logger.warning("Skip input node %s because the fp32 node mapping is missing.", sub_node)
                continue
            input_nodes.append(sub_node)
            fp32_input_nodes.append(fp32_sub_node)

        if not input_nodes:
            logger.warning("Skip node %s because cache building failed.", node)
            continue

        fp32_recorded = save_nodes_data(
            fp32_model,
            fp32_input_nodes + [fp32_target_node],
            cali_data,
            keep_gpu=config.keep_gpu,
        )
        quant_recorded = save_nodes_data(
            quant_model,
            input_nodes,
            cali_data,
            keep_gpu=config.keep_gpu,
        )

        fp32_all_inps = [fp32_recorded[sub_node.name] for sub_node in fp32_input_nodes]
        quant_all_inps = [quant_recorded[sub_node.name] for sub_node in input_nodes]
        cached_inps = (quant_all_inps, fp32_all_inps) if config.prob < 1.0 else quant_all_inps
        cached_oups = fp32_recorded[fp32_target_node.name]
        subgraph = extract_subgraph(
            quant_model,
            layer_node_list,
            target_node,
            g2node,
        )
        logger.info(subgraph.code)
        stats["optimized_targets"].append(target_node.name)
        subgraph_reconstruction(subgraph, cached_inps, cached_oups, config, stats)

        for sub_node in layer_node_list:
            checked_nodes[sub_node] = True

    disable_all(quant_model)
    for node in checked_nodes:
        if node.op == "call_module":
            enable_quantization(quant_modules[node])
            logger.info("set the node %s in quant", node.target)

    return quant_model, stats
