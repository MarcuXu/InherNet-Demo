import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import resnet50, resnet18
# from models.resnet import resnet50, resnet18
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import copy
from collections import OrderedDict

# import os
# os.environ['TORCH_HOME'] = '/root/models'

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_dataset(dataset_name, root='./data', train=True, download=True):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=download, transform=transform_train if train else transform_test)
    elif dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root=root, train=train, download=download, transform=transform_train if train else transform_test)
    elif dataset_name == 'caltech101':
        dataset = torchvision.datasets.Caltech101(root=root, download=download, transform=transform_train if train else transform_test)
    elif dataset_name == 'oxford_pets':
        dataset = torchvision.datasets.OxfordIIITPet(root=root, download=download, transform=transform_train if train else transform_test)
    elif dataset_name == 'stanford_cars':
        dataset = torchvision.datasets.StanfordCars(root=root, download=download, transform=transform_train if train else transform_test)
    elif dataset_name == 'oxford_flowers':
        dataset = torchvision.datasets.Flowers102(root=root, download=download, transform=transform_train if train else transform_test)
    elif dataset_name == 'food101':
        dataset = torchvision.datasets.Food101(root=root, download=download, transform=transform_train if train else transform_test)
    elif dataset_name == 'fgvc_aircraft':
        dataset = torchvision.datasets.FGVCAircraft(root=root, download=download, transform=transform_train if train else transform_test)
    elif dataset_name == 'sun397':
        dataset = torchvision.datasets.SUN397(root=root, download=download, transform=transform_train if train else transform_test)
    elif dataset_name == 'dtd':
        dataset = torchvision.datasets.DTD(root=root, download=download, transform=transform_train if train else transform_test)
    elif dataset_name == 'eurosat':
        dataset = torchvision.datasets.EuroSAT(root=root, download=download, transform=transform_train if train else transform_test)
    elif dataset_name == 'ucf101':
        dataset = torchvision.datasets.UCF101(root=root, download=download, transform=transform_train if train else transform_test)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset

def get_dataloader(dataset_name, batch_size=256, shuffle=True, root='./data', download=True):
    train_set = get_dataset(dataset_name, root=root, train=True, download=download)
    test_set = get_dataset(dataset_name, root=root, train=False, download=download)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class SumConv2d(nn.Module):
        def __init__(self, conv2_list):
            super(SumConv2d, self).__init__()
            self.conv2_list = conv2_list

        def forward(self, x):
            out = sum(conv(x) for conv in self.conv2_list)
            return out


class GatedSumLinear(nn.Module):
    def __init__(self, linear_list, input_dim, head_num):
        super(GatedSumLinear, self).__init__()
        self.linear_list = linear_list
        self.head_num = head_num
        self.gate = nn.Linear(input_dim, head_num)

    def forward(self, x):
        gating_scores = self.gate(x)
        gating_weights = F.softmax(gating_scores, dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.linear_list], dim=-1)
        out = torch.sum(gating_weights.unsqueeze(2) * expert_outputs, dim=-1)
        return out

class GatedSumConv2d(nn.Module):
    def __init__(self, conv2_list, input_dim, head_num):
        super(GatedSumConv2d, self).__init__()
        self.conv2_list = conv2_list
        self.head_num = head_num
        self.gate = nn.Linear(input_dim, head_num)

    def forward(self, x):
        batch_size = x.shape[0]
        y = torch.mean(x, dim=(2, 3)) 
        gating_scores = self.gate(y)
        gating_weights = F.softmax(gating_scores, dim=-1)
        expert_outputs = torch.stack([conv(x) for conv in self.conv2_list], dim=-1)
        gating_weights = gating_weights.view(batch_size, 1, 1, 1, self.head_num)
        out = torch.sum(gating_weights * expert_outputs, dim=-1) 

        return out


class DecoupledGatedSVDConv2d(nn.Module):
    def __init__(self, conv1, conv2_list, gate_input_dim, head_num, use_uncompressed_gate=False):
        super().__init__()
        self.conv1 = conv1
        self.conv2_list = conv2_list
        self.head_num = head_num
        self.use_uncompressed_gate = use_uncompressed_gate
        self.gate = nn.Linear(gate_input_dim, head_num)

    def forward(self, x):
        batch_size = x.shape[0]
        compressed = self.conv1(x)
        expert_outputs = torch.stack([conv(compressed) for conv in self.conv2_list], dim=-1)

        if self.use_uncompressed_gate:
            gate_feat = torch.mean(x, dim=(2, 3))
        else:
            gate_feat = torch.mean(compressed, dim=(2, 3))

        gating_scores = self.gate(gate_feat)
        gating_weights = F.softmax(gating_scores, dim=-1).view(batch_size, 1, 1, 1, self.head_num)
        out = torch.sum(gating_weights * expert_outputs, dim=-1)
        return out

class ResNet18SVD(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = resnet18(num_classes=num_classes)

    def replace_linear_with_svd(self, module, rank):
        if isinstance(module, nn.Linear):
            in_dim, out_dim = module.in_features, module.out_features
            weight = module.weight.data
            bias = module.bias.data.clone() if module.bias is not None else None

            U, S, V = torch.svd(weight)
            if rank > S.numel():
                return module
            r = min(rank, S.numel())

            U_trunc = U[:, :r]
            S_trunc = S[:r]
            V_trunc = V[:, :r]

            B = U_trunc @ torch.diag(S_trunc)
            A = V_trunc.t()

            svd_layer = nn.Sequential(
                nn.Linear(in_dim, r, bias=False),
                nn.Linear(r, out_dim, bias=True)
            )
            svd_layer[0].weight.data = A
            svd_layer[1].weight.data = B
            svd_layer[1].bias.data = bias
            return svd_layer
        return module

    def replace_conv_with_svd(self, module, rank, head_num):
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            C_out, C_in, K_h, K_w = weight.shape
            weight_flat = weight.view(C_out, -1)
            U, S, V = torch.svd(weight_flat)
            if rank >= S.numel():
                return module
            r = min(rank, S.numel())

            U_trunc = U[:, :r]
            S_trunc = S[:r]
            V_trunc = V[:, :r]

            conv1 = nn.Conv2d(C_in, r, kernel_size=(K_h, K_w), stride=module.stride, padding=module.padding, bias=False)
            conv1.weight.data = V_trunc.t().view(r, C_in, K_h, K_w)

            conv2_list = nn.ModuleList()
            for _ in range(head_num):
                conv2 = nn.Conv2d(r, C_out, kernel_size=1, stride=1, padding=0, bias=True)
                conv2.weight.data = (U_trunc @ torch.diag(S_trunc) / head_num).view(C_out, r, 1, 1)
                if module.bias is not None:
                    conv2.bias.data = module.bias.data.clone() / head_num
                conv2_list.append(conv2)

            # return nn.Sequential(conv1, SumConv2d(conv2_list))
            return nn.Sequential(conv1, GatedSumConv2d(conv2_list, r, head_num))
        elif isinstance(module, nn.Sequential) and len(module) == 2 and isinstance(module[0], nn.Conv2d):
            conv = module[0]
            bn = module[1]
            svd_conv = self.replace_conv_with_svd(conv, rank, head_num)
            return nn.Sequential(svd_conv, bn)
        return module

    def apply_svd(self, rank, head_num):
        for name, module in self.resnet.named_children():
            if isinstance(module, nn.Conv2d):
                setattr(self.resnet, name, self.replace_conv_with_svd(module, rank, head_num))
            elif isinstance(module, nn.Sequential) or isinstance(module, nn.Module):
                for sub_name, sub_module in module.named_children():
                    if isinstance(sub_module, nn.Conv2d):
                        setattr(module, sub_name, self.replace_conv_with_svd(sub_module, rank, head_num))
                    else:
                        self.apply_svd_recursive(sub_module, rank, head_num)

    def apply_svd_recursive(self, module, rank, head_num):
        for name, sub_module in module.named_children():
            if isinstance(sub_module, nn.Conv2d):
                setattr(module, name, self.replace_conv_with_svd(sub_module, rank, head_num))
            elif isinstance(sub_module, nn.Module):
                self.apply_svd_recursive(sub_module, rank, head_num)
        

    def initialize_weights_kaiming(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def initialize_weights_gaussian(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, x):
        return self.resnet(x)


class ResNet18HeteroSVD(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = resnet18(num_classes=num_classes)

    def _collect_conv_layers(self):
        conv_layers = OrderedDict()
        for name, module in self.resnet.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers[name] = module
        return conv_layers

    def _get_parent_module(self, module_name):
        if '.' not in module_name:
            return self.resnet, module_name
        parent_name, child_name = module_name.rsplit('.', 1)
        parent = self.resnet.get_submodule(parent_name)
        return parent, child_name

    def _estimate_input_covariances(self, calib_loader, max_batches=5, eps=1e-5):
        conv_layers = self._collect_conv_layers()
        stats = {
            name: {
                'sum': None,
                'sum_outer': None,
                'count': 0,
            }
            for name in conv_layers.keys()
        }

        handles = []

        def make_hook(layer_name):
            def hook(_, layer_input, __):
                x = layer_input[0].detach()
                feat = x.mean(dim=(2, 3))
                sum_vec = feat.sum(dim=0)
                sum_outer = feat.t().matmul(feat)
                if stats[layer_name]['sum'] is None:
                    stats[layer_name]['sum'] = sum_vec
                    stats[layer_name]['sum_outer'] = sum_outer
                else:
                    stats[layer_name]['sum'] += sum_vec
                    stats[layer_name]['sum_outer'] += sum_outer
                stats[layer_name]['count'] += feat.shape[0]
            return hook

        for name, module in conv_layers.items():
            handles.append(module.register_forward_hook(make_hook(name)))

        self.eval()
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(calib_loader):
                if batch_idx >= max_batches:
                    break
                inputs = inputs.to(device)
                _ = self.resnet(inputs)

        for handle in handles:
            handle.remove()

        covariances = {}
        for name, module in conv_layers.items():
            c_in = module.in_channels
            st = stats[name]
            if st['count'] == 0:
                covariances[name] = torch.eye(c_in, device=module.weight.device, dtype=module.weight.dtype)
                continue
            mean = st['sum'] / st['count']
            exx = st['sum_outer'] / st['count']
            cov = exx - torch.outer(mean, mean)
            cov = cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
            covariances[name] = cov.to(device=module.weight.device, dtype=module.weight.dtype)
        return covariances

    def _whiten_weight_channelwise(self, weight, chol_c):
        c_out, c_in, k_h, k_w = weight.shape
        weight_perm = weight.permute(0, 2, 3, 1).reshape(-1, c_in)
        whitened = weight_perm.matmul(chol_c)
        return whitened.view(c_out, k_h, k_w, c_in).permute(0, 3, 1, 2).contiguous()

    def _compute_spectral_entropies(self, covariances):
        conv_layers = self._collect_conv_layers()
        entropies = {}
        max_ranks = {}
        svd_cache = {}
        for name, module in conv_layers.items():
            weight = module.weight.data
            chol_c = torch.linalg.cholesky(covariances[name])
            whitened_weight = self._whiten_weight_channelwise(weight, chol_c)
            weight_flat = whitened_weight.view(whitened_weight.shape[0], -1)
            _, s, v_h = torch.linalg.svd(weight_flat, full_matrices=False)
            sigma_sum = s.sum().clamp_min(1e-12)
            p = (s / sigma_sum).clamp_min(1e-12)
            entropy = -(p * torch.log(p)).sum().item()
            entropies[name] = entropy
            max_ranks[name] = s.numel()
            svd_cache[name] = {
                's': s,
                'v_h': v_h,
            }
        return entropies, max_ranks, svd_cache

    def _allocate_ranks_by_entropy(self, entropies, max_ranks, budget_ratio=0.35, min_rank=1):
        layer_names = list(entropies.keys())
        total_max = sum(max_ranks[name] for name in layer_names)
        budget = int(max(len(layer_names) * min_rank, round(total_max * budget_ratio)))
        budget = min(budget, total_max)

        entropy_sum = sum(entropies[name] for name in layer_names)
        if entropy_sum <= 0:
            raw = {name: budget / len(layer_names) for name in layer_names}
        else:
            raw = {name: (entropies[name] / entropy_sum) * budget for name in layer_names}

        ranks = {}
        for name in layer_names:
            ranks[name] = int(round(raw[name]))
            ranks[name] = max(min_rank, ranks[name])
            ranks[name] = min(max_ranks[name], ranks[name])

        current_total = sum(ranks.values())
        if current_total < budget:
            order = sorted(layer_names, key=lambda n: raw[n] - int(raw[n]), reverse=True)
            idx = 0
            while current_total < budget and order:
                name = order[idx % len(order)]
                if ranks[name] < max_ranks[name]:
                    ranks[name] += 1
                    current_total += 1
                idx += 1
                if idx > len(order) * (max(max_ranks.values()) + 1):
                    break
        elif current_total > budget:
            order = sorted(layer_names, key=lambda n: raw[n] - int(raw[n]))
            idx = 0
            while current_total > budget and order:
                name = order[idx % len(order)]
                if ranks[name] > min_rank:
                    ranks[name] -= 1
                    current_total -= 1
                idx += 1
                if idx > len(order) * (max(max_ranks.values()) + 1):
                    break
        return ranks

    def _replace_conv_with_hetero_svd(self, module, rank, head_num, cov, r_min):
        if not isinstance(module, nn.Conv2d):
            return module

        weight = module.weight.data
        c_out, c_in, k_h, k_w = weight.shape

        chol_c = torch.linalg.cholesky(cov)
        whitened_weight = self._whiten_weight_channelwise(weight, chol_c)
        weight_flat = whitened_weight.view(c_out, -1)
        u, s, v_h = torch.linalg.svd(weight_flat, full_matrices=False)

        rank = max(1, min(rank, s.numel()))
        u_trunc = u[:, :rank]
        s_trunc = s[:rank]
        v_h_trunc = v_h[:rank, :]

        whiten_inv = torch.linalg.inv(chol_c)
        v_4d = v_h_trunc.view(rank, c_in, k_h, k_w)
        v_perm = v_4d.permute(0, 2, 3, 1).reshape(-1, c_in)
        v_unwhiten = v_perm.matmul(whiten_inv)
        conv1_weight = v_unwhiten.view(rank, k_h, k_w, c_in).permute(0, 3, 1, 2).contiguous()

        conv1 = nn.Conv2d(c_in, rank, kernel_size=(k_h, k_w), stride=module.stride, padding=module.padding, bias=False)
        conv1.weight.data = conv1_weight

        conv2_list = nn.ModuleList()
        for _ in range(head_num):
            conv2 = nn.Conv2d(rank, c_out, kernel_size=1, stride=1, padding=0, bias=True)
            conv2.weight.data = (u_trunc @ torch.diag(s_trunc) / head_num).view(c_out, rank, 1, 1)
            if module.bias is not None:
                conv2.bias.data = module.bias.data.clone() / head_num
            conv2_list.append(conv2)

        use_uncompressed_gate = rank < r_min
        gate_input_dim = c_in if use_uncompressed_gate else rank
        return DecoupledGatedSVDConv2d(conv1, conv2_list, gate_input_dim, head_num, use_uncompressed_gate)

    def apply_hetero_svd(self, calib_loader, head_num=2, budget_ratio=0.35, r_min=4, max_calib_batches=5):
        covariances = self._estimate_input_covariances(calib_loader, max_batches=max_calib_batches)
        entropies, max_ranks, _ = self._compute_spectral_entropies(covariances)
        rank_map = self._allocate_ranks_by_entropy(entropies, max_ranks, budget_ratio=budget_ratio, min_rank=1)

        conv_layers = self._collect_conv_layers()
        for name, module in conv_layers.items():
            parent, child_name = self._get_parent_module(name)
            replaced = self._replace_conv_with_hetero_svd(
                module,
                rank=rank_map[name],
                head_num=head_num,
                cov=covariances[name],
                r_min=r_min,
            )
            setattr(parent, child_name, replaced)

        return rank_map

    def initialize_weights_kaiming(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def initialize_weights_gaussian(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, x):
        return self.resnet(x)

class TeacherModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = resnet50(num_classes=num_classes)

    def forward(self, x):
        return self.resnet(x)

class StudentModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = resnet18(num_classes=num_classes)

    def forward(self, x):
        return self.resnet(x)

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=100):
    model.train()
    train_losses = []
    test_accuracies = []
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        # progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)
        test_accuracy = evaluate_model(model, test_loader, criterion)
        train_losses.append(total_loss / len(train_loader))
        test_accuracies.append(test_accuracy)
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%")
    return train_losses, test_accuracies

def evaluate_model(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train_distillation(teacher_model, student_model, train_loader, test_loader, optimizer, temp=7, alpha=0.3, epochs=100):
    student_model.train()
    teacher_model.eval()
    train_losses = []
    test_accuracies = []
    hard_loss = nn.CrossEntropyLoss()
    soft_loss = nn.KLDivLoss(reduction='batchmean')

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        # progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
            student_outputs = student_model(inputs)
            student_loss = hard_loss(student_outputs, labels)
            distillation_loss = soft_loss(
                F.log_softmax(student_outputs / temp, dim=1),
                F.softmax(teacher_outputs / temp, dim=1)
            )
            loss = alpha * student_loss + (1 - alpha) * temp * temp * distillation_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(student_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)
        test_accuracy = evaluate_model(student_model, test_loader, criterion)
        train_losses.append(total_loss / len(train_loader))
        test_accuracies.append(test_accuracy)
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%")
    return train_losses, test_accuracies

def train_svd_model(model, rank, head_num, train_loader, test_loader, criterion, optimizer, init_method=None, epochs=100, teacher_model=None):
    model_svd = copy.deepcopy(model)
    model_svd.apply_svd(rank, head_num)
    if init_method == "distillation":
        model_svd = model_svd.to(device)
        optimizer_student = optim.Adam(model_svd.parameters(), lr=0.001)
        train_losses, test_accuracies = train_distillation(teacher_model, model_svd, train_loader, test_loader, optimizer_student, epochs=epochs)
        return train_losses, test_accuracies
    if init_method == 'kaiming':
        model_svd.initialize_weights_kaiming()
    elif init_method == 'gaussian':
        model_svd.initialize_weights_gaussian()
    model_svd = model_svd.to(device)
    optimizer_svd = optim.Adam(model_svd.parameters(), lr=0.001)
    train_losses, test_accuracies = train_model(model_svd, train_loader, test_loader, criterion, optimizer_svd, epochs=epochs)
    return train_losses, test_accuracies


def train_hetero_svd_model(model, train_loader, test_loader, criterion, head_num=2, budget_ratio=0.35, r_min=4, max_calib_batches=5, init_method=None, epochs=100, teacher_model=None):
    model_svd = copy.deepcopy(model)
    rank_map = model_svd.apply_hetero_svd(
        calib_loader=train_loader,
        head_num=head_num,
        budget_ratio=budget_ratio,
        r_min=r_min,
        max_calib_batches=max_calib_batches,
    )

    if init_method == "distillation":
        model_svd = model_svd.to(device)
        optimizer_student = optim.Adam(model_svd.parameters(), lr=0.001)
        train_losses, test_accuracies = train_distillation(teacher_model, model_svd, train_loader, test_loader, optimizer_student, epochs=epochs)
        return train_losses, test_accuracies, rank_map

    if init_method == 'kaiming':
        model_svd.initialize_weights_kaiming()
    elif init_method == 'gaussian':
        model_svd.initialize_weights_gaussian()

    model_svd = model_svd.to(device)
    optimizer_svd = optim.Adam(model_svd.parameters(), lr=0.001)
    train_losses, test_accuracies = train_model(model_svd, train_loader, test_loader, criterion, optimizer_svd, epochs=epochs)
    return train_losses, test_accuracies, rank_map

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":

    dataset_name = 'cifar10'
    epochs_num = 5
    train_loader, test_loader = get_dataloader(dataset_name)

    if dataset_name == 'cifar10':
        num_classes = 10
    elif dataset_name == 'cifar100':
        num_classes = 100
    elif dataset_name == 'caltech101':
        num_classes = 101
    elif dataset_name == 'oxford_pets':
        num_classes = 37
    elif dataset_name == 'stanford_cars':
        num_classes = 196
    elif dataset_name == 'oxford_flowers':
        num_classes = 102
    elif dataset_name == 'food101':
        num_classes = 101
    elif dataset_name == 'fgvc_aircraft':
        num_classes = 100
    elif dataset_name == 'sun397':
        num_classes = 397
    elif dataset_name == 'dtd':
        num_classes = 47
    elif dataset_name == 'eurosat':
        num_classes = 10
    elif dataset_name == 'ucf101':
        num_classes = 101
    teacher_model = TeacherModel(num_classes=num_classes).to(device)
    student_model = StudentModel(num_classes=num_classes).to(device)
    model = ResNet18SVD(num_classes=num_classes).to(device)
    hetero_model = ResNet18HeteroSVD(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=0.001)
    optimizer_student = optim.Adam(student_model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training Teacher Model (ResNet-50)...")
    train_losses_teacher, test_accuracies_teacher = train_model(teacher_model, train_loader, test_loader, criterion, optimizer_teacher, epochs=epochs_num)

    print("Training Student Model (ResNet-18) with Distillation...")
    train_losses_distill, test_accuracies_distill = train_distillation(teacher_model, student_model, train_loader, test_loader, optimizer_student, epochs=epochs_num)

    print("Training Original ResNet-18...")
    train_losses_original, test_accuracies_original = train_model(model, train_loader, test_loader, criterion, optimizer, epochs=epochs_num)

    ranks = [32]
    head_nums = [1, 2]
    init_methods = [None, 'distillation']
    results = {}

    for rank in ranks:
        for head_num in head_nums:
            for init_method in init_methods:
                key = f"rank_{rank}_head_{head_num}_{init_method if init_method else 'default'}"
                print(f"Training SVD-ResNet-18 with {key}...")
                train_losses, test_accuracies = train_svd_model(model, rank, head_num, train_loader, test_loader, criterion, optimizer, init_method=init_method, epochs=epochs_num, teacher_model=teacher_model)
                results[key] = (train_losses, test_accuracies)

    hetero_results = {}
    hetero_configs = [
        {'head_num': 1, 'budget_ratio': 0.35, 'r_min': 32, 'init_method': None},
        {'head_num': 1, 'budget_ratio': 0.35, 'r_min': 32, 'init_method': 'distillation'},
        {'head_num': 2, 'budget_ratio': 0.35, 'r_min': 32, 'init_method': None},
        {'head_num': 2, 'budget_ratio': 0.35, 'r_min': 32, 'init_method': 'distillation'},
        {'head_num': 3, 'budget_ratio': 0.35, 'r_min': 32, 'init_method': None},
        {'head_num': 3, 'budget_ratio': 0.35, 'r_min': 32, 'init_method': 'distillation'},
    ]

    for cfg in hetero_configs:
        key = f"hetero_budget_{cfg['budget_ratio']}_head_{cfg['head_num']}_{cfg['init_method'] if cfg['init_method'] else 'default'}"
        print(f"Training Heterogeneous SVD-ResNet-18 with {key}...")
        train_losses, test_accuracies, rank_map = train_hetero_svd_model(
            hetero_model,
            train_loader,
            test_loader,
            criterion,
            head_num=cfg['head_num'],
            budget_ratio=cfg['budget_ratio'],
            r_min=cfg['r_min'],
            max_calib_batches=5,
            init_method=cfg['init_method'],
            epochs=epochs_num,
            teacher_model=teacher_model,
        )
        rank_stats = f"min_rank={min(rank_map.values())}, max_rank={max(rank_map.values())}, avg_rank={sum(rank_map.values()) / len(rank_map):.2f}"
        print(f"{key} rank allocation: {rank_stats}")
        hetero_results[key] = (train_losses, test_accuracies, rank_map, cfg)


    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'legend.fontsize': 9,
        'lines.linewidth': 2.0,
    })

    linestyles = ['-', '--', '-.', ':']
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=160)

    ax_loss = axes[0]
    ax_loss.plot(train_losses_original, label='Original ResNet-18', color='black', linewidth=2.6)
    for i, (key, (train_losses, _)) in enumerate(results.items()):
        ax_loss.plot(
            train_losses,
            label=f'SVD-ResNet-18 ({key})',
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            alpha=0.95,
        )
    for i, (key, (train_losses, _, _, _)) in enumerate(hetero_results.items()):
        ax_loss.plot(
            train_losses,
            label=f'Hetero-InherNet ({key})',
            color=colors[(i + len(results)) % len(colors)],
            linestyle=linestyles[(i + len(results)) % len(linestyles)],
            alpha=0.95,
        )
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Training Loss')
    ax_loss.set_title('Training Loss Comparison')
    ax_loss.grid(True, linestyle='--', alpha=0.35)

    original_params = count_parameters(model)
    student_params = count_parameters(student_model)

    ax_acc = axes[1]
    ax_acc.plot(
        test_accuracies_original,
        label=f'Original ResNet-18 ({original_params:,} params)',
        color='black',
        linewidth=2.6,
    )
    ax_acc.plot(
        test_accuracies_distill,
        label=f'Student Distill ({student_params:,} params)',
        color='dimgray',
        linestyle='--',
        linewidth=2.2,
    )

    for i, (key, (_, test_accuracies)) in enumerate(results.items()):
        svd_model = copy.deepcopy(model)
        rank, head_num, init_method = key.split("_")[1], key.split("_")[3], key.split("_")[4]
        svd_model.apply_svd(int(rank), int(head_num))
        svd_params = count_parameters(svd_model)
        ax_acc.plot(
            test_accuracies,
            label=f'SVD-ResNet-18 ({key}, {svd_params:,} params)',
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            alpha=0.95,
        )

    for i, (key, (_, test_accuracies, rank_map, cfg)) in enumerate(hetero_results.items()):
        model_tmp = copy.deepcopy(hetero_model)
        model_tmp.apply_hetero_svd(
            calib_loader=train_loader,
            head_num=cfg['head_num'],
            budget_ratio=cfg['budget_ratio'],
            r_min=cfg['r_min'],
            max_calib_batches=5,
        )
        hetero_params = count_parameters(model_tmp)
        avg_rank = sum(rank_map.values()) / len(rank_map)
        ax_acc.plot(
            test_accuracies,
            label=f'Hetero-InherNet ({key}, avg_r={avg_rank:.1f}, {hetero_params:,} params)',
            color=colors[(i + len(results)) % len(colors)],
            linestyle=linestyles[(i + len(results)) % len(linestyles)],
            alpha=0.95,
        )

    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Test Accuracy (%)')
    ax_acc.set_title('Test Accuracy Comparison')
    ax_acc.grid(True, linestyle='--', alpha=0.35)

    handles, labels = ax_acc.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle('InherNet Variants: Optimization and Generalization Curves', y=1.02, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0.08, 1, 0.98])
    plt.savefig("result.png", dpi=300, bbox_inches='tight')
    plt.show()
