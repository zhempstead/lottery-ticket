import torch
import torch.nn.utils.prune as prune

def train(model, train_loader, criterion, optimizer, num_epochs, callback):
    '''
    Train a model for the specified number of epochs.

    callback should be a function of (epoch, model, running_loss) -> None. It gets called once
    before training starts, and at the end of each epoch.
    '''
    callback(-1, model, -1)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch, data in enumerate(train_loader):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        num_batches = batch + 1

        callback(epoch, model, running_loss)
    return model

def test_classification(model, test_loader):
    '''
    Run image classification and report accuracy on test set
    '''
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')
    return correct / total

def prune_model(model, prune_amount, glob=False, include_modules=None, exclude_modules=None):
    '''
    Prune model by prune_amount. If prune_amount is an int, that many weights will be pruned. If it
    is a float, that fraction of weights in each module will be pruned. By default, all modules are
    pruned, but you can specify either a whitelist (include_modules) or a blacklist
    (exclude_modules).

    If glob is False, each module will prune its smallest weights according to prune_amount.
    Otherwise, the smallest weights across all modules will be pruned (so some modules may have more
    or less than prune_amount actually pruned).

    Note that this simply masks out the relevant weights.
    '''
    if include_modules is not None and exclude_modules is not None:
        raise ValueError("Only one of include_modules and exclude_modules can be specified")

    if exclude_modules is None:
        exclude_modules = []
    if include_modules is None:
        modules = [module for name, module in model.named_modules() if hasattr(module, 'weight') and name not in exclude_modules]
    else:
        modules = [module for name, module in model.named_modules() if name in include_modules]
    if glob:
        prune.global_unstructured(
            parameters=[(m, 'weight') for m in modules],
            pruning_method=prune.L1Unstructured,
            amount=prune_amount)
    else:
        for module in modules:
            prune.l1_unstructured(module, name="weight", amount=prune_amount)
