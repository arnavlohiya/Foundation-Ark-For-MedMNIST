from utils import MetricLogger, ProgressLogger
import time
import torch
from tqdm import tqdm
# import wandb

def train_one_epoch(model, use_head_n, dataset, data_loader_train, device, criterion, optimizer, epoch, ema_mode, teacher, momentum_schedule, coef_schedule, it):
    batch_time = MetricLogger('Time', ':6.3f')
    losses_cls = MetricLogger('Loss_'+dataset+' cls', ':.4e')
    losses_mse = MetricLogger('Loss_'+dataset+' mse', ':.4e')
    progress = ProgressLogger(
        len(data_loader_train),
        [batch_time, losses_cls, losses_mse],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    MSE = torch.nn.MSELoss()
    # coefficient scheduler from  0 to 0.5 
    coff = coef_schedule[it]
    print(coff)
    end = time.time()
    for i, (samples1, samples2, targets) in enumerate(data_loader_train):
        samples1, samples2, targets = samples1.float().to(device), samples2.float().to(device), targets.float().to(device)  # Move batch data to GPU
        """What happens:
        1. teacher(samples2, use_head_n) → triggers teacher.__call__(samples2, use_head_n)
        2. PyTorch's nn.Module.__call__ → calls teacher.forward(samples2, use_head_n)
        3. The ArkSwinTransformer.forward() method executes
        4. Returns two values: feat_t (internal features) and pred_t (final predictions)
        """
        feat_t, pred_t = teacher(samples2, use_head_n)
        feat_s, pred_s = model(samples1, use_head_n)
        loss_cls = criterion(pred_s, targets)
        loss_const = MSE(feat_s, feat_t)
        
        loss = (1-coff) * loss_cls + coff * loss_const

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_cls.update(loss_cls.item(), samples1.size(0))
        losses_mse.update(loss_const.item(), samples1.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            progress.display(i)

        if ema_mode == "iteration":
            ema_update_teacher(model, teacher, momentum_schedule, it)
            it += 1

    if ema_mode == "epoch":
        ema_update_teacher(model, teacher, momentum_schedule, it)
        it += 1

    # wandb.log({"train_loss_cls_{}".format(dataset): losses_cls.avg})
    # wandb.log({"train_loss_mse_{}".format(dataset): losses_mse.avg})


def ema_update_teacher(model, teacher, momentum_schedule, it):
    with torch.no_grad():
        m = momentum_schedule[it]  # momentum parameter
        for param_q, param_k in zip(model.parameters(), teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


def evaluate(model, use_head_n, data_loader_val, device, criterion, dataset):
    model.eval()

    with torch.no_grad():
        batch_time = MetricLogger('Time', ':6.3f')
        losses = MetricLogger('Loss', ':.4e')
        progress = ProgressLogger(
        len(data_loader_val),
        [batch_time, losses], prefix='Val_'+dataset+': ')

        end = time.time()
        for i, (samples, _, targets) in enumerate(data_loader_val):
            samples, targets = samples.float().to(device), targets.float().to(device)  # Move validation batch to GPU

            _, outputs = model(samples, use_head_n)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), samples.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                progress.display(i)

    return losses.avg


def test_classification(model, use_head_n, data_loader_test, device, multiclass=False, task_type="multi-label"):
    """
    MEDMNIST MODIFICATION: Enhanced test_classification function
    
    Now supports different MedMNIST task types with appropriate activation functions:
    - binary classification: sigmoid
    - multi-class classification: softmax 
    - multi-label classification: sigmoid
    - ordinal regression: softmax (treated as multi-class)
    
    Args:
        model: The neural network model
        use_head_n: Which classification head to use
        data_loader_test: Test data loader
        device: CUDA device
        multiclass: Legacy parameter for backwards compatibility
        task_type: MedMNIST task type string
    """ 
       
    model.eval()

    y_test = torch.FloatTensor().to(device)
    p_test = torch.FloatTensor().to(device)

    with torch.no_grad():
        for i, (samples, _, targets) in enumerate(tqdm(data_loader_test)):
            targets = targets.cuda()  # Move test targets to GPU
            y_test = torch.cat((y_test, targets), 0)

            if len(samples.size()) == 4:
                bs, c, h, w = samples.size()
                n_crops = 1
            elif len(samples.size()) == 5:
                bs, n_crops, c, h, w = samples.size()

            varInput = torch.autograd.Variable(samples.view(-1, c, h, w).to(device))  # Move test samples to GPU

            _, out = model(varInput, use_head_n)

            # MEDMNIST MODIFICATION: Handle different task types with appropriate activations
            # Different MedMNIST task types require different output activations
            if task_type == "multi-class classification":
                out = torch.softmax(out, dim=1)  # Multi-class: probabilities sum to 1
            elif task_type == "binary classification":
                out = torch.sigmoid(out)  # Binary: independent probability for each class
            elif task_type == "multi-label classification":
                out = torch.sigmoid(out)  # Multi-label: independent probabilities
            elif task_type == "ordinal regression":
                out = torch.softmax(out, dim=1)  # Ordinal: treat as multi-class for now
            else:
                # Fallback to original logic for backwards compatibility
                if multiclass:
                    out = torch.softmax(out, dim=1)
                else:
                    out = torch.sigmoid(out)
            outMean = out.view(bs, n_crops, -1).mean(1)
            p_test = torch.cat((p_test, outMean.data), 0)

    return y_test, p_test
    
