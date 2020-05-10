import torch
import config as cf

# Save Log Message
def saveLogMsg(msg):
    print(msg)
    with open(cf.LOG_PATH, "a") as log_file:
        log_file.write(msg + "\n")

# Track Best Model
def track_best_model(path, model, epoch, best_f1, val_f1, val_acc, val_loss, patience):
    if best_f1 >= val_f1:
        return best_f1, '', patience + 1
    state = {
        'epoch': epoch,
        'f1': val_f1,
        'acc': val_acc,
        'loss': val_loss,
        'model': model.state_dict()
    }
    torch.save(state, path)
    return val_f1, ' *** ', 0
