import copy

class EarlyStopping():
  def __init__(self, patience=5, restore_best_weights=True):
    self.patience = patience
    self.restore_best_weights = restore_best_weights
    self.best_weights = None
    self.best_loss = None
    self.counter = 0
    
  def __call__(self, model, loss):
    if self.best_loss == None:
      self.best_loss = loss
      self.best_weights = copy.deepcopy(model.state_dict())
    else:
        delta = self.best_loss - loss
        if delta > 0:
          self.best_loss = loss
          self.counter = 0
          self.best_weights = copy.deepcopy(model.state_dict())
        else:
          self.counter += 1
          if self.counter >= self.patience:
            if self.restore_best_weights:
              model.load_state_dict(self.best_weights)
            return True

    return False
