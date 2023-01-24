import copy

class EarlyStopping():
  def __init__(self, patience=3, restore_best_weights=True):
    self.patience = patience
    self.restore_best_weights = restore_best_weights
    self.best_weights = None
    self.best_loss = None
    self.counter = 0
    
  def __call__(self, model, loss):
    if self.best_loss == None:
      self.best_loss = loss
      self.best_weights = copy.deepcopy(model.state_dict()) # save the initial weights
    else:
        delta = self.best_loss - loss
        if delta > 0: # improvement
          self.best_loss = loss
          self.counter = 0
          self.best_weights = copy.deepcopy(model.state_dict()) # update the best weights
        else: # no improvement
          self.counter += 1
          if self.counter >= self.patience:
            if self.restore_best_weights:
              model.load_state_dict(self.best_weights) # replace the current weights with the best
            return True # stop training

    return False # continue training
