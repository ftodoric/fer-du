import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch


def evaluate(model: torch.nn.Module, data, criterion):
    # Set state for evaluation
    model.eval()

    with torch.no_grad():
        losses = list()
        y_true = list()
        y_pred = list()
        for batch_num, batch in enumerate(data):
            logits = model.forward(batch[0]).squeeze(-1)
            y = batch[1].float()

            # Calculate loss
            loss = criterion(logits, y)
            losses.append(float(loss))

            y_true.extend(y.tolist())
            y_pred.extend(model.predict(batch[0]))

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(np.array(y_true, dtype=np.int32),
                              np.array(y_pred, dtype=np.int32))
        f1 = f1_score(y_true, y_pred)

        print(f"\tloss = {np.mean(losses)}")
        print(f"\taccuracy = {accuracy*100}")
        print(f"\tcm = {np.ndarray.tolist(cm)}")
        print(f"\tf1 = {f1*100}")

        return {"loss": np.mean(losses),
                "accuracy": accuracy*100,
                "cm": np.ndarray.tolist(cm),
                "f1": f1*100}
