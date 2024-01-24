import torch


def test_model(model, test_dl, DEVICE):
    model.eval()
    correct = 0
    total = 0

    total_steps = len(test_dl)
    msg = ""
    for batch_num, (image_batch, label_batch) in enumerate(test_dl):
        #batch_sz = len(image_batch)
        label_batch = label_batch.to(DEVICE)
        image_batch = image_batch.to(DEVICE)
        output = model(image_batch)
        preds = torch.argmax(output, dim=1)
        correct += int(torch.eq(preds, label_batch).sum())
        total += label_batch.shape[0]
        if (batch_num + 1) % 5 == 0:
            print(" " * len(msg), end = '\r')
            msg = f'Testing batch[{batch_num + 1}/{total_steps}]'
            print (msg, end='\r' if batch_num < total_steps else "\n", flush=True)
    print(f"\nFinal test accuracy for {total} examples: {100 * correct/total:.5f}")

